//! S5: vello-drawn axes and labels match a Swift Charts reference side by side, and the SVG export
//! matches the screen.
//!
//! Two charts at 720 × 400 pt, drawn at 2× (a Retina display):
//! - **Line and band:** New York City's daily mean 2 m temperature in 2019, with the daily range as a
//!   band, from S9's hourly ERA5 series.
//! - **Bars:** NYC taxi pickups by hour of day, Q1 2015 (S1–S3's 38M rows), in thousands.
//!
//! Each chart is drawn by vello to a PNG, and through the same calls to SVG. The SVG is rasterized
//! with resvg at the same pixel size and compared with the vello image. The run also checks that
//! no two text boxes overlap and that every hairline covers exactly one pixel row. The chart data
//! goes to `target/spike/s5-data.json`, so the same charts can be drawn by Swift Charts for the
//! side-by-side comparison.

use dv_render::gpu::Gpu;
use dv_render::plot::chart::{self, BarChart, Frame, LIGHT, Layout, LineChart};
use dv_render::plot::format::NumberFormat;
use dv_render::text::Fonts;
use dv_render::vector::{Rect, Rgba, SvgScene, VectorRenderer, VelloScene};
use dv_spike::taxi::{HOUR_SHIFT, TaxiPoints};
use dv_spike::{Caches, Report, read_rgba, workspace_root, write_png};

const FRAME: Frame = Frame {
    width: 720.0,
    height: 400.0,
    scale: 2.0,
};
/// Seconds from 1900-01-01 to 1970-01-01.
const EPOCH_1900: f64 = 2_208_988_800.0;
/// A 1 pt cell (2 × 2 pixels, averaged) counts as different between the SVG and the screen above
/// this, in any channel. Averaging over a point cancels the two rasterizers' different edge
/// anti-aliasing (vello computes exact coverage, resvg samples it) but keeps geometry: the run
/// reports how the same measure responds to an SVG off by one pixel.
const CELL_TOLERANCE: f64 = 16.0;

fn main() {
    let mut report = Report::new("S5", Caches::Warm);
    let (line, bars) = (temperature_chart(), trips_chart());
    let size = FRAME.pixels();
    let gpu = Gpu::headless().expect("GPU");
    let mut renderer = VectorRenderer::new(&gpu).expect("vello");
    let mut fonts = Fonts::new();
    // The Swift Charts reference renders in the machine's locale, en-US; compare like with like.
    let numbers = NumberFormat::locale("en-US").expect("en-US");

    for name in ["line", "bars"] {
        let draw_vello = |fonts: &mut Fonts| {
            let mut scene = VelloScene::default();
            let layout = match name {
                "line" => chart::line_chart(&mut scene, fonts, &line, FRAME, LIGHT, &numbers),
                _ => chart::bar_chart(&mut scene, fonts, &bars, FRAME, LIGHT, &numbers),
            };
            (scene, layout)
        };
        let t = std::time::Instant::now();
        let (scene, layout) = draw_vello(&mut fonts);
        let build_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = std::time::Instant::now();
        let texture = renderer
            .render(&gpu, &scene, size, LIGHT.background)
            .expect("render");
        let screen = read_rgba(&gpu, &texture, size);
        let render_ms = t.elapsed().as_secs_f64() * 1e3;
        write_png(&screen, size, &format!("s5-dv-{name}.png"));
        report.metric(
            &format!("{name}.build_scene"),
            build_ms,
            "ms",
            None,
            "layout, text shaping and scene encoding",
        );
        report.metric(
            &format!("{name}.render_and_read_back"),
            render_ms,
            "ms",
            None,
            "first render includes vello's shader setup",
        );

        let mut svg = SvgScene::new(size[0], size[1], FRAME.scale, LIGHT.background);
        match name {
            "line" => chart::line_chart(&mut svg, &mut fonts, &line, FRAME, LIGHT, &numbers),
            _ => chart::bar_chart(&mut svg, &mut fonts, &bars, FRAME, LIGHT, &numbers),
        };
        let svg = svg.finish();
        let svg_path = workspace_root().join(format!("target/spike/s5-dv-{name}.svg"));
        std::fs::write(&svg_path, &svg).expect("write svg");
        let raster = rasterize(&svg, size);
        write_png(&raster, size, &format!("s5-dv-{name}-svg.png"));

        let (differ, mean) = compare(&screen, &raster, size, 0);
        let (shifted, _) = compare(&screen, &raster, size, 1);
        report.metric(
            &format!("{name}.svg_cells_differing"),
            differ * 100.0,
            "%",
            Some(0.1),
            &format!(
                "SVG rasterized by resvg vs the vello image, 1 pt cells off by more than {CELL_TOLERANCE}/255; mean {mean:.2}/255; an SVG one pixel off would give {:.2}%; SVG {} KB",
                shifted * 100.0,
                svg.len() / 1000
            ),
        );
        let raw = screen
            .chunks(4)
            .zip(raster.chunks(4))
            .filter(|(a, b)| a.iter().zip(*b).take(3).any(|(x, y)| x.abs_diff(*y) > 8))
            .count() as f64
            / f64::from(size[0] * size[1]);
        report.metric(
            &format!("{name}.svg_pixels_differing_raw"),
            raw * 100.0,
            "%",
            None,
            "single pixels off by more than 8/255: edge anti-aliasing, recorded",
        );
        let overlaps = overlapping(&layout.labels);
        report.metric(
            &format!("{name}.labels_overlapping"),
            overlaps.len() as f64,
            "pairs",
            Some(0.5),
            &format!(
                "{overlaps:?} {} text boxes; x labels kept {} of {}",
                layout.labels.len(),
                layout.x_ticks.iter().filter(|t| !t.2.is_empty()).count(),
                layout.x_ticks.len()
            ),
        );
        let (crisp, checked) = hairlines_crisp(&screen, size, &layout);
        report.metric(
            &format!("{name}.hairlines_not_crisp"),
            (checked - crisp) as f64,
            "lines",
            Some(0.5),
            &format!(
                "{crisp} of {checked} horizontal hairlines cover exactly one pixel row at full strength, with clean rows on both sides"
            ),
        );
        let ticks: Vec<&str> = layout.y_ticks.iter().map(|t| t.2.as_str()).collect();
        let xs: Vec<&str> = layout
            .x_ticks
            .iter()
            .map(|t| t.2.as_str())
            .filter(|s| !s.is_empty())
            .collect();
        println!("{name}: y ticks {ticks:?}; x labels {xs:?}");
    }

    let data = serde_json::json!({
        "width": FRAME.width, "height": FRAME.height, "scale": FRAME.scale,
        "line": {
            "title": line.title, "subtitle": line.subtitle,
            "t": line.t, "mean": line.line,
            "min": line.band.as_ref().map(|b| &b.0), "max": line.band.as_ref().map(|b| &b.1),
            "line_label": line.line_label, "band_label": line.band_label,
        },
        "bars": {
            "title": bars.title, "subtitle": bars.subtitle,
            "categories": bars.categories, "values": bars.values,
        },
    });
    std::fs::write(
        workspace_root().join("target/spike/s5-data.json"),
        serde_json::to_string(&data).expect("json"),
    )
    .expect("write data");
    std::process::exit(if report.finish() { 0 } else { 1 });
}

/// Daily mean and range of S9's hourly 2019 series at New York, in °C.
fn temperature_chart() -> LineChart {
    let path = workspace_root().join("target/spike/s9-series.json");
    let s9: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&path).expect("run S9 first: target/spike/s9-series.json"),
    )
    .expect("json");
    let first = s9["first_hour_since_1900"].as_f64().expect("first hour") * 3600.0 - EPOCH_1900;
    let kelvin: Vec<f64> = s9["kelvin"]
        .as_array()
        .expect("values")
        .iter()
        .map(|v| v.as_f64().expect("value"))
        .collect();
    let (mut t, mut mean, mut lo, mut hi) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (d, day) in kelvin.chunks(24).enumerate() {
        let c: Vec<f64> = day.iter().map(|k| k - 273.15).collect();
        t.push(first + d as f64 * 86_400.0 + 43_200.0);
        mean.push(c.iter().sum::<f64>() / c.len() as f64);
        lo.push(c.iter().copied().fold(f64::INFINITY, f64::min));
        hi.push(c.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    }
    LineChart {
        title: "New York City air temperature, 2019".into(),
        subtitle: "°C at 2 m · daily mean and range (UTC days) · ERA5 reanalysis".into(),
        t,
        line: mean,
        line_label: "Daily mean".into(),
        band: Some((lo, hi)),
        band_label: "Daily range".into(),
    }
}

/// Taxi pickups by hour of day, in thousands.
fn trips_chart() -> BarChart {
    let taxi = TaxiPoints::load();
    let mut counts = [0u64; 24];
    for a in &taxi.attrs {
        counts[((a >> HOUR_SHIFT) & 31) as usize % 24] += 1;
    }
    BarChart {
        title: "Taxi pickups by hour of day".into(),
        subtitle: format!(
            "Thousands of trips · New York City yellow cabs · Q1 2015 · {:.1}M trips",
            taxi.attrs.len() as f64 / 1e6
        ),
        categories: (0..24).map(|h| format!("{h:02}")).collect(),
        values: counts.iter().map(|&c| c as f64 / 1000.0).collect(),
    }
}

fn rasterize(svg: &str, size: [u32; 2]) -> Vec<u8> {
    let tree =
        resvg::usvg::Tree::from_str(svg, &resvg::usvg::Options::default()).expect("parse svg");
    let mut pixmap = resvg::tiny_skia::Pixmap::new(size[0], size[1]).expect("pixmap");
    let s = size[0] as f32 / tree.size().width();
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(s, s),
        &mut pixmap.as_mut(),
    );
    // The background is opaque, so premultiplied and straight alpha agree.
    pixmap.take()
}

/// The share of 1 pt cells (2 × 2 pixels, averaged) with any channel off by more than
/// [`CELL_TOLERANCE`], and the mean difference, with `b` shifted right by `shift` pixels.
fn compare(a: &[u8], b: &[u8], size: [u32; 2], shift: usize) -> (f64, f64) {
    let (w, h) = (size[0] as usize, size[1] as usize);
    let px = |img: &[u8], x: usize, y: usize, c: usize, s: usize| {
        f64::from(img[(y * w + x.saturating_sub(s)) * 4 + c])
    };
    let (mut differ, mut total, mut n) = (0usize, 0.0, 0usize);
    for cy in 0..h / 2 {
        for cx in 0..w / 2 {
            let mut worst = 0.0_f64;
            for c in 0..3 {
                let mut d = 0.0;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (x, y) = (cx * 2 + dx, cy * 2 + dy);
                    d += px(a, x, y, c, 0) - px(b, x, y, c, shift);
                }
                worst = worst.max((d / 4.0).abs());
            }
            total += worst;
            n += 1;
            if worst > CELL_TOLERANCE {
                differ += 1;
            }
        }
    }
    (differ as f64 / n as f64, total / n as f64)
}

fn overlapping(labels: &[(String, Rect)]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (i, (a, ra)) in labels.iter().enumerate() {
        for (b, rb) in &labels[i + 1..] {
            let inter = ra.intersect(*rb);
            if inter.width() > 0.5 && inter.height() > 0.5 {
                out.push((a.clone(), b.clone()));
            }
        }
    }
    out
}

/// Counts horizontal hairlines drawn as exactly one pixel row: along the line, wherever nothing else
/// is drawn within two rows, the line's row has the hairline's full color over the background and
/// the rows next to it are pure background. A line straddling two rows would fail on both counts.
fn hairlines_crisp(pixels: &[u8], size: [u32; 2], layout: &Layout) -> (usize, usize) {
    let bg = LIGHT.background.0;
    let at = |x: usize, y: usize| {
        let i = (y * size[0] as usize + x) * 4;
        [pixels[i], pixels[i + 1], pixels[i + 2]]
    };
    let close =
        |p: [u8; 3], q: [u8; 3], tol: u8| p.iter().zip(q).all(|(a, b)| a.abs_diff(b) <= tol);
    let over_bg = |c: Rgba| -> [u8; 3] {
        let a = f64::from(c.0[3]) / 255.0;
        std::array::from_fn(|i| {
            (f64::from(c.0[i]) * a + f64::from(bg[i]) * (1.0 - a)).round() as u8
        })
    };
    let (mut crisp, mut checked) = (0, 0);
    let plot = layout.plot;
    for &(row, color) in &layout.hairlines {
        let y = row.floor() as usize;
        if y < 2 || y + 2 >= size[1] as usize {
            continue;
        }
        let want = over_bg(color);
        let (mut clean_cols, mut good) = (0, 0);
        for x in (plot.x0.ceil() as usize + 2)..(plot.x1.floor() as usize).saturating_sub(2) {
            let clear = [y - 2, y + 2]
                .iter()
                .all(|&r| close(at(x, r), [bg[0], bg[1], bg[2]], 0));
            if !clear {
                continue;
            }
            clean_cols += 1;
            let neighbors = [y - 1, y + 1]
                .iter()
                .all(|&r| close(at(x, r), [bg[0], bg[1], bg[2]], 1));
            if neighbors && close(at(x, y), want, 2) {
                good += 1;
            }
        }
        if clean_cols > 0 {
            checked += 1;
            // Vertical gridlines cross the row at a few columns; allow for them.
            if good * 100 >= clean_cols * 97 {
                crisp += 1;
            }
        }
    }
    (crisp, checked)
}
