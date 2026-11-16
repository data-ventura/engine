//! Golden-image tests: reference charts in both themes, drawn through every export path and
//! compared with PNGs checked in under `tests/golden/`.
//!
//! - **SVG:** the SVG export rasterized by resvg. This is how the goldens are made, so it must
//!   match them almost exactly.
//! - **PDF:** the PDF export rasterized by hayro, an independent PDF renderer. It must match the
//!   goldens within the tolerance for a different rasterizer, and its bytes must equal the golden
//!   PDF (the export is deterministic). pdfium's text extraction of the golden PDFs, in
//!   `fixtures/reference/pdf-text.json`, must find every label as text where the layout put it.
//! - **Screen and PNG export:** vello on the GPU, read back. Skipped when no GPU is available (CI
//!   runners have none), so the CPU paths above are what CI checks.
//!
//! Images are compared perceptually: each 1 pt cell (scale × scale pixels) is averaged, and a cell
//! counts as different when the CIEDE2000 difference of the averages exceeds [`CELL_DELTA_E`].
//! Averaging over a point cancels rasterizers' different edge anti-aliasing but keeps geometry: a
//! label or line one pixel off still fails.
//!
//! To update the goldens after an intended change, run
//! `DV_UPDATE_GOLDENS=1 cargo test -p dataventura-render --test golden`, look at the new PNGs, and
//! regenerate the text reference with `scripts/reference/pdf-text.py`. On a failure, the images
//! drawn and a diff (differing cells in magenta) go to `target/golden/`.

use std::path::PathBuf;

use dv_render::color::{delta_e2000, linear_to_lab, srgb_to_linear};
use dv_render::gpu::Gpu;
use dv_render::pdf::PdfScene;
use dv_render::plot::chart::{self, BarChart, DARK, Frame, LIGHT, Layout, LineChart, Theme};
use dv_render::plot::format::NumberFormat;
use dv_render::text::Fonts;
use dv_render::vector::{Scene, SvgScene, VectorRenderer, VelloScene, decode_png, encode_png};

const FRAME: Frame = Frame {
    width: 640.0,
    height: 360.0,
    scale: 2.0,
};
/// A cell differs when its average color is further than this (CIEDE2000) from the golden's.
const CELL_DELTA_E: f64 = 6.0;
/// Share of cells allowed to differ: the goldens' own rasterizer, and another one (hayro, vello).
/// Other rasterizers differ in under 0.04% of cells on the goldens; a chart one pixel off differs
/// in several percent (`tolerance_catches_a_pixel_shift`).
const SAME_RASTERIZER: f64 = 0.000_1;
const OTHER_RASTERIZER: f64 = 0.002;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn out_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/golden")
}

fn updating() -> bool {
    std::env::var_os("DV_UPDATE_GOLDENS").is_some()
}

/// A tiny deterministic generator, so the charts' data needs no fixture.
fn noise(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

/// A year of daily temperatures: a seasonal cycle with weather on top, and a daily range.
fn line() -> LineChart {
    let mut s = 0x9e37_79b9_7f4a_7c15;
    let (mut t, mut mean, mut lo, mut hi) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut weather = 0.0;
    for d in 0..365 {
        weather = 0.8 * weather + 2.5 * noise(&mut s);
        let m = 12.0 - 11.0 * (2.0 * std::f64::consts::PI * (f64::from(d) - 20.0) / 365.0).cos()
            + weather;
        let half = 4.0 + 1.5 * noise(&mut s);
        // 2023-01-01, noon UTC.
        t.push(1_672_531_200.0 + f64::from(d) * 86_400.0 + 43_200.0);
        mean.push(m);
        lo.push(m - half);
        hi.push(m + half);
    }
    LineChart {
        title: "Air temperature, 2023".into(),
        subtitle: "°C at 2 m · daily mean and range · synthetic".into(),
        t,
        line: mean,
        line_label: "Daily mean".into(),
        band: Some((lo, hi)),
        band_label: "Daily range".into(),
    }
}

/// Counts by hour of day with a morning and an evening peak.
fn bars() -> BarChart {
    let values = (0..24)
        .map(|h| {
            let h = f64::from(h);
            let peak = |c: f64, w: f64| (-((h - c) / w).powi(2)).exp();
            (20.0 + 120.0 * peak(8.5, 2.0) + 160.0 * peak(18.0, 3.0)).round()
        })
        .collect();
    BarChart {
        title: "Trips by hour of day".into(),
        subtitle: "Thousands of trips · synthetic".into(),
        categories: (0..24).map(|h| format!("{h:02}")).collect(),
        values,
    }
}

fn draw(scene: &mut impl Scene, fonts: &mut Fonts, chart: &str, theme: Theme) -> Layout {
    let numbers = NumberFormat::locale("en-US").unwrap();
    match chart {
        "line" => chart::line_chart(scene, fonts, &line(), FRAME, theme, &numbers),
        _ => chart::bar_chart(scene, fonts, &bars(), FRAME, theme, &numbers),
    }
}

const CASES: [(&str, &str, Theme); 4] = [
    ("line", "light", LIGHT),
    ("line", "dark", DARK),
    ("bars", "light", LIGHT),
    ("bars", "dark", DARK),
];

fn svg(chart: &str, theme: Theme) -> String {
    let size = FRAME.pixels();
    let mut s = SvgScene::new(size[0], size[1], FRAME.scale, theme.background);
    draw(&mut s, &mut Fonts::new(), chart, theme);
    s.finish()
}

fn pdf(chart: &str, theme: Theme) -> (Vec<u8>, Layout) {
    let size = FRAME.pixels();
    let mut s = PdfScene::new(size[0], size[1], FRAME.scale, theme.background);
    let layout = draw(&mut s, &mut Fonts::new(), chart, theme);
    (s.finish().unwrap(), layout)
}

fn rasterize_svg(svg: &str) -> Vec<u8> {
    let size = FRAME.pixels();
    let tree = resvg::usvg::Tree::from_str(svg, &resvg::usvg::Options::default()).unwrap();
    let mut pixmap = resvg::tiny_skia::Pixmap::new(size[0], size[1]).unwrap();
    let s = size[0] as f32 / tree.size().width();
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(s, s),
        &mut pixmap.as_mut(),
    );
    // The background is opaque, so premultiplied and straight alpha agree.
    pixmap.take()
}

fn rasterize_pdf(pdf: Vec<u8>) -> Vec<u8> {
    let pdf = hayro::hayro_syntax::Pdf::new(pdf).unwrap();
    let pages = pdf.pages();
    let pixmap = hayro::render(
        &pages[0],
        &hayro::RenderCache::new(),
        &hayro::hayro_interpret::InterpreterSettings::default(),
        &hayro::RenderSettings {
            // The other paths draw unhinted outlines.
            hinting: false,
            ..Default::default()
        },
        &hayro::PixmapSettings {
            x_scale: FRAME.scale as f32,
            y_scale: FRAME.scale as f32,
            bg_color: hayro::vello_cpu::color::palette::css::WHITE,
        },
    );
    assert_eq!(
        [u32::from(pixmap.width()), u32::from(pixmap.height())],
        FRAME.pixels()
    );
    pixmap.data_as_u8_slice().to_vec()
}

/// The share of 1 pt cells further than [`CELL_DELTA_E`] from the golden, the worst cell's
/// difference, and the golden with differing cells in magenta.
fn cell_diff(got: &[u8], golden: &[u8]) -> (f64, f64, Vec<u8>) {
    let size = FRAME.pixels();
    let (w, h, k) = (size[0] as usize, size[1] as usize, FRAME.scale as usize);
    let cell = |img: &[u8], cx: usize, cy: usize| {
        let mut sum = [0.0; 3];
        for y in cy * k..(cy + 1) * k {
            for x in cx * k..(cx + 1) * k {
                for (c, s) in sum.iter_mut().enumerate() {
                    *s += srgb_to_linear(f64::from(img[(y * w + x) * 4 + c]) / 255.0);
                }
            }
        }
        linear_to_lab(sum.map(|s| s / (k * k) as f64))
    };
    let mut diff = golden.to_vec();
    let (mut differ, mut worst, mut n) = (0usize, 0.0_f64, 0usize);
    for cy in 0..h / k {
        for cx in 0..w / k {
            let d = delta_e2000(cell(got, cx, cy), cell(golden, cx, cy));
            worst = worst.max(d);
            n += 1;
            if d > CELL_DELTA_E {
                differ += 1;
                for y in cy * k..(cy + 1) * k {
                    for x in cx * k..(cx + 1) * k {
                        diff[(y * w + x) * 4..][..4].copy_from_slice(&[255, 0, 255, 255]);
                    }
                }
            }
        }
    }
    (differ as f64 / n as f64, worst, diff)
}

/// Fails if more than `allowed` of the cells differ, writing the image and a diff to
/// `target/golden/`.
fn compare(name: &str, got: &[u8], golden: &[u8], allowed: f64) -> Result<(), String> {
    let size = FRAME.pixels();
    let (share, worst, diff) = cell_diff(got, golden);
    println!(
        "{name}: {:.4}% of cells differ, worst ΔE {worst:.1}",
        share * 100.0
    );
    if share <= allowed {
        return Ok(());
    }
    let dir = out_dir();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join(format!("{name}.png")),
        encode_png(got, size).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join(format!("{name}-diff.png")),
        encode_png(&diff, size).unwrap(),
    )
    .unwrap();
    Err(format!(
        "{name}: {:.3}% of cells differ from the golden by more than ΔE {CELL_DELTA_E} (allowed {:.3}%); see {}",
        share * 100.0,
        allowed * 100.0,
        dir.display()
    ))
}

fn golden(chart: &str, theme: &str) -> Vec<u8> {
    let path = golden_dir().join(format!("{chart}-{theme}.png"));
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (pixels, size) = decode_png(&data).unwrap();
    assert_eq!(size, FRAME.pixels(), "{}", path.display());
    pixels
}

/// Writes the goldens when updating; otherwise checks the SVG and PDF paths against them.
#[test]
fn exports_match_goldens() {
    let mut failures = Vec::new();
    for (chart, theme_name, theme) in CASES {
        let name = format!("{chart}-{theme_name}");
        let svg_pixels = rasterize_svg(&svg(chart, theme));
        let (pdf_bytes, _) = pdf(chart, theme);
        if updating() {
            let dir = golden_dir();
            std::fs::create_dir_all(&dir).unwrap();
            let png = encode_png(&svg_pixels, FRAME.pixels()).unwrap();
            std::fs::write(dir.join(format!("{name}.png")), png).unwrap();
            std::fs::write(dir.join(format!("{name}.pdf")), &pdf_bytes).unwrap();
            continue;
        }
        let want = golden(chart, theme_name);
        let pdf_pixels = rasterize_pdf(pdf_bytes.clone());
        for result in [
            compare(&format!("{name}-svg"), &svg_pixels, &want, SAME_RASTERIZER),
            compare(&format!("{name}-pdf"), &pdf_pixels, &want, OTHER_RASTERIZER),
        ] {
            failures.extend(result.err());
        }
        let golden_pdf = std::fs::read(golden_dir().join(format!("{name}.pdf"))).unwrap();
        if pdf_bytes != golden_pdf {
            failures.push(format!("{name}: the PDF export differs from {name}.pdf"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The tolerance still catches geometry: every golden shifted right by one pixel fails it.
#[test]
fn tolerance_catches_a_pixel_shift() {
    if updating() {
        return;
    }
    let w = FRAME.pixels()[0] as usize;
    for (chart, theme_name, _) in CASES {
        let want = golden(chart, theme_name);
        let mut shifted = want.clone();
        for row in shifted.chunks_mut(w * 4) {
            row.copy_within(..(w - 1) * 4, 4);
        }
        let (share, _, _) = cell_diff(&shifted, &want);
        assert!(
            share > 5.0 * OTHER_RASTERIZER,
            "{chart}-{theme_name}: a one-pixel shift differs in only {:.3}% of cells",
            share * 100.0
        );
    }
}

/// The screen path (and PNG export) against the same goldens.
#[test]
fn gpu_matches_goldens() {
    let Ok(gpu) = Gpu::headless() else {
        eprintln!("no GPU: skipped");
        return;
    };
    if updating() {
        return;
    }
    let mut renderer = VectorRenderer::new(&gpu).unwrap();
    let mut failures = Vec::new();
    for (chart, theme_name, theme) in CASES {
        let mut scene = VelloScene::default();
        draw(&mut scene, &mut Fonts::new(), chart, theme);
        let size = FRAME.pixels();
        let texture = renderer
            .render(&gpu, &scene, size, theme.background)
            .unwrap();
        let png = encode_png(&gpu.read_rgba(&texture, size), size).unwrap();
        let (pixels, _) = decode_png(&png).unwrap();
        let name = format!("{chart}-{theme_name}-gpu");
        failures
            .extend(compare(&name, &pixels, &golden(chart, theme_name), OTHER_RASTERIZER).err());
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Every label the layout drew is found, as text, inside its box by pdfium's text extraction of
/// the golden PDF, and pdfium finds no other text.
#[test]
fn pdf_text_matches_pdfium() {
    if updating() {
        return;
    }
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference/pdf-text.json");
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for (chart, theme_name, theme) in CASES {
        let name = format!("{chart}-{theme_name}");
        let page = &reference[&name];
        let (w, h) = (
            page["width"].as_f64().unwrap(),
            page["height"].as_f64().unwrap(),
        );
        assert_eq!((w, h), (FRAME.width, FRAME.height), "{name}");
        // Character centers in device pixels, y down.
        let mut chars: Vec<(char, f64, f64, bool)> = page["chars"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let b: Vec<f64> = c["box"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                let ch = c["char"].as_str().unwrap().chars().next().unwrap();
                let x = (b[0] + b[2]) / 2.0 * FRAME.scale;
                let y = (h - (b[1] + b[3]) / 2.0) * FRAME.scale;
                (ch, x, y, false)
            })
            .filter(|c| !c.0.is_whitespace())
            .collect();
        let (_, layout) = pdf(chart, theme);
        for (label, r) in &layout.labels {
            let inside: String = chars
                .iter_mut()
                .filter(|c| !c.3 && r.inflate(1.0, 1.0).contains((c.1, c.2)))
                .map(|c| {
                    c.3 = true;
                    c.0
                })
                .collect();
            let want: String = label.chars().filter(|c| !c.is_whitespace()).collect();
            assert_eq!(inside, want, "{name}: label at {r:?}");
        }
        let stray: String = chars.iter().filter(|c| !c.3).map(|c| c.0).collect();
        assert!(
            stray.is_empty(),
            "{name}: text outside every label: {stray}"
        );
    }
}
