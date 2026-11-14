//! Chart views drawn through the vector [`Scene`]: a line chart with a band, and a bar chart.
//!
//! The layout follows the chart quality bar: a 4 pt grid with 16 pt margins; a title at 1.2× the
//! label size with a subtitle; faint gridlines one device pixel wide, snapped to pixel centers;
//! Wilkinson ticks with the value axis ending on ticks; date axes with context shown once; tick
//! labels that never overlap (a collision pass thins them instead of rotating them); and direct
//! labels at the end of each series, nudged apart.
//!
//! Sizes are in points; `scale` device pixels per point. The returned [`Layout`] records where
//! everything went, in device pixels, for tests.

use super::format::NumberFormat;
use super::ticks::extended;
use super::time::time_ticks;
use crate::text::{Fonts, TextBlock, Weight};
use crate::vector::{BezPath, Point, Rect, Rgba, Scene, Shape};

/// Colors for one theme.
#[derive(Debug, Clone, Copy)]
pub struct Theme {
    pub background: Rgba,
    pub text: Rgba,
    pub secondary: Rgba,
    pub grid: Rgba,
    /// The zero line and the baseline of bars.
    pub axis: Rgba,
    pub series: Rgba,
}

pub const LIGHT: Theme = Theme {
    background: Rgba::rgb(255, 255, 255),
    text: Rgba::rgb(29, 29, 31),
    secondary: Rgba::rgb(110, 110, 115),
    grid: Rgba::rgb(0, 0, 0).with_alpha(20),
    axis: Rgba::rgb(0, 0, 0).with_alpha(90),
    series: Rgba::rgb(37, 99, 235),
};

/// Points.
const PAD: f64 = 16.0;
const LABEL: f32 = 11.0;
const TITLE: f32 = LABEL * 1.2;
const LABEL_GAP: f64 = 8.0;
/// Minimum space between two tick labels.
const LABEL_SPACE: f64 = 8.0;
const LINE_WIDTH: f64 = 2.0;
const BAND_ALPHA: u8 = 46;

/// The size of a chart.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub width: f64,
    pub height: f64,
    pub scale: f64,
}

impl Frame {
    pub fn pixels(&self) -> [u32; 2] {
        [
            (self.width * self.scale).round() as u32,
            (self.height * self.scale).round() as u32,
        ]
    }
}

/// A line over time, with an optional band (such as a daily range) behind it.
#[derive(Debug, Clone)]
pub struct LineChart {
    pub title: String,
    pub subtitle: String,
    /// Seconds since the epoch.
    pub t: Vec<f64>,
    pub line: Vec<f64>,
    pub line_label: String,
    pub band: Option<(Vec<f64>, Vec<f64>)>,
    pub band_label: String,
}

/// One bar per category.
#[derive(Debug, Clone)]
pub struct BarChart {
    pub title: String,
    pub subtitle: String,
    pub categories: Vec<String>,
    pub values: Vec<f64>,
}

/// Where things were drawn, in device pixels.
#[derive(Debug, Clone, Default)]
pub struct Layout {
    pub plot: Rect,
    /// Value-axis ticks: value, pixel row of the gridline's center, label.
    pub y_ticks: Vec<(f64, f64, String)>,
    /// Category or time ticks: position, pixel column, label (empty if thinned out).
    pub x_ticks: Vec<(f64, f64, String)>,
    /// Every text box drawn, with its text.
    pub labels: Vec<(String, Rect)>,
    /// Horizontal hairlines: pixel row of the center, and color.
    pub hairlines: Vec<(f64, Rgba)>,
}

struct Ctx<'a, S: Scene> {
    scene: &'a mut S,
    fonts: &'a mut Fonts,
    numbers: &'a NumberFormat,
    frame: Frame,
    theme: Theme,
    layout: Layout,
}

impl<S: Scene> Ctx<'_, S> {
    fn px(&self, pt: f64) -> f64 {
        pt * self.frame.scale
    }

    fn text(&mut self, s: &str, size: f32, weight: Weight) -> TextBlock {
        self.fonts.layout(s, size * self.frame.scale as f32, weight)
    }

    /// Draws `block` with its top-left corner near `(x, y)` pixels, the baseline on a whole pixel.
    fn draw_text(&mut self, s: &str, block: &TextBlock, x: f64, y: f64, color: Rgba) -> Rect {
        let baseline = (y + f64::from(block.baseline)).round();
        let top = baseline - f64::from(block.baseline);
        for run in block.at(x as f32, top as f32) {
            self.scene.glyphs(&run, color);
        }
        let r = Rect::new(
            x,
            top,
            x + f64::from(block.width),
            top + f64::from(block.height),
        );
        self.layout.labels.push((s.to_string(), r));
        r
    }

    /// A horizontal hairline one device pixel thick across `x0..x1`, centered on a pixel row.
    fn hairline(&mut self, y: f64, x0: f64, x1: f64, color: Rgba) {
        let row = y.floor() + 0.5;
        let mut p = BezPath::new();
        p.move_to(Point::new(x0.round(), row));
        p.line_to(Point::new(x1.round(), row));
        self.scene.stroke(&p, 1.0, color, false);
        self.layout.hairlines.push((row, color));
    }

    /// A vertical hairline one device pixel thick, centered on a pixel column.
    fn vline(&mut self, x: f64, y0: f64, y1: f64, color: Rgba) {
        let col = x.floor() + 0.5;
        let mut p = BezPath::new();
        p.move_to(Point::new(col, y0.round()));
        p.line_to(Point::new(col, y1.round()));
        self.scene.stroke(&p, 1.0, color, false);
    }

    /// Title and subtitle at the top left; returns the pixel row below them.
    fn header(&mut self, title: &str, subtitle: &str) -> f64 {
        let (pad, theme) = (self.px(PAD), self.theme);
        let t = self.text(title, TITLE, Weight::Medium);
        let r = self.draw_text(title, &t, pad, pad, theme.text);
        let s = self.text(subtitle, LABEL, Weight::Regular);
        let r = self.draw_text(subtitle, &s, pad, r.y1 + self.px(2.0), theme.secondary);
        r.y1
    }

    /// Value-axis ticks for `lo..=hi` over `height` pixels: about one per 48 pt, ending on ticks.
    fn value_ticks(&self, lo: f64, hi: f64, height: f64) -> Vec<f64> {
        let m = ((height / self.px(48.0)).round() as usize).clamp(2, 10);
        extended(lo, hi, m, true)
    }

    /// Y labels right-aligned against the plot, and gridlines; returns the label column width.
    fn y_axis(&mut self, ticks: &[f64], plot: Rect, y_of: impl Fn(f64) -> f64, zero_line: bool) {
        let theme = self.theme;
        for ((v, label), block) in ticks
            .iter()
            .zip(self.numbers.tick_labels(ticks))
            .zip(self.y_label_blocks(ticks))
        {
            let y = y_of(*v);
            let color = if zero_line && *v == 0.0 {
                theme.axis
            } else {
                theme.grid
            };
            self.hairline(y, plot.x0, plot.x1, color);
            let x = plot.x0 - self.px(LABEL_GAP) - f64::from(block.width);
            let top = y - f64::from(block.height) / 2.0;
            self.draw_text(&label, &block, x, top, theme.secondary);
            self.layout.y_ticks.push((*v, y.floor() + 0.5, label));
        }
    }

    fn y_label_blocks(&mut self, ticks: &[f64]) -> Vec<TextBlock> {
        self.numbers
            .tick_labels(ticks)
            .iter()
            .map(|l| self.text(l, LABEL, Weight::Regular))
            .collect()
    }

    /// X labels under their ticks (centered, or starting just after the tick for dates, whose
    /// labels name the period that begins there), kept inside the frame. Labels that would come
    /// closer than [`LABEL_SPACE`] to a kept one are dropped, keeping every n-th label for the
    /// smallest n that fits, so the survivors stay evenly spaced.
    fn x_labels(&mut self, ticks: &[(f64, f64, String)], top: f64, start: bool) {
        let theme = self.theme;
        let blocks: Vec<TextBlock> = ticks
            .iter()
            .map(|t| self.text(&t.2, LABEL, Weight::Regular))
            .collect();
        let (min_x, max_x) = (self.px(PAD), self.px(self.frame.width - PAD));
        let nudge = self.px(3.0);
        let place = |i: usize| {
            let w = f64::from(blocks[i].width);
            let x = if start {
                ticks[i].1 + nudge
            } else {
                ticks[i].1 - w / 2.0
            };
            x.clamp(min_x, max_x - w)
        };
        let space = self.px(LABEL_SPACE);
        let every = (1..=ticks.len().max(1))
            .find(|&n| {
                let kept: Vec<usize> = (0..ticks.len()).step_by(n).collect();
                kept.windows(2)
                    .all(|w| place(w[0]) + f64::from(blocks[w[0]].width) + space <= place(w[1]))
            })
            .unwrap_or(1);
        for (i, t) in ticks.iter().enumerate() {
            let label = if i % every == 0 {
                let x = place(i);
                self.draw_text(&t.2, &blocks[i], x, top, theme.secondary);
                t.2.clone()
            } else {
                String::new()
            };
            self.layout.x_ticks.push((t.0, t.1, label));
        }
    }

    /// Direct labels at the right of the plot, at their series' last values, nudged apart so they
    /// don't overlap and kept inside the plot's height.
    fn direct_labels(&mut self, labels: &[(String, f64, Rgba)], x: f64, plot: Rect) {
        let mut placed: Vec<(String, TextBlock, f64, Rgba)> = labels
            .iter()
            .map(|(s, y, c)| {
                let b = self.text(s, LABEL, Weight::Medium);
                let top = y - f64::from(b.height) / 2.0;
                (s.clone(), b, top, *c)
            })
            .collect();
        placed.sort_by(|a, b| a.2.total_cmp(&b.2));
        let gap = self.px(2.0);
        for i in 1..placed.len() {
            let min = placed[i - 1].2 + f64::from(placed[i - 1].1.height) + gap;
            if placed[i].2 < min {
                placed[i].2 = min;
            }
        }
        // If the nudge pushed the last one out of the plot, shift all up.
        if let Some(last) = placed.last() {
            let over = last.2 + f64::from(last.1.height) - plot.y1;
            if over > 0.0 {
                for p in &mut placed {
                    p.2 -= over;
                }
            }
        }
        for (s, b, top, c) in placed {
            self.draw_text(&s, &b, x, top, c);
        }
    }
}

/// Draws `chart` and returns its layout.
pub fn line_chart(
    scene: &mut impl Scene,
    fonts: &mut Fonts,
    chart: &LineChart,
    frame: Frame,
    theme: Theme,
    numbers: &NumberFormat,
) -> Layout {
    let mut c = Ctx {
        scene,
        fonts,
        numbers,
        frame,
        theme,
        layout: Layout::default(),
    };
    let below_header = c.header(&chart.title, &chart.subtitle);
    // The time axis spans whole UTC days, so ticks on day and month boundaries at the ends count.
    let day = 86_400.0;
    let t0 = (chart.t[0] / day).floor() * day;
    let t1 = (chart.t[chart.t.len() - 1] / day).ceil() * day;
    let values = chart
        .line
        .iter()
        .chain(chart.band.iter().flat_map(|(lo, hi)| lo.iter().chain(hi)));
    let (lo, hi) = values.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| {
        (a.min(*v), b.max(*v))
    });

    // The plot's height decides the y ticks, whose labels' width decides the plot's left edge.
    let label_h = f64::from(c.text("0", LABEL, Weight::Regular).height);
    let top = snap4(below_header / frame.scale + 16.0) * frame.scale;
    let bottom = c.px(frame.height - PAD) - label_h - c.px(6.0);
    let y_ticks = c.value_ticks(lo, hi, bottom - top);
    let label_w = c
        .y_label_blocks(&y_ticks)
        .iter()
        .fold(0.0_f32, |m, b| m.max(b.width));
    let direct_w = [&chart.line_label, &chart.band_label]
        .iter()
        .map(|s| c.text(s, LABEL, Weight::Medium).width)
        .fold(0.0_f32, f32::max);
    let left = c.px(snap4(PAD + f64::from(label_w) / frame.scale + LABEL_GAP));
    let right = c.px(frame.width - PAD) - f64::from(direct_w) - c.px(LABEL_GAP);
    let plot = Rect::new(left, top, right.floor(), bottom.floor());
    c.layout.plot = plot;

    let (d0, d1) = (y_ticks[0], y_ticks[y_ticks.len() - 1]);
    let y_of = |v: f64| plot.y1 - (v - d0) / (d1 - d0) * plot.height();
    let x_of = |t: f64| plot.x0 + (t - t0) / (t1 - t0) * plot.width();

    // Time ticks no closer than 36 pt; the label pass thins them further if labels collide.
    let max_ticks = (plot.width() / c.px(36.0)).floor().max(2.0) as usize;
    // Ticks stop at the last data point: a tick at the axis's end would label a period with no data.
    let last = chart.t[chart.t.len() - 1];
    let ticks: Vec<(f64, f64, String)> = time_ticks(t0, last, max_ticks)
        .into_iter()
        .map(|t| (t.t, x_of(t.t), t.label))
        .collect();
    for t in &ticks {
        c.vline(t.1, plot.y0, plot.y1, theme.grid);
    }
    c.y_axis(&y_ticks, plot, y_of, true);
    c.x_labels(&ticks, plot.y1 + c.px(6.0), true);

    // The band, then the line over it.
    let mut ends = Vec::new();
    if let Some((lo, hi)) = &chart.band {
        let mut p = BezPath::new();
        for (i, (t, v)) in chart.t.iter().zip(hi).enumerate() {
            let pt = Point::new(x_of(*t), y_of(*v));
            if i == 0 { p.move_to(pt) } else { p.line_to(pt) }
        }
        for (t, v) in chart.t.iter().zip(lo).rev() {
            p.line_to(Point::new(x_of(*t), y_of(*v)));
        }
        p.close_path();
        c.scene.fill(&p, theme.series.with_alpha(BAND_ALPHA));
        let last = (hi[hi.len() - 1] + lo[lo.len() - 1]) / 2.0;
        ends.push((chart.band_label.clone(), y_of(last), theme.secondary));
    }
    let mut p = BezPath::new();
    for (i, (t, v)) in chart.t.iter().zip(&chart.line).enumerate() {
        let pt = Point::new(x_of(*t), y_of(*v));
        if i == 0 { p.move_to(pt) } else { p.line_to(pt) }
    }
    c.scene.stroke(&p, c.px(LINE_WIDTH), theme.series, true);
    ends.push((
        chart.line_label.clone(),
        y_of(chart.line[chart.line.len() - 1]),
        theme.series,
    ));
    c.direct_labels(&ends, plot.x1 + c.px(LABEL_GAP), plot);
    c.layout
}

/// Draws `chart` and returns its layout.
pub fn bar_chart(
    scene: &mut impl Scene,
    fonts: &mut Fonts,
    chart: &BarChart,
    frame: Frame,
    theme: Theme,
    numbers: &NumberFormat,
) -> Layout {
    let mut c = Ctx {
        scene,
        fonts,
        numbers,
        frame,
        theme,
        layout: Layout::default(),
    };
    let below_header = c.header(&chart.title, &chart.subtitle);
    let hi = chart.values.iter().fold(0.0_f64, |m, v| m.max(*v));
    let label_h = f64::from(c.text("0", LABEL, Weight::Regular).height);
    let top = snap4(below_header / frame.scale + 16.0) * frame.scale;
    let bottom = c.px(frame.height - PAD) - label_h - c.px(6.0);
    // Bars start at zero.
    let y_ticks = c.value_ticks(0.0, hi, bottom - top);
    let label_w = c
        .y_label_blocks(&y_ticks)
        .iter()
        .fold(0.0_f32, |m, b| m.max(b.width));
    let left = c.px(snap4(PAD + f64::from(label_w) / frame.scale + LABEL_GAP));
    let right = c.px(frame.width - PAD);
    let plot = Rect::new(left, top, right.floor(), bottom.floor());
    c.layout.plot = plot;
    let (d0, d1) = (y_ticks[0], y_ticks[y_ticks.len() - 1]);
    let y_of = |v: f64| plot.y1 - (v - d0) / (d1 - d0) * plot.height();

    // Gridlines under the bars, the zero baseline over them.
    let n = chart.values.len() as f64;
    let slot = plot.width() / n;
    let gap = (slot * 0.2).max(c.px(2.0)).round();
    c.y_axis(&y_ticks[1..], plot, y_of, false);
    for (i, v) in chart.values.iter().enumerate() {
        // Bar edges on whole pixels, so neighbors keep identical gaps.
        let x0 = (plot.x0 + i as f64 * slot + gap / 2.0).round();
        let x1 = (plot.x0 + (i + 1) as f64 * slot - gap / 2.0).round();
        let r = Rect::new(x0, y_of(*v).round(), x1, plot.y1.round());
        c.scene.fill(&r.to_path(0.1), theme.series);
    }
    c.y_axis(&y_ticks[..1], plot, y_of, true);
    let ticks: Vec<(f64, f64, String)> = chart
        .categories
        .iter()
        .enumerate()
        .map(|(i, s)| (i as f64, plot.x0 + (i as f64 + 0.5) * slot, s.clone()))
        .collect();
    c.x_labels(&ticks, plot.y1 + c.px(6.0), false);
    c.layout
}

/// Rounds points up to the 4 pt grid.
fn snap4(pt: f64) -> f64 {
    (pt / 4.0).ceil() * 4.0
}
