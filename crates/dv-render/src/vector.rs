//! The vector scene for everything people read closely: axes, text, marks and annotations.
//!
//! Drawing goes through the [`Scene`] trait, so the backend can change without touching the charts:
//! [`VelloScene`] draws on the GPU with vello's analytic anti-aliasing, and [`SvgScene`] writes the
//! same calls as SVG for export (text as glyph outlines, so the file needs no fonts and matches the
//! screen). A lyon + MSAA backend can take vello's place behind the same trait.
//!
//! Coordinates are device pixels, y down.

use std::fmt::Write as _;
use std::num::NonZeroUsize;

use vello::kurbo::{Affine, Cap, Join, Stroke};
pub use vello::kurbo::{BezPath, Line, Point, Rect, Shape};
use vello::peniko::{Color, Fill, FontData};

use crate::gpu::Gpu;

/// An sRGB color with straight alpha.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rgba(pub [u8; 4]);

impl Rgba {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self([r, g, b, 255])
    }

    pub const fn with_alpha(self, a: u8) -> Self {
        Self([self.0[0], self.0[1], self.0[2], a])
    }

    fn color(self) -> Color {
        Color::from_rgba8(self.0[0], self.0[1], self.0[2], self.0[3])
    }

    fn hex(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.0[0], self.0[1], self.0[2])
    }

    fn opacity(self) -> Option<String> {
        (self.0[3] != 255).then(|| format!("{:.4}", f64::from(self.0[3]) / 255.0))
    }
}

/// One glyph, positioned in device pixels (x along the baseline, y the baseline).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Glyph {
    pub id: u32,
    pub x: f32,
    pub y: f32,
}

/// Glyphs in one font at one size.
#[derive(Debug, Clone)]
pub struct GlyphRun {
    pub font: FontData,
    /// Pixels per em.
    pub size: f32,
    pub glyphs: Vec<Glyph>,
}

/// What every vector backend draws.
pub trait Scene {
    fn fill(&mut self, path: &BezPath, color: Rgba);
    /// Strokes `path` `width` pixels wide, with round caps and joins if `round`, else butt caps and
    /// miter joins (for hairlines that must end exactly on a pixel edge).
    fn stroke(&mut self, path: &BezPath, width: f64, color: Rgba, round: bool);
    fn glyphs(&mut self, run: &GlyphRun, color: Rgba);
}

/// A vello scene, drawn by [`VectorRenderer`].
#[derive(Default)]
pub struct VelloScene {
    pub scene: vello::Scene,
}

impl std::fmt::Debug for VelloScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("VelloScene")
    }
}

impl Scene for VelloScene {
    fn fill(&mut self, path: &BezPath, color: Rgba) {
        self.scene
            .fill(Fill::NonZero, Affine::IDENTITY, color.color(), None, path);
    }

    fn stroke(&mut self, path: &BezPath, width: f64, color: Rgba, round: bool) {
        let style = if round {
            Stroke::new(width)
                .with_caps(Cap::Round)
                .with_join(Join::Round)
        } else {
            Stroke::new(width)
                .with_caps(Cap::Butt)
                .with_join(Join::Miter)
        };
        self.scene
            .stroke(&style, Affine::IDENTITY, color.color(), None, path);
    }

    fn glyphs(&mut self, run: &GlyphRun, color: Rgba) {
        self.scene
            .draw_glyphs(&run.font)
            .font_size(run.size)
            .hint(false)
            .brush(color.color())
            .draw(
                Fill::NonZero,
                run.glyphs.iter().map(|g| vello::Glyph {
                    id: g.id,
                    x: g.x,
                    y: g.y,
                }),
            );
    }
}

/// An SVG document, `width` × `height` device pixels, shown at `scale` device pixels per CSS pixel.
#[derive(Debug)]
pub struct SvgScene {
    width: u32,
    height: u32,
    scale: f64,
    body: String,
}

impl SvgScene {
    pub fn new(width: u32, height: u32, scale: f64, background: Rgba) -> Self {
        let mut s = Self {
            width,
            height,
            scale,
            body: String::new(),
        };
        let bg = Rect::new(0.0, 0.0, f64::from(width), f64::from(height)).to_path(0.1);
        s.fill(&bg, background);
        s
    }

    /// The finished document.
    pub fn finish(self) -> String {
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n{}</svg>\n",
            f64::from(self.width) / self.scale,
            f64::from(self.height) / self.scale,
            self.width,
            self.height,
            self.body
        )
    }

    fn paint(color: Rgba, attr: &str) -> String {
        let mut s = format!(" {attr}=\"{}\"", color.hex());
        if let Some(o) = color.opacity() {
            let _ = write!(s, " {attr}-opacity=\"{o}\"");
        }
        s
    }
}

impl Scene for SvgScene {
    fn fill(&mut self, path: &BezPath, color: Rgba) {
        let _ = writeln!(
            self.body,
            "<path d=\"{}\"{}/>",
            path.to_svg(),
            Self::paint(color, "fill")
        );
    }

    fn stroke(&mut self, path: &BezPath, width: f64, color: Rgba, round: bool) {
        let (cap, join) = if round {
            ("round", "round")
        } else {
            ("butt", "miter")
        };
        let _ = writeln!(
            self.body,
            "<path d=\"{}\" fill=\"none\"{} stroke-width=\"{width}\" stroke-linecap=\"{cap}\" stroke-linejoin=\"{join}\"/>",
            path.to_svg(),
            Self::paint(color, "stroke")
        );
    }

    fn glyphs(&mut self, run: &GlyphRun, color: Rgba) {
        use skrifa::MetadataProvider;
        use skrifa::instance::{LocationRef, Size};
        use skrifa::outline::DrawSettings;

        let Ok(font) = skrifa::FontRef::from_index(run.font.data.as_ref(), run.font.index) else {
            return;
        };
        let outlines = font.outline_glyphs();
        let mut pen = SvgPen::default();
        for g in &run.glyphs {
            let Some(outline) = outlines.get(skrifa::GlyphId::new(g.id)) else {
                continue;
            };
            pen.origin = (g.x, g.y);
            let settings = DrawSettings::unhinted(Size::new(run.size), LocationRef::default());
            let _ = outline.draw(settings, &mut pen);
        }
        if !pen.path.elements().is_empty() {
            self.fill(&pen.path, color);
        }
    }
}

/// Turns font outlines (y up) into a path in device pixels (y down) at a glyph's origin.
#[derive(Default)]
struct SvgPen {
    origin: (f32, f32),
    path: BezPath,
}

impl SvgPen {
    fn p(&self, x: f32, y: f32) -> Point {
        Point::new(f64::from(self.origin.0 + x), f64::from(self.origin.1 - y))
    }
}

impl skrifa::outline::OutlinePen for SvgPen {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.p(x, y);
        self.path.move_to(p);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.p(x, y);
        self.path.line_to(p);
    }
    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        let (c, p) = (self.p(cx0, cy0), self.p(x, y));
        self.path.quad_to(c, p);
    }
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        let (c0, c1, p) = (self.p(cx0, cy0), self.p(cx1, cy1), self.p(x, y));
        self.path.curve_to(c0, c1, p);
    }
    fn close(&mut self) {
        self.path.close_path();
    }
}

/// Draws [`VelloScene`]s into textures with vello's analytic (area) anti-aliasing.
pub struct VectorRenderer {
    renderer: vello::Renderer,
}

impl std::fmt::Debug for VectorRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("VectorRenderer")
    }
}

impl VectorRenderer {
    pub fn new(gpu: &Gpu) -> Result<Self, String> {
        let renderer = vello::Renderer::new(
            &gpu.device,
            vello::RendererOptions {
                use_cpu: false,
                antialiasing_support: vello::AaSupport::area_only(),
                num_init_threads: NonZeroUsize::new(1),
                pipeline_cache: None,
            },
        )
        .map_err(|e| format!("vello: {e}"))?;
        Ok(Self { renderer })
    }

    /// Draws `scene` over `background` into a new `Rgba8Unorm` texture of `size` pixels, which can
    /// be copied out or sampled.
    pub fn render(
        &mut self,
        gpu: &Gpu,
        scene: &VelloScene,
        size: [u32; 2],
        background: Rgba,
    ) -> Result<wgpu::Texture, String> {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vector"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        self.renderer
            .render_to_texture(
                &gpu.device,
                &gpu.queue,
                &scene.scene,
                &view,
                &vello::RenderParams {
                    base_color: background.color(),
                    width: size[0],
                    height: size[1],
                    antialiasing_method: vello::AaConfig::Area,
                },
            )
            .map_err(|e| format!("vello: {e}"))?;
        Ok(texture)
    }
}
