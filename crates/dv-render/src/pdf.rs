//! PDF export: the [`Scene`] calls written as a one-page vector PDF.
//!
//! Paths stay paths and text stays text, so the file is sharp at any zoom, and its text can be
//! searched, selected and copied. Each font is embedded as a subset (only the glyphs drawn, through
//! the `subsetter` crate) in a CID-keyed Type 0 font with a `ToUnicode` map built from the text
//! the layout shaped. Glyphs keep the positions the layout gave them: each run is one text object
//! whose `TJ` adjustments correct the font's advances to parley's. The output has no timestamps
//! or random IDs, so the same chart always gives the same bytes.
//!
//! Coordinates come in as device pixels, y down, like every [`Scene`]; the page is
//! `width / scale` × `height / scale` points.

use std::collections::{BTreeMap, BTreeSet};

use pdf_writer::types::{CidFontType, FontFlags, LineCapStyle, LineJoinStyle, SystemInfo};
use pdf_writer::{Content, Filter, Finish, Name, Pdf, Rect as PdfRect, Ref, Str};
use skrifa::MetadataProvider;
use skrifa::instance::{LocationRef, Size};
use subsetter::GlyphRemapper;
use vello::kurbo::PathEl;
use vello::peniko::FontData;

use crate::vector::{BezPath, GlyphRun, Point, Rect, Rgba, Scene, Shape};

/// A PDF page, `width` × `height` device pixels at `scale` device pixels per point.
pub struct PdfScene {
    scale: f64,
    /// Page height in points.
    height: f64,
    width: f64,
    content: Content,
    /// Fonts in order of first use, keyed by blob and collection index.
    fonts: Vec<EmbeddedFont>,
    /// Alpha values in use, each with an extended graphics state.
    alphas: BTreeSet<u8>,
}

impl std::fmt::Debug for PdfScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PdfScene")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("fonts", &self.fonts)
            .finish_non_exhaustive()
    }
}

struct EmbeddedFont {
    key: (u64, u32),
    font: FontData,
    /// Old glyph IDs to the subset's, which are also the CIDs written in the content.
    glyphs: GlyphRemapper,
    /// The text each subset glyph first stood for, from the layout.
    text: BTreeMap<u16, String>,
}

impl std::fmt::Debug for EmbeddedFont {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddedFont")
            .field("glyphs", &self.glyphs.num_gids())
            .finish()
    }
}

impl PdfScene {
    pub fn new(width: u32, height: u32, scale: f64, background: Rgba) -> Self {
        let mut s = Self {
            scale,
            width: f64::from(width) / scale,
            height: f64::from(height) / scale,
            content: Content::new(),
            fonts: Vec::new(),
            alphas: BTreeSet::new(),
        };
        let bg = Rect::new(0.0, 0.0, f64::from(width), f64::from(height)).to_path(0.1);
        s.fill(&bg, background);
        s
    }

    /// A device pixel position in page points, y up.
    fn pt(&self, p: Point) -> (f32, f32) {
        (
            (p.x / self.scale) as f32,
            (self.height - p.y / self.scale) as f32,
        )
    }

    fn path(&mut self, path: &BezPath) {
        let mut last = Point::ZERO;
        for el in path.elements() {
            match *el {
                PathEl::MoveTo(p) => {
                    let (x, y) = self.pt(p);
                    self.content.move_to(x, y);
                    last = p;
                }
                PathEl::LineTo(p) => {
                    let (x, y) = self.pt(p);
                    self.content.line_to(x, y);
                    last = p;
                }
                PathEl::QuadTo(c, p) => {
                    // The cubic with the same curve.
                    let c1 = last + (c - last) * (2.0 / 3.0);
                    let c2 = p + (c - p) * (2.0 / 3.0);
                    self.cubic(c1, c2, p);
                    last = p;
                }
                PathEl::CurveTo(c1, c2, p) => {
                    self.cubic(c1, c2, p);
                    last = p;
                }
                PathEl::ClosePath => {
                    self.content.close_path();
                }
            }
        }
    }

    fn cubic(&mut self, c1: Point, c2: Point, p: Point) {
        let ((x1, y1), (x2, y2), (x3, y3)) = (self.pt(c1), self.pt(c2), self.pt(p));
        self.content.cubic_to(x1, y1, x2, y2, x3, y3);
    }

    /// Runs `draw` with `color`'s alpha in effect: inside a saved state with an extended graphics
    /// state when translucent.
    fn with_alpha(&mut self, color: Rgba, draw: impl FnOnce(&mut Self)) {
        let a = color.0[3];
        if a == 255 {
            draw(self);
            return;
        }
        self.alphas.insert(a);
        self.content.save_state();
        self.content.set_parameters(Name(alpha_name(a).as_bytes()));
        draw(self);
        self.content.restore_state();
    }

    fn font_index(&mut self, font: &FontData) -> usize {
        let key = (font.data.id(), font.index);
        if let Some(i) = self.fonts.iter().position(|f| f.key == key) {
            return i;
        }
        self.fonts.push(EmbeddedFont {
            key,
            font: font.clone(),
            glyphs: GlyphRemapper::new(),
            text: BTreeMap::new(),
        });
        self.fonts.len() - 1
    }

    /// The finished document.
    pub fn finish(self) -> Result<Vec<u8>, String> {
        let mut alloc = Ref::new(1);
        let mut next = || alloc.bump();
        let (catalog, pages, page, content) = (next(), next(), next(), next());
        let font_refs: Vec<Ref> = self.fonts.iter().map(|_| next()).collect();
        let alpha_refs: Vec<(u8, Ref)> = self.alphas.iter().map(|a| (*a, next())).collect();

        let mut pdf = Pdf::new();
        pdf.catalog(catalog).pages(pages);
        pdf.pages(pages).kids([page]).count(1);
        let mut p = pdf.page(page);
        p.parent(pages)
            .media_box(PdfRect::new(
                0.0,
                0.0,
                self.width as f32,
                self.height as f32,
            ))
            .contents(content);
        let mut res = p.resources();
        let mut fonts = res.fonts();
        for (i, r) in font_refs.iter().enumerate() {
            fonts.pair(Name(font_name(i).as_bytes()), *r);
        }
        fonts.finish();
        let mut states = res.ext_g_states();
        for (a, r) in &alpha_refs {
            states.pair(Name(alpha_name(*a).as_bytes()), *r);
        }
        states.finish();
        res.finish();
        p.finish();

        let data = deflate(&self.content.finish());
        pdf.stream(content, &data).filter(Filter::FlateDecode);
        for (a, r) in alpha_refs {
            let v = f32::from(a) / 255.0;
            pdf.ext_graphics(r).non_stroking_alpha(v).stroking_alpha(v);
        }
        for (font, r) in self.fonts.iter().zip(font_refs) {
            write_font(&mut pdf, font, r, &mut next)?;
        }
        Ok(pdf.finish())
    }
}

fn font_name(i: usize) -> String {
    format!("F{i}")
}

fn alpha_name(a: u8) -> String {
    format!("A{a}")
}

fn deflate(data: &[u8]) -> Vec<u8> {
    miniz_oxide::deflate::compress_to_vec_zlib(data, 6)
}

/// Writes `font` as a Type 0 font at `type0` with a subset of its glyphs.
fn write_font(
    pdf: &mut Pdf,
    font: &EmbeddedFont,
    type0: Ref,
    next: &mut impl FnMut() -> Ref,
) -> Result<(), String> {
    let data = font.font.data.as_ref();
    let face = skrifa::FontRef::from_index(data, font.font.index).map_err(|e| e.to_string())?;
    let metrics = face.metrics(Size::unscaled(), LocationRef::default());
    let units = f32::from(metrics.units_per_em.max(1));
    let to_1000 = |v: f32| v * 1000.0 / units;
    let advances = face.glyph_metrics(Size::unscaled(), LocationRef::default());
    let old_gids: Vec<u16> = font.glyphs.remapped_gids().collect();

    let subset =
        subsetter::subset(data, font.font.index, &font.glyphs).map_err(|e| e.to_string())?;
    let postscript = face
        .localized_strings(skrifa::string::StringId::POSTSCRIPT_NAME)
        .english_or_first()
        .map(|s| s.chars().filter(|c| c.is_ascii_graphic()).collect())
        .unwrap_or_else(|| "Font".to_string());
    // Subset fonts are named with a tag of six capitals, here derived from the glyphs kept.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for g in &old_gids {
        for b in g.to_be_bytes() {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    let tag: String = (0..6)
        .map(|i| char::from(b'A' + ((hash >> (i * 8)) % 26) as u8))
        .collect();
    let base = format!("{tag}+{postscript}");

    let (cid, descriptor, cmap, file) = (next(), next(), next(), next());
    let system = SystemInfo {
        registry: Str(b"Adobe"),
        ordering: Str(b"Identity"),
        supplement: 0,
    };
    pdf.type0_font(type0)
        .base_font(Name(base.as_bytes()))
        .encoding_predefined(Name(b"Identity-H"))
        .descendant_font(cid)
        .to_unicode(cmap);
    let mut c = pdf.cid_font(cid);
    c.subtype(CidFontType::Type2)
        .base_font(Name(base.as_bytes()))
        .system_info(system)
        .font_descriptor(descriptor)
        .cid_to_gid_map_predefined(Name(b"Identity"));
    c.widths().consecutive(
        0,
        old_gids.iter().map(|g| {
            to_1000(
                advances
                    .advance_width(skrifa::GlyphId::new(u32::from(*g)))
                    .unwrap_or(0.0),
            )
        }),
    );
    c.finish();

    let bounds = metrics.bounds.unwrap_or_default();
    let cap = metrics.cap_height.unwrap_or(metrics.ascent);
    pdf.font_descriptor(descriptor)
        .name(Name(base.as_bytes()))
        .flags(FontFlags::NON_SYMBOLIC)
        .bbox(PdfRect::new(
            to_1000(bounds.x_min),
            to_1000(bounds.y_min),
            to_1000(bounds.x_max),
            to_1000(bounds.y_max),
        ))
        .italic_angle(metrics.italic_angle)
        .ascent(to_1000(metrics.ascent))
        .descent(to_1000(metrics.descent))
        .cap_height(to_1000(cap))
        // Viewers only use the stem width as a hint when they substitute a font; this one's
        // embedded.
        .stem_v(80.0)
        .font_file2(file);
    let compressed = deflate(&subset);
    pdf.stream(file, &compressed).filter(Filter::FlateDecode);

    // Each glyph's text from the layout, which also covers glyphs the character map can't reach
    // (tabular figures, ligatures). Glyphs a cluster drew after its first get no text. As a
    // fallback, the lowest printable code point the character map gives the glyph.
    let mut chars: BTreeMap<u32, char> = BTreeMap::new();
    for (cp, gid) in face.charmap().mappings() {
        if let Some(ch) = char::from_u32(cp).filter(|c| !c.is_control()) {
            chars.entry(gid.to_u32()).or_insert(ch);
        }
    }
    let mut unicode = pdf_writer::types::UnicodeCmap::new(Name(b"Custom"), system);
    for (new, old) in old_gids.iter().enumerate() {
        let new = new as u16;
        if let Some(text) = font.text.get(&new) {
            unicode.pair_with_multiple(new, text.chars());
        } else if let Some(ch) = chars.get(&u32::from(*old)) {
            unicode.pair(new, *ch);
        }
    }
    let cmap_data = deflate(&unicode.finish());
    pdf.cmap(cmap, &cmap_data).filter(Filter::FlateDecode);
    Ok(())
}

impl Scene for PdfScene {
    fn fill(&mut self, path: &BezPath, color: Rgba) {
        self.with_alpha(color, |s| {
            let [r, g, b, _] = rgb(color);
            s.content.set_fill_rgb(r, g, b);
            s.path(path);
            s.content.fill_nonzero();
        });
    }

    fn stroke(&mut self, path: &BezPath, width: f64, color: Rgba, round: bool) {
        self.with_alpha(color, |s| {
            let [r, g, b, _] = rgb(color);
            s.content.set_stroke_rgb(r, g, b);
            s.content.set_line_width((width / s.scale) as f32);
            if round {
                s.content.set_line_cap(LineCapStyle::RoundCap);
                s.content.set_line_join(LineJoinStyle::RoundJoin);
            } else {
                s.content.set_line_cap(LineCapStyle::ButtCap);
                s.content.set_line_join(LineJoinStyle::MiterJoin);
            }
            s.path(path);
            s.content.stroke();
        });
    }

    fn glyphs(&mut self, run: &GlyphRun, color: Rgba) {
        if run.glyphs.is_empty() {
            return;
        }
        let i = self.font_index(&run.font);
        let Ok(face) = skrifa::FontRef::from_index(run.font.data.as_ref(), run.font.index) else {
            return;
        };
        let advances = face.glyph_metrics(Size::new(run.size), LocationRef::default());
        let size_pt = (f64::from(run.size) / self.scale) as f32;
        let ids: Vec<u16> = run
            .glyphs
            .iter()
            .map(|g| {
                let old = u16::try_from(g.id).unwrap_or(0);
                self.fonts[i].glyphs.remap(old)
            })
            .collect();
        for (k, text) in &run.text {
            if let Some(id) = ids.get(*k) {
                self.fonts[i]
                    .text
                    .entry(*id)
                    .or_insert_with(|| text.clone());
            }
        }
        self.with_alpha(color, |s| {
            let [r, g, b, _] = rgb(color);
            s.content.set_fill_rgb(r, g, b);
            s.content.begin_text();
            s.content.set_font(Name(font_name(i).as_bytes()), size_pt);
            // One text object per stretch of glyphs on the same baseline.
            let mut start = 0;
            while start < run.glyphs.len() {
                let y = run.glyphs[start].y;
                let end = run.glyphs[start..]
                    .iter()
                    .position(|g| g.y != y)
                    .map_or(run.glyphs.len(), |n| start + n);
                let first = run.glyphs[start];
                let (x0, y0) = s.pt(Point::new(f64::from(first.x), f64::from(first.y)));
                s.content.set_text_matrix([1.0, 0.0, 0.0, 1.0, x0, y0]);
                let mut tj = s.content.show_positioned();
                let mut items = tj.items();
                let glyphs = &run.glyphs[start..end];
                for (k, (g, id)) in glyphs.iter().zip(&ids[start..end]).enumerate() {
                    items.show(Str(&id.to_be_bytes()));
                    if let Some(next) = glyphs.get(k + 1) {
                        // TJ moves the next glyph left by adjust / 1000 em.
                        let advance = advances
                            .advance_width(skrifa::GlyphId::new(g.id))
                            .unwrap_or(0.0);
                        let adjust = (advance - (next.x - g.x)) * 1000.0 / run.size;
                        if adjust.abs() > 0.01 {
                            items.adjust(adjust);
                        }
                    }
                }
                items.finish();
                tj.finish();
                start = end;
            }
            s.content.end_text();
        });
    }
}

/// sRGB components in 0..=1.
fn rgb(c: Rgba) -> [f32; 4] {
    c.0.map(|v| f32::from(v) / 255.0)
}
