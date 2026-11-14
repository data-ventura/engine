//! Text set in the UI face, Inter (SIL Open Font License, `fonts/OFL.txt`), with parley. The fonts
//! are compiled in, so layouts and exports are the same on every machine.

use std::borrow::Cow;

use parley::fontique::{Blob, Collection, CollectionOptions, SourceCache};
use parley::{
    FontContext, FontFamily, FontFeature, FontFeatures, FontWeight, LayoutContext,
    PositionedLayoutItem, StyleProperty,
};

use crate::vector::{Glyph, GlyphRun};

const INTER_REGULAR: &[u8] = include_bytes!("../fonts/Inter-Regular.ttf");
const INTER_MEDIUM: &[u8] = include_bytes!("../fonts/Inter-Medium.ttf");
const FAMILY: &str = "Inter";
/// Tabular figures, so digits line up in columns.
const TABULAR: [FontFeature; 1] = [FontFeature::new(parley::setting::Tag::new(b"tnum"), 1)];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weight {
    Regular,
    Medium,
}

/// A line of text laid out at a size, ready to draw with its top-left corner at an origin.
#[derive(Debug, Clone)]
pub struct TextBlock {
    pub width: f32,
    pub height: f32,
    /// Distance from the top to the baseline.
    pub baseline: f32,
    /// Glyph runs positioned relative to the top-left corner.
    pub runs: Vec<GlyphRun>,
}

impl TextBlock {
    /// The runs moved so the block's top-left corner is at `(x, y)`.
    pub fn at(&self, x: f32, y: f32) -> Vec<GlyphRun> {
        self.runs
            .iter()
            .map(|r| GlyphRun {
                font: r.font.clone(),
                size: r.size,
                glyphs: r
                    .glyphs
                    .iter()
                    .map(|g| Glyph {
                        id: g.id,
                        x: g.x + x,
                        y: g.y + y,
                    })
                    .collect(),
            })
            .collect()
    }
}

/// The font collection (Inter only, no system fonts) and parley's reusable layout state.
pub struct Fonts {
    fcx: FontContext,
    lcx: LayoutContext<[u8; 4]>,
}

impl std::fmt::Debug for Fonts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Fonts")
    }
}

impl Default for Fonts {
    fn default() -> Self {
        Self::new()
    }
}

impl Fonts {
    pub fn new() -> Self {
        let mut collection = Collection::new(CollectionOptions {
            shared: false,
            system_fonts: false,
        });
        for data in [INTER_REGULAR, INTER_MEDIUM] {
            collection.register_fonts(Blob::new(std::sync::Arc::new(data)), None);
        }
        Self {
            fcx: FontContext {
                collection,
                source_cache: SourceCache::default(),
            },
            lcx: LayoutContext::new(),
        }
    }

    /// Lays out one line of `text` at `size` pixels per em, with tabular figures.
    pub fn layout(&mut self, text: &str, size: f32, weight: Weight) -> TextBlock {
        let mut b = self.lcx.ranged_builder(&mut self.fcx, text, 1.0, false);
        b.push_default(FontFamily::from(FAMILY));
        b.push_default(StyleProperty::FontSize(size));
        b.push_default(StyleProperty::FontWeight(match weight {
            Weight::Regular => FontWeight::NORMAL,
            Weight::Medium => FontWeight::new(500.0),
        }));
        b.push_default(StyleProperty::FontFeatures(FontFeatures::List(
            Cow::Borrowed(&TABULAR),
        )));
        let mut layout = b.build(text);
        layout.break_all_lines(None);
        let mut runs = Vec::new();
        let mut baseline = 0.0;
        for line in layout.lines() {
            baseline = line.metrics().baseline;
            for item in line.items() {
                if let PositionedLayoutItem::GlyphRun(run) = item {
                    runs.push(GlyphRun {
                        font: run.run().font().clone(),
                        size: run.run().font_size(),
                        glyphs: run
                            .positioned_glyphs()
                            .map(|g| Glyph {
                                id: g.id,
                                x: g.x,
                                y: g.y,
                            })
                            .collect(),
                    });
                }
            }
        }
        TextBlock {
            width: layout.width(),
            height: layout.height(),
            baseline,
            runs,
        }
    }
}
