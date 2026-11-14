//! Tick labels: the same number of decimals on every label of an axis, a true minus sign (U+2212),
//! and digit grouping and decimal separators from a [`NumberFormat`]: a locale's (from Unicode CLDR
//! via ICU4X), or SI style.
//!
//! Labels on one axis share their grouping: if the largest would be grouped, all are, so an axis
//! never mixes "5000" and "10 000" (which a locale like Spanish, which groups only from five
//! digits, would otherwise do).

use fixed_decimal::Decimal;
use icu_decimal::DecimalFormatter;
use icu_decimal::options::{DecimalFormatterOptions, GroupingStrategy};
use icu_locale_core::Locale;

/// The minus sign, not a hyphen.
pub const MINUS: char = '\u{2212}';
/// SI style's digit group separator.
pub const THIN_SPACE: char = '\u{2009}';

/// How numbers are written.
pub struct NumberFormat {
    style: Style,
    true_minus: bool,
}

enum Style {
    /// Groups of three separated by thin spaces once a number has five integer digits, and a point
    /// for the decimal separator, as the SI brochure and ISO 80000 write them.
    Si,
    Locale {
        tag: String,
        /// The locale's own grouping, which may leave four-digit numbers ungrouped.
        auto: Box<DecimalFormatter>,
        plain: Box<DecimalFormatter>,
        /// The locale's group separator, if it groups at all.
        separator: Option<char>,
    },
}

impl std::fmt::Debug for NumberFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let style = match &self.style {
            Style::Si => "SI",
            Style::Locale { tag, .. } => tag,
        };
        f.debug_struct("NumberFormat")
            .field("style", &style)
            .field("true_minus", &self.true_minus)
            .finish()
    }
}

impl Default for NumberFormat {
    /// SI style: the same on every machine, for headless rendering and exports. The app passes
    /// the user's locale instead.
    fn default() -> Self {
        Self::si()
    }
}

impl NumberFormat {
    pub fn si() -> Self {
        Self {
            style: Style::Si,
            true_minus: true,
        }
    }

    /// The conventions of a BCP 47 locale, such as `en-US`, `de-DE`, `fr` or `en-IN`. Locales
    /// CLDR doesn't know fall back to their language, then to the root locale.
    pub fn locale(tag: &str) -> Result<Self, String> {
        let locale: Locale = tag
            .parse()
            .map_err(|e| format!("not a locale: {tag}: {e:?}"))?;
        let make = |grouping| {
            let mut options = DecimalFormatterOptions::default();
            options.grouping_strategy = Some(grouping);
            DecimalFormatter::try_new((&locale).into(), options)
                .map(Box::new)
                .map_err(|e| format!("no number format for {tag}: {e:?}"))
        };
        let auto = make(GroupingStrategy::Auto)?;
        let separator = auto
            .format(&Decimal::from(10_000_000))
            .to_string()
            .chars()
            .find(|c| !c.is_numeric());
        Ok(Self {
            style: Style::Locale {
                tag: tag.to_string(),
                auto,
                plain: make(GroupingStrategy::Never)?,
                separator,
            },
            true_minus: true,
        })
    }

    /// Whether negative numbers use U+2212 (the default) rather than the locale's own sign.
    pub fn with_true_minus(mut self, on: bool) -> Self {
        self.true_minus = on;
        self
    }

    /// Labels for one axis's tick values. Every label gets the fewest decimals that represent all
    /// the values exactly (to 1e-9 of the largest), and all labels share one grouping decision.
    pub fn tick_labels(&self, values: &[f64]) -> Vec<String> {
        let biggest = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let tolerance = 1e-9 * biggest.max(1e-300);
        let decimals = (0..=15)
            .find(|&d| {
                let f = 10f64.powi(d);
                values
                    .iter()
                    .all(|v| ((v * f).round() / f - v).abs() <= tolerance)
            })
            .unwrap_or(15) as usize;
        let group = self.groups(biggest.round());
        values
            .iter()
            .map(|&v| self.number(v, decimals, group))
            .collect()
    }

    /// Whether the locale (with its own minimum grouping digits) would group `v`.
    fn groups(&self, v: f64) -> bool {
        match &self.style {
            Style::Si => v >= 10_000.0,
            Style::Locale { auto, plain, .. } => {
                let d = Decimal::try_from_str(&format!("{v:.0}")).expect("an integer parses");
                auto.format(&d).to_string() != plain.format(&d).to_string()
            }
        }
    }

    /// `v` with `decimals` decimals, grouped if `group`.
    pub fn number(&self, v: f64, decimals: usize, group: bool) -> String {
        let text = format!("{:.*}", decimals, v.abs());
        // Zero is never negative, even when it rounds from a tiny negative value.
        let negative = v < 0.0 && text.chars().any(|c| c.is_ascii_digit() && c != '0');
        let out = match &self.style {
            Style::Si => si(&text, group),
            Style::Locale {
                auto,
                plain,
                separator,
                ..
            } => {
                let d = Decimal::try_from_str(&text).expect("a formatted number parses");
                if group {
                    force_group(auto.format(&d).to_string(), *separator)
                } else {
                    plain.format(&d).to_string()
                }
            }
        };
        if !negative {
            return out;
        }
        if self.true_minus {
            format!("{MINUS}{out}")
        } else {
            match &self.style {
                Style::Si => format!("-{out}"),
                Style::Locale {
                    auto,
                    plain,
                    separator,
                    ..
                } => {
                    let mut d = Decimal::try_from_str(&text).expect("a formatted number parses");
                    d.set_sign(fixed_decimal::Sign::Negative);
                    if group {
                        force_group(auto.format(&d).to_string(), *separator)
                    } else {
                        plain.format(&d).to_string()
                    }
                }
            }
        }
    }
}

/// Groups a four-digit integer part that the locale's minimum grouping digits left alone (CLDR's
/// minimum grouping is at most 2, so only four-digit numbers are affected), so it matches the
/// grouped labels next to it on the same axis.
fn force_group(s: String, separator: Option<char>) -> String {
    let Some(sep) = separator else { return s };
    let start = s.find(|c: char| c.is_numeric()).unwrap_or(0);
    let digits = s[start..].chars().take_while(|c| c.is_numeric()).count();
    if digits != 4 {
        return s;
    }
    let first = s[start..].chars().next().expect("a digit");
    let split = start + first.len_utf8();
    format!("{}{sep}{}", &s[..split], &s[split..])
}

fn si(text: &str, group: bool) -> String {
    let (int, frac) = match text.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (text, None),
    };
    let mut out = String::new();
    for (i, c) in int.chars().enumerate() {
        if group && i > 0 && (int.len() - i) % 3 == 0 {
            out.push(THIN_SPACE);
        }
        out.push(c);
    }
    if let Some(frac) = frac {
        out.push('.');
        out.push_str(frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const NNBSP: char = '\u{202F}';

    fn labels(f: &NumberFormat, v: &[f64]) -> Vec<String> {
        f.tick_labels(v)
    }

    #[test]
    fn si_style() {
        let f = NumberFormat::si();
        assert_eq!(
            labels(&f, &[-20.0, -10.0, 0.0, 10.0]),
            ["\u{2212}20", "\u{2212}10", "0", "10"]
        );
        assert_eq!(labels(&f, &[0.0, 2.5, 5.0]), ["0.0", "2.5", "5.0"]);
        // Floating-point steps don't leak extra digits.
        assert_eq!(
            labels(&f, &[0.1, 0.2, 0.30000000000000004]),
            ["0.1", "0.2", "0.3"]
        );
        // Four-digit values alone stay ungrouped; with a five-digit value, all are grouped.
        assert_eq!(labels(&f, &[0.0, 1000.0, 2000.0]), ["0", "1000", "2000"]);
        assert_eq!(
            labels(&f, &[0.0, 5000.0, 10_000.0]),
            ["0", "5\u{2009}000", "10\u{2009}000"]
        );
        assert_eq!(f.number(-0.0001, 2, false), "0.00");
        assert_eq!(f.number(-1234.5, 1, true), "\u{2212}1\u{2009}234.5");
    }

    /// Expected strings are CLDR's conventions for each locale (CLDR 47, as compiled into ICU4X).
    #[test]
    fn locales() {
        let en = NumberFormat::locale("en-US").unwrap();
        assert_eq!(
            labels(&en, &[0.0, 500.0, 1000.0, 2500.0]),
            ["0", "500", "1,000", "2,500"]
        );
        assert_eq!(
            labels(&en, &[-1.5, 0.0, 1.5]),
            ["\u{2212}1.5", "0.0", "1.5"]
        );
        assert_eq!(
            labels(&en, &[-15_000.0, 0.0, 15_000.0]),
            ["\u{2212}15,000", "0", "15,000"]
        );
        let de = NumberFormat::locale("de-DE").unwrap();
        assert_eq!(labels(&de, &[0.0, 1000.0, 2000.0]), ["0", "1.000", "2.000"]);
        assert_eq!(labels(&de, &[0.0, 2.5]), ["0,0", "2,5"]);
        let fr = NumberFormat::locale("fr-FR").unwrap();
        assert_eq!(
            labels(&fr, &[0.0, 25_000.0]),
            ["0".to_string(), format!("25{NNBSP}000")]
        );
        // Indian grouping: lakhs and crores.
        let hi = NumberFormat::locale("en-IN").unwrap();
        assert_eq!(labels(&hi, &[0.0, 2_500_000.0]), ["0", "25,00,000"]);
        // Spanish groups only from five digits; an axis that reaches five groups all its labels.
        let es = NumberFormat::locale("es-ES").unwrap();
        assert_eq!(labels(&es, &[0.0, 1000.0, 2000.0]), ["0", "1000", "2000"]);
        assert_eq!(
            labels(&es, &[0.0, 5000.0, 10_000.0]),
            ["0", "5.000", "10.000"]
        );
        // The locale's own minus, if asked for.
        let en_hyphen = NumberFormat::locale("en-US")
            .unwrap()
            .with_true_minus(false);
        assert_eq!(en_hyphen.number(-20.0, 0, false), "-20");
        assert!(NumberFormat::locale("not a locale!").is_err());
    }
}
