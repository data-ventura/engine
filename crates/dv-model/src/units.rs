//! Physical units, parsed from UDUNITS-style strings such as `kg m-2 s-1`, `m/s` or `hPa`.
//!
//! Values are combined or compared only when their units are convertible: same dimension in SI
//! base units. A string the parser doesn't know gives an error, and callers treat that field's
//! unit as unknown, which blocks arithmetic until someone assigns one.
//!
//! Temperatures with an offset (`degC`, `degF`) convert with their offset only when they stand
//! alone. Inside a product (`degC/day`) they're intervals, as in UDUNITS.

use std::fmt;

/// Exponents of the SI base units, in the order m, kg, s, K, A, mol, cd.
pub type Dims = [i8; 7];

const BASE: [&str; 7] = ["m", "kg", "s", "K", "A", "mol", "cd"];

/// A parsed unit: `si = value * scale + offset`.
#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    text: String,
    dims: Dims,
    scale: f64,
    offset: f64,
}

/// A unit string the parser couldn't read.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("unknown unit {text:?}: {reason}")]
pub struct UnitError {
    pub text: String,
    pub reason: String,
}

impl Unit {
    /// Parses a UDUNITS-style unit string.
    pub fn parse(text: &str) -> Result<Self, UnitError> {
        let err = |reason: String| UnitError {
            text: text.to_string(),
            reason,
        };
        let trimmed = text.trim();
        if trimmed.contains(" since ") {
            return Err(err(
                "a time reference, not a unit (see dv_model::time)".into()
            ));
        }
        let tokens = tokenize(trimmed).map_err(err)?;
        let mut p = Parser { tokens, pos: 0 };
        let mut f = if p.tokens.is_empty() {
            Factor::ONE
        } else {
            p.product().map_err(err)?
        };
        if p.pos != p.tokens.len() {
            return Err(err(format!("unexpected {:?}", p.tokens[p.pos])));
        }
        if !f.lone_offset {
            f.offset = 0.0;
        }
        Ok(Unit {
            text: trimmed.to_string(),
            dims: f.dims,
            scale: f.scale,
            offset: f.offset,
        })
    }

    /// The string this unit was parsed from.
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn dims(&self) -> Dims {
        self.dims
    }

    pub fn is_dimensionless(&self) -> bool {
        self.dims == [0; 7]
    }

    /// Whether values in `self` can be converted to `other`.
    pub fn convertible(&self, other: &Unit) -> bool {
        self.dims == other.dims
    }

    /// A converter from `self` to `other`, or `None` if their dimensions differ.
    pub fn converter(&self, other: &Unit) -> Option<Converter> {
        self.convertible(other).then(|| Converter {
            scale: self.scale / other.scale,
            offset: (self.offset - other.offset) / other.scale,
        })
    }

    /// The unit in SI base units, such as `kg m-2 s-1`, ignoring scale and offset.
    pub fn base(&self) -> String {
        let parts: Vec<String> = BASE
            .iter()
            .zip(self.dims)
            .filter(|(_, e)| *e != 0)
            .map(|(s, e)| {
                if e == 1 {
                    s.to_string()
                } else {
                    format!("{s}{e}")
                }
            })
            .collect();
        if parts.is_empty() {
            "1".into()
        } else {
            parts.join(" ")
        }
    }

    /// The product of two units. Offsets don't carry into products.
    pub fn mul(&self, other: &Unit) -> Unit {
        self.combine(other, 1, format!("{} {}", self.wrapped(), other.wrapped()))
    }

    /// The quotient of two units.
    pub fn div(&self, other: &Unit) -> Unit {
        self.combine(other, -1, format!("{}/{}", self.wrapped(), other.wrapped()))
    }

    /// The unit raised to an integer power.
    pub fn powi(&self, n: i32) -> Unit {
        Unit {
            text: format!("{}^{n}", self.wrapped()),
            dims: self.dims.map(|d| d * n as i8),
            scale: self.scale.powi(n),
            offset: 0.0,
        }
    }

    /// The square root, if every dimension's exponent is even.
    pub fn sqrt(&self) -> Option<Unit> {
        self.dims.iter().all(|d| d % 2 == 0).then(|| Unit {
            text: format!("sqrt({})", self.text),
            dims: self.dims.map(|d| d / 2),
            scale: self.scale.sqrt(),
            offset: 0.0,
        })
    }

    fn combine(&self, other: &Unit, sign: i8, text: String) -> Unit {
        let mut dims = self.dims;
        for (d, e) in dims.iter_mut().zip(other.dims) {
            *d += sign * e;
        }
        Unit {
            text,
            dims,
            scale: self.scale * other.scale.powi(sign as i32),
            offset: 0.0,
        }
    }

    /// The text, in parentheses if it has more than one factor.
    fn wrapped(&self) -> String {
        if self.text.contains([' ', '/', '.', '*', '^']) {
            format!("({})", self.text)
        } else {
            self.text.clone()
        }
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

/// Converts values between two convertible units: `to = from * scale + offset`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Converter {
    pub scale: f64,
    pub offset: f64,
}

impl Converter {
    pub fn apply(&self, v: f64) -> f64 {
        v * self.scale + self.offset
    }

    pub fn is_identity(&self) -> bool {
        self.scale == 1.0 && self.offset == 0.0
    }
}

#[derive(Debug, Clone, Copy)]
struct Factor {
    dims: Dims,
    scale: f64,
    offset: f64,
    /// True while the factor is a single offset unit, so its offset still applies.
    lone_offset: bool,
}

impl Factor {
    const ONE: Factor = Factor {
        dims: [0; 7],
        scale: 1.0,
        offset: 0.0,
        lone_offset: false,
    };

    fn mul(self, o: Factor, sign: i8) -> Factor {
        let mut dims = self.dims;
        for (d, e) in dims.iter_mut().zip(o.dims) {
            *d += sign * e;
        }
        Factor {
            dims,
            scale: self.scale * o.scale.powi(sign as i32),
            offset: 0.0,
            lone_offset: false,
        }
    }

    fn pow(self, n: i32) -> Factor {
        if n == 1 {
            return self;
        }
        Factor {
            dims: self.dims.map(|d| d * n as i8),
            scale: self.scale.powi(n),
            offset: 0.0,
            lone_offset: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    /// A unit symbol with its exponent (`m2`, `s-1`).
    Name(String, i32),
    Mul,
    Div,
    Pow,
    Open,
    Close,
}

fn tokenize(s: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            c if c.is_whitespace() => {
                i += 1;
                continue;
            }
            '*' if chars.get(i + 1) == Some(&'*') => {
                out.push(Token::Pow);
                i += 2;
                continue;
            }
            '*' | '.' | '·' => out.push(Token::Mul),
            '/' => out.push(Token::Div),
            '^' => out.push(Token::Pow),
            '(' => out.push(Token::Open),
            ')' => out.push(Token::Close),
            c if c.is_ascii_digit() || ((c == '-' || c == '+') && next_is_digit(&chars, i)) => {
                let start = i;
                i += 1;
                while i < chars.len() {
                    let d = chars[i];
                    let exp_sign = (d == '-' || d == '+') && matches!(chars[i - 1], 'e' | 'E');
                    if d.is_ascii_digit() || d == '.' || d == 'e' || d == 'E' || exp_sign {
                        i += 1;
                    } else {
                        break;
                    }
                }
                let text: String = chars[start..i].iter().collect();
                let n = text.parse().map_err(|_| format!("bad number {text:?}"))?;
                out.push(Token::Number(n));
                continue;
            }
            c if is_name_char(c) => {
                let start = i;
                while i < chars.len() && is_name_char(chars[i]) {
                    i += 1;
                }
                let name: String = chars[start..i].iter().collect();
                let exp_start = i;
                if i < chars.len() && (chars[i] == '-' || chars[i] == '+') {
                    i += 1;
                }
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
                let exp = if i > exp_start {
                    let text: String = chars[exp_start..i].iter().collect();
                    text.parse()
                        .map_err(|_| format!("bad exponent in {name}{text}"))?
                } else {
                    1
                };
                out.push(Token::Name(name, exp));
                continue;
            }
            c => return Err(format!("unexpected character {c:?}")),
        }
        i += 1;
    }
    Ok(out)
}

fn next_is_digit(chars: &[char], i: usize) -> bool {
    chars.get(i + 1).is_some_and(|c| c.is_ascii_digit())
}

fn is_name_char(c: char) -> bool {
    c.is_alphabetic() || c == '_' || c == '%' || c == '°' || c == 'µ'
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    /// product := power ((Mul | Div | juxtaposition) power)*
    fn product(&mut self) -> Result<Factor, String> {
        let mut f = self.power()?;
        loop {
            match self.peek() {
                Some(Token::Mul) => {
                    self.pos += 1;
                    f = f.mul(self.power()?, 1);
                }
                Some(Token::Div) => {
                    self.pos += 1;
                    f = f.mul(self.power()?, -1);
                }
                Some(Token::Number(_) | Token::Name(..) | Token::Open) => {
                    f = f.mul(self.power()?, 1);
                }
                _ => return Ok(f),
            }
        }
    }

    /// power := atom (Pow Number)?
    fn power(&mut self) -> Result<Factor, String> {
        let f = self.atom()?;
        if self.peek() == Some(&Token::Pow) {
            self.pos += 1;
            match self.tokens.get(self.pos) {
                Some(Token::Number(n)) if n.fract() == 0.0 => {
                    self.pos += 1;
                    return Ok(f.pow(*n as i32));
                }
                other => return Err(format!("expected an integer exponent, found {other:?}")),
            }
        }
        Ok(f)
    }

    fn atom(&mut self) -> Result<Factor, String> {
        let token = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        match token {
            Some(Token::Number(n)) => Ok(Factor {
                scale: n,
                ..Factor::ONE
            }),
            Some(Token::Name(name, exp)) => lookup(&name)
                .map(|f| f.pow(exp))
                .ok_or_else(|| format!("unknown symbol {name:?}")),
            Some(Token::Open) => {
                let f = self.product()?;
                if self.peek() != Some(&Token::Close) {
                    return Err("missing )".into());
                }
                self.pos += 1;
                Ok(f)
            }
            other => Err(format!("unexpected {other:?}")),
        }
    }
}

const fn d(m: i8, kg: i8, s: i8, k: i8) -> Dims {
    [m, kg, s, k, 0, 0, 0]
}

const NONE: Dims = [0; 7];
const DEG: f64 = std::f64::consts::PI / 180.0;

/// Symbols and names: (names, scale, dims, offset, takes SI prefixes).
#[rustfmt::skip]
const UNITS: &[(&[&str], f64, Dims, f64, bool)] = &[
    (&["m", "meter", "metre"], 1.0, d(1, 0, 0, 0), 0.0, true),
    (&["g", "gram", "gramme"], 1e-3, d(0, 1, 0, 0), 0.0, true),
    (&["s", "sec", "second"], 1.0, d(0, 0, 1, 0), 0.0, true),
    (&["K", "kelvin"], 1.0, d(0, 0, 0, 1), 0.0, true),
    (&["A", "ampere"], 1.0, [0, 0, 0, 0, 1, 0, 0], 0.0, true),
    (&["mol", "mole"], 1.0, [0, 0, 0, 0, 0, 1, 0], 0.0, true),
    (&["cd", "candela"], 1.0, [0, 0, 0, 0, 0, 0, 1], 0.0, true),
    (&["min", "minute"], 60.0, d(0, 0, 1, 0), 0.0, false),
    (&["h", "hr", "hour"], 3600.0, d(0, 0, 1, 0), 0.0, false),
    (&["d", "day"], 86400.0, d(0, 0, 1, 0), 0.0, false),
    (&["week"], 604800.0, d(0, 0, 1, 0), 0.0, false),
    // UDUNITS' year is the tropical year.
    (&["yr", "year", "a"], 3.15569259747e7, d(0, 0, 1, 0), 0.0, false),
    (&["common_year"], 31536000.0, d(0, 0, 1, 0), 0.0, false),
    (&["degC", "deg_C", "degree_Celsius", "degrees_Celsius", "celsius", "°C"], 1.0, d(0, 0, 0, 1), 273.15, false),
    (&["degF", "deg_F", "degree_Fahrenheit", "degrees_Fahrenheit", "fahrenheit", "°F"], 5.0 / 9.0, d(0, 0, 0, 1), 459.67 * 5.0 / 9.0, false),
    (&["degR", "rankine"], 5.0 / 9.0, d(0, 0, 0, 1), 0.0, false),
    (&["Hz", "hertz"], 1.0, d(0, 0, -1, 0), 0.0, true),
    (&["N", "newton"], 1.0, d(1, 1, -2, 0), 0.0, true),
    (&["Pa", "pascal"], 1.0, d(-1, 1, -2, 0), 0.0, true),
    (&["bar"], 1e5, d(-1, 1, -2, 0), 0.0, true),
    (&["atm", "atmosphere"], 101325.0, d(-1, 1, -2, 0), 0.0, false),
    (&["J", "joule"], 1.0, d(2, 1, -2, 0), 0.0, true),
    (&["W", "watt"], 1.0, d(2, 1, -3, 0), 0.0, true),
    (&["C", "coulomb"], 1.0, [0, 0, 1, 0, 1, 0, 0], 0.0, true),
    (&["V", "volt"], 1.0, [2, 1, -3, 0, -1, 0, 0], 0.0, true),
    (&["L", "l", "liter", "litre"], 1e-3, d(3, 0, 0, 0), 0.0, true),
    (&["t", "tonne"], 1e3, d(0, 1, 0, 0), 0.0, false),
    (&["ha", "hectare"], 1e4, d(2, 0, 0, 0), 0.0, false),
    (&["in", "inch"], 0.0254, d(1, 0, 0, 0), 0.0, false),
    (&["ft", "foot", "feet"], 0.3048, d(1, 0, 0, 0), 0.0, false),
    (&["mi", "mile"], 1609.344, d(1, 0, 0, 0), 0.0, false),
    (&["nmi", "nautical_mile"], 1852.0, d(1, 0, 0, 0), 0.0, false),
    (&["kt", "kn", "knot"], 1852.0 / 3600.0, d(1, 0, -1, 0), 0.0, false),
    (&["mph"], 1609.344 / 3600.0, d(1, 0, -1, 0), 0.0, false),
    (&["lb", "pound"], 0.45359237, d(0, 1, 0, 0), 0.0, false),
    (&["1"], 1.0, NONE, 0.0, false),
    (&["%", "percent"], 0.01, NONE, 0.0, false),
    (&["ppm"], 1e-6, NONE, 0.0, false),
    (&["ppb"], 1e-9, NONE, 0.0, false),
    (&["rad", "radian"], 1.0, NONE, 0.0, false),
    (&["sr", "steradian"], 1.0, NONE, 0.0, false),
    (&["°", "deg", "degree", "arc_degree",
       "degrees_north", "degree_north", "degree_N", "degrees_N", "degreeN", "degreesN",
       "degrees_east", "degree_east", "degree_E", "degrees_E", "degreeE", "degreesE"], DEG, NONE, 0.0, false),
];

#[rustfmt::skip]
const PREFIXES: &[(&str, &str, f64)] = &[
    ("Y", "yotta", 1e24), ("Z", "zetta", 1e21), ("E", "exa", 1e18), ("P", "peta", 1e15),
    ("T", "tera", 1e12), ("G", "giga", 1e9), ("M", "mega", 1e6), ("k", "kilo", 1e3),
    ("h", "hecto", 1e2), ("da", "deka", 1e1), ("d", "deci", 1e-1), ("c", "centi", 1e-2),
    ("m", "milli", 1e-3), ("u", "micro", 1e-6), ("µ", "micro", 1e-6), ("n", "nano", 1e-9),
    ("p", "pico", 1e-12), ("f", "femto", 1e-15), ("a", "atto", 1e-18),
];

fn exact(name: &str) -> Option<(Factor, bool, bool)> {
    UNITS
        .iter()
        .find_map(|(names, scale, dims, offset, prefixed)| {
            names.iter().position(|n| *n == name).map(|i| {
                let f = Factor {
                    dims: *dims,
                    scale: *scale,
                    offset: *offset,
                    lone_offset: *offset != 0.0,
                };
                // Names longer than their symbol take long prefixes (kilometre), symbols short ones (km).
                (f, *prefixed, i > 0 && names[0].len() < names[i].len())
            })
        })
}

fn lookup(name: &str) -> Option<Factor> {
    let singular = |n: &str| {
        n.strip_suffix('s')
            .filter(|s| s.len() > 2)
            .map(str::to_string)
    };
    let find = |n: &str| exact(n).or_else(|| singular(n).and_then(|s| exact(&s)));
    if let Some((f, _, _)) = find(name) {
        return Some(f);
    }
    for (short, long, scale) in PREFIXES {
        for (prefix, wants_long) in [(*long, true), (*short, false)] {
            if let Some(rest) = name.strip_prefix(prefix)
                && !rest.is_empty()
                && let Some((f, true, is_long)) = find(rest)
                && is_long == wants_long
            {
                return Some(Factor {
                    scale: f.scale * scale,
                    ..f
                });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(from: &str, to: &str, v: f64) -> f64 {
        let a = Unit::parse(from).unwrap();
        let b = Unit::parse(to).unwrap();
        a.converter(&b).unwrap().apply(v)
    }

    #[test]
    fn parses_udunits_spellings() {
        for s in [
            "kg m-2 s-1",
            "kg/m2/s",
            "kg.m^-2.s^-1",
            "kg m**-2 s**-1",
            "kg/(m2 s)",
        ] {
            assert_eq!(Unit::parse(s).unwrap().base(), "m-2 kg s-1", "{s}");
        }
        assert_eq!(Unit::parse("W m-2").unwrap().base(), "kg s-3");
        assert_eq!(Unit::parse("").unwrap().base(), "1");
        assert!(Unit::parse("1").unwrap().is_dimensionless());
        assert!(Unit::parse("degrees_north").unwrap().is_dimensionless());
    }

    #[test]
    fn prefixes_and_names() {
        assert_eq!(conv("km", "m", 1.0), 1000.0);
        assert_eq!(conv("kilometres", "m", 2.0), 2000.0);
        assert_eq!(conv("hPa", "Pa", 1.0), 100.0);
        assert_eq!(conv("mbar", "hPa", 1.0), 1.0);
        assert_eq!(conv("min", "s", 1.0), 60.0);
        assert_eq!(conv("ms", "s", 1.0), 1e-3);
        assert!(
            Unit::parse("kmetre").is_err(),
            "short prefix on a long name"
        );
    }

    #[test]
    fn temperature_offsets_only_stand_alone() {
        assert!((conv("degC", "K", 0.0) - 273.15).abs() < 1e-12);
        assert!((conv("degF", "degC", 212.0) - 100.0).abs() < 1e-12);
        // An interval: 1 degC/day is 1 K/day.
        assert_eq!(conv("degC/day", "K/day", 1.0), 1.0);
    }

    #[test]
    fn rejects_unknown_and_inconvertible() {
        assert!(Unit::parse("furlongs").is_err());
        assert!(Unit::parse("days since 1970-01-01").is_err());
        let m = Unit::parse("m").unwrap();
        assert!(m.converter(&Unit::parse("s").unwrap()).is_none());
    }
}
