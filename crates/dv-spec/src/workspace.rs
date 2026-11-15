//! The `.dvspec` workspace: sources, field fixes, derived fields and joins as recipes, views,
//! layout and selection, in TOML.
//!
//! A workspace never holds data or secrets. Credentials live in the keychain behind a named
//! `connection`, so [`Workspace::check`] refuses URLs with passwords and settings whose names say
//! they're secrets. Views and layout belong to the app, so they're kept as free-form tables.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The format version this engine reads and writes.
pub const VERSION: u32 = 0;

/// A workspace file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workspace {
    /// Format version.
    pub dvspec: u32,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sources: BTreeMap<String, Source>,
    /// Corrections to inferred roles and units.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fixes: Vec<FieldFix>,
    /// Derived fields, by name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fields: BTreeMap<String, Derived>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub joins: BTreeMap<String, Join>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub views: Vec<toml::Table>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<toml::Table>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<toml::Table>,
}

/// Where a dataset comes from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    /// A path, relative to the workspace or absolute, or a URL (`https`, `s3`, `gs`, `az`, `ssh`).
    pub url: String,
    /// A saved connection's name. Its credentials stay in the keychain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connection: Option<String>,
    /// The format, when the extension doesn't say (`csv`, `parquet`, `geojson`…).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Which part of a multi-part file: a sheet, table, layer or variable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
}

/// A user's correction to a field's role or unit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldFix {
    pub source: String,
    pub field: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

/// A derived field: a formula over one source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Derived {
    pub source: String,
    pub formula: String,
}

/// A join recipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "by", rename_all = "snake_case", deny_unknown_fields)]
pub enum Join {
    /// Equal keys.
    Key {
        left: String,
        right: String,
        left_keys: Vec<String>,
        right_keys: Vec<String>,
        /// `keep_all`, `first`, `last`, `mean`, `sum`, `min`, `max` or `count`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duplicates: Option<String>,
        /// The column `first` and `last` order by.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        order_by: Option<String>,
    },
    /// Nearest time within a tolerance.
    Time {
        left: String,
        right: String,
        left_time: String,
        right_time: String,
        /// Such as `10m`, `1h`, `30s`, `500ms` or `2d`.
        tolerance: String,
        /// `nearest`, `backward` or `forward`.
        #[serde(default = "nearest")]
        direction: String,
        /// Pairs of `[left, right]` columns that must also match.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        match_on: Vec<[String; 2]>,
    },
}

fn nearest() -> String {
    "nearest".into()
}

/// Something wrong with a workspace, and where.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{at}: {message}")]
pub struct Problem {
    pub at: String,
    pub message: String,
}

/// A workspace that couldn't be read.
#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    #[error("not a valid workspace: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("{} problem(s): {}", .0.len(), .0.iter().map(ToString::to_string).collect::<Vec<_>>().join("; "))]
    Invalid(Vec<Problem>),
}

const ROLES: [&str; 12] = [
    "time",
    "latitude",
    "longitude",
    "x",
    "y",
    "vertical",
    "geometry",
    "identifier",
    "category",
    "measure",
    "text",
    "other",
];
const DUPLICATES: [&str; 8] = [
    "keep_all", "first", "last", "mean", "sum", "min", "max", "count",
];
const SECRET_WORDS: [&str; 6] = ["password", "passwd", "secret", "token", "api_key", "apikey"];

/// A tolerance such as `10m` in microseconds.
pub fn parse_duration(s: &str) -> Option<i64> {
    let s = s.trim();
    let split = s.find(|c: char| !c.is_ascii_digit() && c != '.')?;
    let (n, unit) = s.split_at(split);
    let n: f64 = n.parse().ok()?;
    let us = match unit.trim() {
        "us" | "µs" => 1.0,
        "ms" => 1e3,
        "s" => 1e6,
        "m" | "min" => 60e6,
        "h" => 3600e6,
        "d" => 86400e6,
        _ => return None,
    };
    (n >= 0.0).then(|| (n * us).round() as i64)
}

impl Workspace {
    pub fn new() -> Self {
        Self {
            dvspec: VERSION,
            sources: BTreeMap::new(),
            fixes: Vec::new(),
            fields: BTreeMap::new(),
            joins: BTreeMap::new(),
            views: Vec::new(),
            layout: None,
            selection: None,
        }
    }

    /// Reads and checks a workspace.
    pub fn from_toml(text: &str) -> Result<Self, ReadError> {
        let w: Workspace = toml::from_str(text)?;
        let problems = w.check();
        if problems.is_empty() {
            Ok(w)
        } else {
            Err(ReadError::Invalid(problems))
        }
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("a workspace always serializes")
    }

    /// Everything wrong with the workspace: references to missing sources, unknown roles and
    /// units, incomplete joins, an unsupported version, and anything that looks like a secret.
    pub fn check(&self) -> Vec<Problem> {
        let mut out = Vec::new();
        let mut problem = |at: String, message: String| out.push(Problem { at, message });
        if self.dvspec != VERSION {
            problem(
                "dvspec".into(),
                format!(
                    "format version {} isn't supported (this engine reads {VERSION})",
                    self.dvspec
                ),
            );
        }
        let has = |s: &str| self.sources.contains_key(s);
        for (name, s) in &self.sources {
            if let Some(why) = url_secret(&s.url) {
                problem(format!("sources.{name}.url"), why);
            }
        }
        for (i, f) in self.fixes.iter().enumerate() {
            let at = format!("fixes[{i}]");
            if !has(&f.source) {
                problem(at.clone(), format!("no source {:?}", f.source));
            }
            if let Some(r) = &f.role
                && !ROLES.contains(&r.as_str())
            {
                problem(at.clone(), format!("unknown role {r:?}"));
            }
            if let Some(u) = &f.unit
                && let Err(e) = dv_model::units::Unit::parse(u)
            {
                problem(at.clone(), e.to_string());
            }
            if f.role.is_none() && f.unit.is_none() {
                problem(at, "a fix needs a role or a unit".into());
            }
        }
        for (name, d) in &self.fields {
            if !has(&d.source) {
                problem(
                    format!("fields.{name}"),
                    format!("no source {:?}", d.source),
                );
            }
            if d.formula.trim().is_empty() {
                problem(format!("fields.{name}"), "empty formula".into());
            }
        }
        for (name, j) in &self.joins {
            let at = format!("joins.{name}");
            let (left, right) = match j {
                Join::Key { left, right, .. } | Join::Time { left, right, .. } => (left, right),
            };
            for s in [left, right] {
                if !has(s) && !self.joins.contains_key(s) {
                    problem(at.clone(), format!("no source or join {s:?}"));
                }
            }
            match j {
                Join::Key {
                    left_keys,
                    right_keys,
                    duplicates,
                    order_by,
                    ..
                } => {
                    if left_keys.is_empty() || left_keys.len() != right_keys.len() {
                        problem(
                            at.clone(),
                            "needs the same number of keys on each side".into(),
                        );
                    }
                    if let Some(d) = duplicates {
                        if !DUPLICATES.contains(&d.as_str()) {
                            problem(at.clone(), format!("unknown duplicates policy {d:?}"));
                        }
                        if matches!(d.as_str(), "first" | "last") && order_by.is_none() {
                            problem(at.clone(), format!("{d:?} needs order_by"));
                        }
                    }
                }
                Join::Time {
                    tolerance,
                    direction,
                    ..
                } => {
                    if parse_duration(tolerance).is_none() {
                        problem(
                            at.clone(),
                            format!("can't read the tolerance {tolerance:?}"),
                        );
                    }
                    if !matches!(direction.as_str(), "nearest" | "backward" | "forward") {
                        problem(at.clone(), format!("unknown direction {direction:?}"));
                    }
                }
            }
        }
        let tables = self
            .views
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("views[{i}]"), v))
            .chain(self.layout.iter().map(|l| ("layout".to_string(), l)))
            .chain(self.selection.iter().map(|s| ("selection".to_string(), s)));
        for (at, t) in tables {
            secret_keys(&at, t, &mut out);
        }
        out
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Why a URL holds a secret, if it does: a password in it, or a credential-like query parameter.
fn url_secret(url: &str) -> Option<String> {
    if let Some((_, rest)) = url.split_once("://") {
        let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
        if let Some((userinfo, _)) = authority.rsplit_once('@')
            && userinfo.contains(':')
        {
            return Some("URLs can't hold passwords; save a connection instead".into());
        }
    }
    let query = url
        .split_once('?')
        .map(|(_, q)| q.to_ascii_lowercase())
        .unwrap_or_default();
    query
        .split('&')
        .filter_map(|kv| kv.split('=').next())
        .find(|k| {
            SECRET_WORDS.iter().any(|w| k.contains(w)) || *k == "x-amz-signature" || *k == "sig"
        })
        .map(|k| format!("the URL has a credential ({k}); save a connection instead"))
}

fn secret_keys(at: &str, t: &toml::Table, out: &mut Vec<Problem>) {
    for (k, v) in t {
        let lower = k.to_ascii_lowercase();
        if SECRET_WORDS.iter().any(|w| lower.contains(w)) {
            out.push(Problem {
                at: format!("{at}.{k}"),
                message: "workspaces never hold secrets".into(),
            });
        }
        if let toml::Value::Table(inner) = v {
            secret_keys(&format!("{at}.{k}"), inner, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations() {
        assert_eq!(parse_duration("10m"), Some(600_000_000));
        assert_eq!(parse_duration("1.5h"), Some(5_400_000_000));
        assert_eq!(parse_duration("500ms"), Some(500_000));
        assert_eq!(parse_duration("10"), None);
        assert_eq!(parse_duration("3 weeks"), None);
    }

    #[test]
    fn secrets_are_refused() {
        assert!(url_secret("https://user:hunter2@example.com/x.csv").is_some());
        assert!(url_secret("https://example.com/x.csv?X-Amz-Signature=abc").is_some());
        assert!(url_secret("https://user@example.com/x.csv").is_none());
        assert!(url_secret("s3://bucket/key.parquet").is_none());
    }
}
