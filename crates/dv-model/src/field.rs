//! Datasets and fields, with the role each field plays.
//!
//! A role says how a field is used: as time, as a coordinate, as a measure to plot or as a
//! category to split by. Roles are inferred from the field's type, name and CF attributes, and the
//! inference records its reason so the UI can say why. A user can fix a role, and a fixed role is
//! never inferred again.

use std::collections::HashMap;

use arrow::datatypes::{DataType, Schema};
use serde::Serialize;

use crate::units::Unit;

/// What a dataset is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Family {
    /// Rows and columns.
    Table,
    /// N-dimensional arrays on named dimensions.
    Array,
    /// Features with geometry.
    Geo,
}

/// How a field is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Time,
    Latitude,
    Longitude,
    /// A projected or other horizontal x coordinate.
    X,
    /// A projected or other horizontal y coordinate.
    Y,
    /// A vertical coordinate: height, depth or pressure level.
    Vertical,
    Geometry,
    /// A key that identifies a row or an entity, for joining and lookup.
    Identifier,
    Category,
    Measure,
    Text,
    Other,
}

/// Where a field's role came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum RoleSource {
    /// Inferred, with a short reason such as "named like a latitude".
    Inferred { reason: &'static str },
    /// Set by the user. Kept as is.
    Fixed,
}

/// A field (a table column or an array variable or coordinate).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Field {
    pub name: String,
    #[serde(skip)]
    pub data_type: DataType,
    pub role: Role,
    pub role_source: RoleSource,
    /// The unit as written in the source, if any.
    pub unit_text: Option<String>,
    /// The parsed unit; `None` when there's no unit or it couldn't be parsed.
    #[serde(skip)]
    pub unit: Option<Unit>,
    /// Source attributes, such as CF `standard_name`, `long_name` or `axis`.
    pub attrs: HashMap<String, String>,
}

impl Field {
    /// A field with its role inferred from its type, name and attributes.
    pub fn new(
        name: impl Into<String>,
        data_type: DataType,
        attrs: HashMap<String, String>,
    ) -> Self {
        let name = name.into();
        let unit_text = attrs.get("units").cloned();
        let unit = unit_text.as_deref().and_then(|u| Unit::parse(u).ok());
        let (role, reason) = infer_role(&name, &data_type, &attrs);
        Field {
            name,
            data_type,
            role,
            role_source: RoleSource::Inferred { reason },
            unit_text,
            unit,
            attrs,
        }
    }

    /// Whether the field has a unit the engine can't read. Arithmetic on it is blocked.
    pub fn unit_unknown(&self) -> bool {
        self.unit_text.is_some() && self.unit.is_none()
    }
}

/// A dimension of an array dataset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Dim {
    pub name: String,
    pub size: u64,
}

/// An opened dataset: its fields and, for arrays, its dimensions.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Dataset {
    pub name: String,
    pub family: Family,
    pub fields: Vec<Field>,
    /// Row count, when the source knows it without a scan.
    pub rows: Option<u64>,
    pub dims: Vec<Dim>,
}

impl Dataset {
    /// A table or geo dataset from an Arrow schema. Field metadata supplies attributes (`units`, …).
    pub fn from_schema(name: impl Into<String>, family: Family, schema: &Schema) -> Self {
        let fields = schema
            .fields()
            .iter()
            .map(|f| Field::new(f.name(), f.data_type().clone(), f.metadata().clone()))
            .collect();
        Dataset {
            name: name.into(),
            family,
            fields,
            rows: None,
            dims: Vec::new(),
        }
    }

    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Fields in a role, in order.
    pub fn with_role(&self, role: Role) -> impl Iterator<Item = &Field> {
        self.fields.iter().filter(move |f| f.role == role)
    }

    /// Fixes a field's role. Returns false if there's no such field.
    pub fn fix_role(&mut self, name: &str, role: Role) -> bool {
        match self.fields.iter_mut().find(|f| f.name == name) {
            Some(f) => {
                f.role = role;
                f.role_source = RoleSource::Fixed;
                true
            }
            None => false,
        }
    }
}

/// Infers a field's role. CF attributes win over names, and names over types.
pub fn infer_role(
    name: &str,
    dt: &DataType,
    attrs: &HashMap<String, String>,
) -> (Role, &'static str) {
    let attr = |k: &str| attrs.get(k).map(|v| v.trim().to_ascii_lowercase());
    let units = attr("units").unwrap_or_default();
    let standard = attr("standard_name").unwrap_or_default();
    let axis = attr("axis").unwrap_or_default();

    if attrs
        .get("ARROW:extension:name")
        .is_some_and(|e| e.starts_with("geoarrow."))
    {
        return (Role::Geometry, "a GeoArrow geometry");
    }
    if standard == "time" || axis == "t" || units.contains(" since ") {
        return (Role::Time, "CF time attributes");
    }
    if standard == "latitude"
        || matches!(
            units.as_str(),
            "degrees_north" | "degree_north" | "degree_n" | "degrees_n"
        )
    {
        return (Role::Latitude, "CF latitude attributes");
    }
    if standard == "longitude"
        || matches!(
            units.as_str(),
            "degrees_east" | "degree_east" | "degree_e" | "degrees_e"
        )
    {
        return (Role::Longitude, "CF longitude attributes");
    }
    if standard == "projection_x_coordinate" || axis == "x" {
        return (Role::X, "CF x-axis attributes");
    }
    if standard == "projection_y_coordinate" || axis == "y" {
        return (Role::Y, "CF y-axis attributes");
    }
    if axis == "z"
        || matches!(
            standard.as_str(),
            "height" | "depth" | "altitude" | "air_pressure"
        )
        || attr("positive").is_some()
    {
        return (Role::Vertical, "CF vertical attributes");
    }

    let words = words(name);
    let last = words.last().map(String::as_str).unwrap_or("");
    let numeric = dt.is_numeric();
    if matches!(
        dt,
        DataType::Timestamp(..) | DataType::Date32 | DataType::Date64
    ) {
        return (Role::Time, "a date or timestamp type");
    }
    if is_geometry_type(dt) && matches!(last, "geometry" | "geom" | "wkb" | "shape") {
        return (Role::Geometry, "named like a geometry");
    }
    if numeric {
        match last {
            "lat" | "latitude" => return (Role::Latitude, "named like a latitude"),
            "lon" | "lng" | "long" | "longitude" => {
                return (Role::Longitude, "named like a longitude");
            }
            _ => {}
        }
    }
    if matches!(
        dt,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    ) && matches!(last, "time" | "date" | "datetime" | "timestamp")
    {
        return (Role::Time, "named like a time");
    }
    if (dt.is_integer() || is_text(dt)) && (last == "id" || last == "key" || last == "code") {
        return (Role::Identifier, "named like an identifier");
    }
    if matches!(dt, DataType::Boolean) || is_text(dt) {
        return (Role::Category, "text or boolean");
    }
    if numeric {
        return (Role::Measure, "a number");
    }
    (Role::Other, "an unsupported type")
}

fn is_text(dt: &DataType) -> bool {
    match dt {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => true,
        DataType::Dictionary(_, v) => is_text(v),
        _ => false,
    }
}

fn is_geometry_type(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView
    ) || matches!(dt, DataType::Struct(_) | DataType::List(_))
}

/// Splits a name into lowercase words at `_`, `-`, spaces and camelCase boundaries:
/// `tpepPickupDateTime` → `tpep pickup date time`, `puLocationId` → `pu location id`.
fn words(name: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = name.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if !c.is_alphanumeric() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        let boundary = i > 0
            && c.is_uppercase()
            && (chars[i - 1].is_lowercase()
                || chars.get(i + 1).is_some_and(|n| n.is_lowercase())
                    && chars[i - 1].is_uppercase());
        if boundary && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        cur.extend(c.to_lowercase());
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::TimeUnit;

    fn role(name: &str, dt: DataType) -> Role {
        infer_role(name, &dt, &HashMap::new()).0
    }

    fn role_with(name: &str, dt: DataType, attrs: &[(&str, &str)]) -> Role {
        let attrs = attrs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        infer_role(name, &dt, &attrs).0
    }

    #[test]
    fn splits_names() {
        assert_eq!(
            words("tpepPickupDateTime"),
            ["tpep", "pickup", "date", "time"]
        );
        assert_eq!(words("puLocationId"), ["pu", "location", "id"]);
        assert_eq!(words("vendorID"), ["vendor", "id"]);
        assert_eq!(words("start_lat"), ["start", "lat"]);
        assert_eq!(words("HTTPStatus"), ["http", "status"]);
    }

    #[test]
    fn infers_taxi_roles() {
        use DataType::*;
        assert_eq!(
            role("tpepPickupDateTime", Timestamp(TimeUnit::Microsecond, None)),
            Role::Time
        );
        assert_eq!(role("startLat", Float64), Role::Latitude);
        assert_eq!(role("endLon", Float64), Role::Longitude);
        assert_eq!(role("vendorID", Utf8), Role::Identifier);
        assert_eq!(role("paymentType", Utf8), Role::Category);
        assert_eq!(role("fareAmount", Float64), Role::Measure);
        assert_eq!(role("latency", Float64), Role::Measure, "not a latitude");
    }

    #[test]
    fn cf_attributes_win() {
        use DataType::*;
        assert_eq!(
            role_with("y", Float64, &[("units", "degrees_north")]),
            Role::Latitude
        );
        assert_eq!(
            role_with("t", Float64, &[("units", "hours since 1900-01-01")]),
            Role::Time
        );
        assert_eq!(
            role_with("lev", Float64, &[("positive", "down")]),
            Role::Vertical
        );
        assert_eq!(role_with("xc", Float64, &[("axis", "X")]), Role::X);
    }

    #[test]
    fn fixed_roles_stay() {
        let schema = Schema::new(vec![arrow::datatypes::Field::new(
            "code",
            DataType::Utf8,
            true,
        )]);
        let mut ds = Dataset::from_schema("t", Family::Table, &schema);
        assert_eq!(ds.fields[0].role, Role::Identifier);
        assert!(ds.fix_role("code", Role::Category));
        assert_eq!(ds.fields[0].role_source, RoleSource::Fixed);
        assert!(!ds.fix_role("missing", Role::Category));
    }

    #[test]
    fn unknown_units_are_flagged() {
        let attrs = HashMap::from([("units".to_string(), "furlongs".to_string())]);
        assert!(Field::new("x", DataType::Float64, attrs).unit_unknown());
    }
}
