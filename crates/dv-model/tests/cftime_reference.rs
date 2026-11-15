//! Checks CF time decoding in every calendar against cftime, and calendar conversion against
//! xarray's `convert_calendar`. The reference comes from `scripts/reference/cftime-calendars.py`.

use std::path::PathBuf;

use dv_model::time::{AlignOn, CalDate, Calendar, CfTime, convert_calendar};
use serde_json::Value as Json;

fn reference() -> Json {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference/cftime.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Parses the reference's `YYYY-MM-DDTHH:MM:SS[.ffffff]`.
fn date(s: &str) -> CalDate {
    let (d, t) = s.split_once('T').unwrap();
    let neg = d.starts_with('-');
    let mut p = d.trim_start_matches('-').split('-');
    let mut n = || p.next().unwrap().parse::<i64>().unwrap();
    let (y, m, day) = (n() * if neg { -1 } else { 1 }, n() as u32, n() as u32);
    let (hms, frac) = t.split_once('.').unwrap_or((t, "0"));
    let mut q = hms.split(':').map(|x| x.parse::<u32>().unwrap());
    let us = format!("{frac:0<6}").parse().unwrap();
    CalDate::new(y, m, day).with_time(q.next().unwrap(), q.next().unwrap(), q.next().unwrap(), us)
}

#[test]
fn decoding_matches_cftime() {
    let r = reference();
    let mut checked = 0;
    for case in r["decode"].as_array().unwrap() {
        let (cal, units) = (
            case["calendar"].as_str().unwrap(),
            case["units"].as_str().unwrap(),
        );
        let cf = CfTime::parse(units, cal).unwrap_or_else(|e| panic!("{units} {cal}: {e}"));
        let values = case["values"].as_array().unwrap();
        for (i, v) in values.iter().enumerate() {
            let v = v.as_f64().unwrap();
            let want = case["dates"][i].as_str().unwrap();
            let got = cf.decode(v).unwrap();
            assert_eq!(got.to_string(), want, "{v} {units} ({cal})");
            let back = cf.encode(got).unwrap();
            let enc = case["encoded"][i].as_f64().unwrap();
            assert!(
                (back - enc).abs() <= 1e-9 * enc.abs().max(1.0),
                "encode {want} ({cal}): {back} vs {enc}"
            );
            checked += 1;
        }
    }
    assert!(checked > 400);
}

#[test]
fn calendar_conversion_matches_xarray() {
    for case in reference()["convert"].as_array().unwrap() {
        let from: Calendar = case["from"].as_str().unwrap().parse().unwrap();
        let to: Calendar = case["to"].as_str().unwrap().parse().unwrap();
        let align = match case["align_on"].as_str() {
            Some("year") => Some(AlignOn::Year),
            Some("date") => Some(AlignOn::Date),
            _ => None,
        };
        let dates: Vec<CalDate> = case["dates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| date(d.as_str().unwrap()))
            .collect();
        let got = convert_calendar(&dates, from, to, align).unwrap();
        for (i, want) in case["converted"].as_array().unwrap().iter().enumerate() {
            let got = got[i].map(|d| d.to_string());
            assert_eq!(
                got.as_deref(),
                want.as_str(),
                "{from} -> {to} ({align:?}): {}",
                dates[i]
            );
        }
    }
}

#[test]
fn conversions_involving_360_day_need_an_alignment() {
    let d = [CalDate::new(2000, 1, 1)];
    assert!(convert_calendar(&d, Calendar::NoLeap, Calendar::Day360, None).is_err());
}
