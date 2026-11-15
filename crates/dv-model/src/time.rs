//! CF time: numbers such as `hours since 1900-01-01` in any CF calendar.
//!
//! Values stay in their native calendar. Only the standard and proleptic Gregorian calendars name
//! real instants, so only they convert to Unix timestamps; the others (`noleap`, `all_leap`,
//! `360_day`, `julian`) decode to [`CalDate`]s in their own calendar. Moving dates between
//! calendars is explicit ([`convert_calendar`]) and follows xarray's `convert_calendar`.

use std::fmt;
use std::str::FromStr;

/// A CF calendar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Calendar {
    /// Julian before 1582-10-15, Gregorian from then on (CF `standard`, `gregorian`).
    Standard,
    ProlepticGregorian,
    Julian,
    /// 365 days every year (CF `noleap`, `365_day`).
    NoLeap,
    /// 366 days every year (CF `all_leap`, `366_day`).
    AllLeap,
    /// Twelve 30-day months.
    Day360,
}

impl FromStr for Calendar {
    type Err = TimeError;

    fn from_str(s: &str) -> Result<Self, TimeError> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "standard" | "gregorian" | "" => Calendar::Standard,
            "proleptic_gregorian" => Calendar::ProlepticGregorian,
            "julian" => Calendar::Julian,
            "noleap" | "365_day" => Calendar::NoLeap,
            "all_leap" | "366_day" => Calendar::AllLeap,
            "360_day" => Calendar::Day360,
            other => return Err(TimeError(format!("unknown calendar {other:?}"))),
        })
    }
}

impl Calendar {
    /// Whether dates in this calendar are real instants that convert to Unix time.
    pub fn is_real(self) -> bool {
        matches!(self, Calendar::Standard | Calendar::ProlepticGregorian)
    }

    /// Whether the calendar has a year zero. As in cftime, the standard and Julian calendars
    /// don't (1 BC, year -1, is followed by AD 1); the others, proleptic Gregorian included as in
    /// ISO 8601, do.
    pub fn has_year_zero(self) -> bool {
        !matches!(self, Calendar::Standard | Calendar::Julian)
    }

    /// The astronomical year (with a year zero) of a year in this calendar.
    fn astronomical(self, year: i64) -> i64 {
        if !self.has_year_zero() && year < 0 {
            year + 1
        } else {
            year
        }
    }

    fn year_of_astronomical(self, year: i64) -> i64 {
        if !self.has_year_zero() && year <= 0 {
            year - 1
        } else {
            year
        }
    }

    pub fn is_leap(self, year: i64) -> bool {
        let year = self.astronomical(year);
        match self {
            Calendar::NoLeap | Calendar::Day360 => false,
            Calendar::AllLeap => true,
            Calendar::Julian => year.rem_euclid(4) == 0,
            Calendar::ProlepticGregorian => gregorian_leap(year),
            Calendar::Standard => {
                if year < 1582 {
                    year.rem_euclid(4) == 0
                } else {
                    gregorian_leap(year)
                }
            }
        }
    }

    pub fn days_in_month(self, year: i64, month: u32) -> u32 {
        if self == Calendar::Day360 {
            return 30;
        }
        match month {
            2 if self.is_leap(year) => 29,
            2 => 28,
            4 | 6 | 9 | 11 => 30,
            _ => 31,
        }
    }

    pub fn days_in_year(self, year: i64) -> u32 {
        match self {
            Calendar::Day360 => 360,
            Calendar::Standard if year == 1582 => 355,
            _ if self.is_leap(year) => 366,
            _ => 365,
        }
    }

    /// Whether a date exists in this calendar.
    pub fn is_valid(self, year: i64, month: u32, day: u32) -> bool {
        (year != 0 || self.has_year_zero())
            && (1..=12).contains(&month)
            && day >= 1
            && day <= self.days_in_month(year, month)
            && !(self == Calendar::Standard
                && year == 1582
                && month == 10
                && (5..=14).contains(&day))
    }

    /// A day count that increases by one each day. Its origin differs between calendars.
    fn day_number(self, y: i64, m: u32, d: u32) -> i64 {
        let y = self.astronomical(y);
        match self {
            Calendar::ProlepticGregorian => gregorian_jdn(y, m, d),
            Calendar::Julian => julian_jdn(y, m, d),
            Calendar::Standard => {
                if (y, m, d) >= (1582, 10, 15) {
                    gregorian_jdn(y, m, d)
                } else {
                    julian_jdn(y, m, d)
                }
            }
            Calendar::NoLeap | Calendar::AllLeap => {
                let len = if self == Calendar::NoLeap { 365 } else { 366 };
                let before: i64 = (1..m).map(|mm| self.days_in_month(y, mm) as i64).sum();
                y * len + before + d as i64 - 1
            }
            Calendar::Day360 => y * 360 + (m as i64 - 1) * 30 + d as i64 - 1,
        }
    }

    fn date_of_day(self, n: i64) -> (i64, u32, u32) {
        let (y, m, d) = self.date_of_day_astronomical(n);
        (self.year_of_astronomical(y), m, d)
    }

    fn date_of_day_astronomical(self, n: i64) -> (i64, u32, u32) {
        match self {
            Calendar::ProlepticGregorian => gregorian_from_jdn(n),
            Calendar::Julian => julian_from_jdn(n),
            Calendar::Standard => {
                if n >= gregorian_jdn(1582, 10, 15) {
                    gregorian_from_jdn(n)
                } else {
                    julian_from_jdn(n)
                }
            }
            Calendar::NoLeap | Calendar::AllLeap => {
                let len = if self == Calendar::NoLeap { 365 } else { 366 };
                let y = n.div_euclid(len);
                let mut doy = n.rem_euclid(len) as u32;
                let mut m = 1;
                while doy >= self.days_in_month(y, m) {
                    doy -= self.days_in_month(y, m);
                    m += 1;
                }
                (y, m, doy + 1)
            }
            Calendar::Day360 => {
                let y = n.div_euclid(360);
                let doy = n.rem_euclid(360) as u32;
                (y, doy / 30 + 1, doy % 30 + 1)
            }
        }
    }
}

impl fmt::Display for Calendar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Calendar::Standard => "standard",
            Calendar::ProlepticGregorian => "proleptic_gregorian",
            Calendar::Julian => "julian",
            Calendar::NoLeap => "noleap",
            Calendar::AllLeap => "all_leap",
            Calendar::Day360 => "360_day",
        })
    }
}

fn gregorian_leap(y: i64) -> bool {
    y.rem_euclid(4) == 0 && (y.rem_euclid(100) != 0 || y.rem_euclid(400) == 0)
}

/// Julian day number of a proleptic Gregorian date (astronomical years: 0 is 1 BC).
fn gregorian_jdn(y: i64, m: u32, d: u32) -> i64 {
    let a = (14 - m as i64) / 12;
    let y = y + 4800 - a;
    let m = m as i64 + 12 * a - 3;
    d as i64 + (153 * m + 2) / 5 + 365 * y + y.div_euclid(4) - y.div_euclid(100) + y.div_euclid(400)
        - 32045
}

fn julian_jdn(y: i64, m: u32, d: u32) -> i64 {
    let a = (14 - m as i64) / 12;
    let y = y + 4800 - a;
    let m = m as i64 + 12 * a - 3;
    d as i64 + (153 * m + 2) / 5 + 365 * y + y.div_euclid(4) - 32083
}

fn gregorian_from_jdn(j: i64) -> (i64, u32, u32) {
    let a = j + 32044;
    let b = (4 * a + 3).div_euclid(146097);
    let c = a - (146097 * b).div_euclid(4);
    civil(b * 100, c)
}

fn julian_from_jdn(j: i64) -> (i64, u32, u32) {
    civil(0, j + 32082)
}

/// The shared tail of the JDN-to-date algorithms: `c` days into a 4-year cycle sequence.
fn civil(century_years: i64, c: i64) -> (i64, u32, u32) {
    let d = (4 * c + 3).div_euclid(1461);
    let e = c - (1461 * d).div_euclid(4);
    let m = (5 * e + 2).div_euclid(153);
    let day = e - (153 * m + 2).div_euclid(5) + 1;
    let month = m + 3 - 12 * m.div_euclid(10);
    let year = century_years + d - 4800 + m.div_euclid(10);
    (year, month as u32, day as u32)
}

/// A date and time in some calendar, to the microsecond.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CalDate {
    pub year: i64,
    pub month: u32,
    pub day: u32,
    /// Microseconds since midnight.
    pub micros: i64,
}

impl CalDate {
    pub fn new(year: i64, month: u32, day: u32) -> Self {
        Self {
            year,
            month,
            day,
            micros: 0,
        }
    }

    pub fn with_time(self, h: u32, m: u32, s: u32, us: u32) -> Self {
        Self {
            micros: ((h as i64 * 60 + m as i64) * 60 + s as i64) * 1_000_000 + us as i64,
            ..self
        }
    }

    pub fn hour(&self) -> u32 {
        (self.micros / 3_600_000_000) as u32
    }

    pub fn minute(&self) -> u32 {
        (self.micros / 60_000_000 % 60) as u32
    }

    pub fn second(&self) -> u32 {
        (self.micros / 1_000_000 % 60) as u32
    }

    pub fn microsecond(&self) -> u32 {
        (self.micros % 1_000_000) as u32
    }

    /// Day of the year, from 1.
    pub fn day_of_year(&self, cal: Calendar) -> u32 {
        (cal.day_number(self.year, self.month, self.day) - cal.day_number(self.year, 1, 1)) as u32
            + 1
    }
}

impl fmt::Display for CalDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.year < 0 {
            write!(f, "{:05}", self.year)?;
        } else {
            write!(f, "{:04}", self.year)?;
        }
        write!(
            f,
            "-{:02}-{:02}T{:02}:{:02}:{:02}",
            self.month,
            self.day,
            self.hour(),
            self.minute(),
            self.second()
        )?;
        if self.microsecond() != 0 {
            write!(f, ".{:06}", self.microsecond())?;
        }
        Ok(())
    }
}

/// A time value that can't be read or doesn't exist.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct TimeError(pub String);

/// The unit and epoch of CF time values, such as `days since 1850-01-01` in `noleap`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CfTime {
    pub calendar: Calendar,
    /// Microseconds per unit.
    unit_us: i64,
    epoch: CalDate,
}

impl CfTime {
    /// Parses CF `units` and `calendar` attributes.
    pub fn parse(units: &str, calendar: &str) -> Result<Self, TimeError> {
        let calendar: Calendar = calendar.parse()?;
        let (unit, since) = units
            .split_once(" since ")
            .ok_or_else(|| TimeError(format!("{units:?} isn't \"<unit> since <date>\"")))?;
        let unit_us = match unit.trim().to_ascii_lowercase().trim_end_matches('s') {
            "day" | "d" => 86_400_000_000,
            "hour" | "hr" | "h" => 3_600_000_000,
            "minute" | "min" => 60_000_000,
            "second" | "sec" => 1_000_000,
            "millisecond" | "msec" | "m" => 1000,
            "microsecond" | "usec" | "u" => 1,
            other => {
                return Err(TimeError(format!(
                    "time unit {other:?} isn't supported; months and years have no fixed length"
                )));
            }
        };
        let (epoch, offset_us) = parse_date(since.trim())?;
        if !calendar.is_valid(epoch.year, epoch.month, epoch.day) {
            return Err(TimeError(format!(
                "{since:?} doesn't exist in the {calendar} calendar"
            )));
        }
        let mut cf = CfTime {
            calendar,
            unit_us,
            epoch,
        };
        if offset_us != 0 {
            // `since 2000-01-01 00:00 +02:00` is 22:00 the day before in UTC.
            cf.epoch = cf.add_micros(epoch, -offset_us);
        }
        Ok(cf)
    }

    fn add_micros(&self, d: CalDate, us: i64) -> CalDate {
        let cal = self.calendar;
        let total = d.micros + us;
        let days = total.div_euclid(86_400_000_000);
        let micros = total.rem_euclid(86_400_000_000);
        let (year, month, day) = cal.date_of_day(cal.day_number(d.year, d.month, d.day) + days);
        CalDate {
            year,
            month,
            day,
            micros,
        }
    }

    /// The date a value means, rounded to the microsecond.
    pub fn decode(&self, value: f64) -> Option<CalDate> {
        if !value.is_finite() {
            return None;
        }
        let us = (value * self.unit_us as f64).round();
        (us.abs() < 9.0e18).then(|| self.add_micros(self.epoch, us as i64))
    }

    /// The value for a date, or `None` if the date doesn't exist in this calendar.
    pub fn encode(&self, d: CalDate) -> Option<f64> {
        let cal = self.calendar;
        if !cal.is_valid(d.year, d.month, d.day) {
            return None;
        }
        let days = cal.day_number(d.year, d.month, d.day)
            - cal.day_number(self.epoch.year, self.epoch.month, self.epoch.day);
        let us = days * 86_400_000_000 + d.micros - self.epoch.micros;
        Some(us as f64 / self.unit_us as f64)
    }

    /// The value as microseconds since the Unix epoch, for real calendars.
    pub fn to_unix_micros(&self, value: f64) -> Option<i64> {
        if !self.calendar.is_real() {
            return None;
        }
        let d = self.decode(value)?;
        let days = self.calendar.day_number(d.year, d.month, d.day) - gregorian_jdn(1970, 1, 1);
        Some(days * 86_400_000_000 + d.micros)
    }
}

/// Parses `YYYY-M-D[( |T)h:m[:s[.f]]][Z| UTC|±hh[:mm]]`, returning the date and the zone offset.
fn parse_date(s: &str) -> Result<(CalDate, i64), TimeError> {
    let bad = || TimeError(format!("can't read the date {s:?}"));
    let s = s.trim();
    let (s, mut offset) = if let Some(rest) = s.strip_suffix('Z').or_else(|| s.strip_suffix(" UTC"))
    {
        (rest.trim(), 0)
    } else {
        (s, 0)
    };
    let (date, time) = match s.split_once(['T', ' ']) {
        Some((d, t)) => (d, t.trim()),
        None => (s, ""),
    };
    let neg_year = date.starts_with('-');
    let mut parts = date.trim_start_matches('-').splitn(3, '-');
    let num = |p: Option<&str>| p.and_then(|x| x.parse::<i64>().ok()).ok_or_else(bad);
    let year = num(parts.next())? * if neg_year { -1 } else { 1 };
    let month = num(parts.next())? as u32;
    let day = num(parts.next())? as u32;
    let mut time = time;
    if let Some(i) = time.rfind(['+', '-']).filter(|&i| i > 0) {
        let (t, z) = time.split_at(i);
        let sign = if z.starts_with('-') { -1 } else { 1 };
        let z = &z[1..];
        let (zh, zm) = z.split_once(':').unwrap_or((z, "0"));
        let zh: i64 = zh.parse().map_err(|_| bad())?;
        let zm: i64 = zm.parse().map_err(|_| bad())?;
        offset = sign * (zh * 60 + zm) * 60_000_000;
        time = t.trim();
    }
    let mut micros = 0;
    if !time.is_empty() {
        let mut hms = time.splitn(3, ':');
        let h: i64 = hms.next().and_then(|x| x.parse().ok()).ok_or_else(bad)?;
        let m: i64 = hms.next().unwrap_or("0").parse().map_err(|_| bad())?;
        let sec: f64 = hms.next().unwrap_or("0").parse().map_err(|_| bad())?;
        micros = (h * 60 + m) * 60_000_000 + (sec * 1e6).round() as i64;
    }
    Ok((
        CalDate {
            year,
            month,
            day,
            micros,
        },
        offset,
    ))
}

/// How [`convert_calendar`] maps dates to or from the 360-day calendar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignOn {
    /// Keep the date; dates that don't exist in the target are dropped.
    Date,
    /// Keep the position in the year: day-of-year is scaled between year lengths.
    Year,
}

/// Converts dates to another calendar, as xarray's `convert_calendar` does. `None` marks a
/// dropped date. Conversions to or from `360_day` need `align_on`:
///
/// - [`AlignOn::Date`] keeps each date and drops those that don't exist in the target.
/// - [`AlignOn::Year`] keeps each date's position in its year: the new day of year is
///   `round(doy × new_length / old_length)` (ties to even), and when several dates land on the
///   same new date only the first is kept.
///
/// Other conversions keep the date and drop invalid ones, ignoring `align_on`.
pub fn convert_calendar(
    dates: &[CalDate],
    from: Calendar,
    to: Calendar,
    align_on: Option<AlignOn>,
) -> Result<Vec<Option<CalDate>>, TimeError> {
    let involves_360 = (from == Calendar::Day360) != (to == Calendar::Day360);
    if involves_360 && align_on.is_none() {
        return Err(TimeError(
            "converting to or from 360_day needs align_on: date or year".into(),
        ));
    }
    if !(involves_360 && align_on == Some(AlignOn::Year)) {
        return Ok(dates
            .iter()
            .map(|d| to.is_valid(d.year, d.month, d.day).then_some(*d))
            .collect());
    }
    let mut seen = std::collections::HashSet::new();
    Ok(dates
        .iter()
        .map(|d| {
            let old_len = from.days_in_year(d.year) as f64;
            let new_len = to.days_in_year(d.year) as f64;
            let doy = d.day_of_year(from) as f64;
            let new_doy = (new_len * doy / old_len).round_ties_even() as i64;
            let n = to.day_number(d.year, 1, 1) + new_doy - 1;
            let (year, month, day) = to.date_of_day(n);
            let out = CalDate {
                year,
                month,
                day,
                micros: d.micros,
            };
            seen.insert(out).then_some(out)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_numbers_round_trip() {
        for cal in [
            Calendar::Standard,
            Calendar::ProlepticGregorian,
            Calendar::Julian,
            Calendar::NoLeap,
            Calendar::AllLeap,
            Calendar::Day360,
        ] {
            for n in (-800_000..2_000_000).step_by(997) {
                let (y, m, d) = cal.date_of_day(cal.day_number(0, 1, 1) + n);
                assert!(cal.is_valid(y, m, d), "{cal}: {y}-{m}-{d}");
                assert_eq!(
                    cal.day_number(y, m, d),
                    cal.day_number(0, 1, 1) + n,
                    "{cal}"
                );
            }
        }
    }

    #[test]
    fn standard_calendar_skips_ten_days() {
        let cf = CfTime::parse("days since 1582-10-04", "standard").unwrap();
        assert_eq!(cf.decode(1.0).unwrap(), CalDate::new(1582, 10, 15));
        assert!(!Calendar::Standard.is_valid(1582, 10, 10));
    }

    #[test]
    fn parses_epochs() {
        let cf = CfTime::parse("hours since 1970-01-01T06:00:00+06:00", "standard").unwrap();
        assert_eq!(cf.to_unix_micros(0.0), Some(0));
        let cf = CfTime::parse("seconds since 1970-1-1 0:0:0 UTC", "gregorian").unwrap();
        assert_eq!(cf.to_unix_micros(1.5), Some(1_500_000));
        assert!(CfTime::parse("months since 2000-01-01", "360_day").is_err());
        assert!(CfTime::parse("days since 2001-02-29", "noleap").is_err());
        assert_eq!(
            CfTime::parse("days since 2000-01-01", "noleap")
                .unwrap()
                .to_unix_micros(0.0),
            None
        );
    }
}
