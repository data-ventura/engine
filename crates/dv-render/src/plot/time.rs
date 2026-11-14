//! Date axes: ticks on calendar boundaries in UTC, at the finest step that fits (hours, then days,
//! then months, then years), labeled with context shown once ("Jan 2019", then "Feb", "Mar", …).

/// One tick on a date axis.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeTick {
    /// Seconds since 1970-01-01T00:00Z.
    pub t: f64,
    pub label: String,
}

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

#[derive(Debug, Clone, Copy)]
enum Step {
    Hours(i64),
    Days(i64),
    Months(i64),
    Years(i64),
}

const STEPS: [Step; 15] = [
    Step::Hours(1),
    Step::Hours(3),
    Step::Hours(6),
    Step::Hours(12),
    Step::Days(1),
    Step::Days(2),
    Step::Days(7),
    Step::Months(1),
    Step::Months(2),
    Step::Months(3),
    Step::Months(6),
    Step::Years(1),
    Step::Years(2),
    Step::Years(5),
    Step::Years(10),
];

/// Ticks between `start` and `end` (seconds since the epoch, inclusive), at the finest calendar step
/// that gives at most `max` ticks.
pub fn time_ticks(start: f64, end: f64, max: usize) -> Vec<TimeTick> {
    for step in STEPS {
        let ticks = ticks_at(start, end, step);
        if ticks.len() <= max.max(2) {
            return label(&ticks, step);
        }
    }
    let mut years = 20;
    loop {
        let ticks = ticks_at(start, end, Step::Years(years));
        if ticks.len() <= max.max(2) {
            return label(&ticks, Step::Years(years));
        }
        years *= 2;
    }
}

fn ticks_at(start: f64, end: f64, step: Step) -> Vec<f64> {
    let day = 86_400.0;
    let mut out = Vec::new();
    let fixed = |s: f64, offset: f64, out: &mut Vec<f64>| {
        let mut t = ((start - offset) / s).ceil() * s + offset;
        while t <= end {
            out.push(t);
            t += s;
        }
    };
    match step {
        Step::Hours(h) => fixed(3600.0 * h as f64, 0.0, &mut out),
        // Weeks start on Mondays (1970-01-05 was one); other day steps count from the epoch, so
        // they land on the same days at every zoom.
        Step::Days(7) => fixed(7.0 * day, 4.0 * day, &mut out),
        Step::Days(d) => fixed(d as f64 * day, 0.0, &mut out),
        Step::Months(_) | Step::Years(_) => {
            let months = match step {
                Step::Years(y) => 12 * y,
                Step::Months(m) => m,
                _ => unreachable!(),
            };
            // The first month boundary at or after start, aligned to the step.
            let (y, m, _) = civil((start / day).floor() as i64);
            let mut idx = y * 12 + (m - 1);
            if (days(y, m, 1) as f64) * day < start {
                idx += 1;
            }
            idx = idx.div_euclid(months) * months
                + if idx.rem_euclid(months) == 0 {
                    0
                } else {
                    months
                };
            loop {
                let t = days(idx.div_euclid(12), idx.rem_euclid(12) + 1, 1) as f64 * day;
                if t > end {
                    break;
                }
                out.push(t);
                idx += months;
            }
        }
    }
    out
}

fn label(ticks: &[f64], step: Step) -> Vec<TimeTick> {
    let mut last_context: Option<(i64, i64)> = None;
    ticks
        .iter()
        .map(|&t| {
            let secs = t as i64;
            let (y, m, d) = civil(secs.div_euclid(86_400));
            let hour = secs.rem_euclid(86_400) / 3600;
            let label = match step {
                Step::Years(_) => y.to_string(),
                Step::Months(_) => {
                    let name = MONTHS[(m - 1) as usize];
                    if last_context != Some((y, 0)) {
                        last_context = Some((y, 0));
                        format!("{name} {y}")
                    } else {
                        name.to_string()
                    }
                }
                Step::Days(_) => {
                    let name = MONTHS[(m - 1) as usize];
                    if last_context != Some((y, m)) {
                        last_context = Some((y, m));
                        format!("{name} {d}")
                    } else {
                        d.to_string()
                    }
                }
                Step::Hours(_) => {
                    if last_context != Some((y, m * 100 + d)) {
                        last_context = Some((y, m * 100 + d));
                        format!("{} {d} {hour:02}:00", MONTHS[(m - 1) as usize])
                    } else {
                        format!("{hour:02}:00")
                    }
                }
            };
            TimeTick { t, label }
        })
        .collect()
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's `days_from_civil`).
pub fn days(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The date of a day number since 1970-01-01 (Howard Hinnant's `civil_from_days`).
pub fn civil(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calendar_matches_python_datetime() {
        // Reference: (date(y, m, d) - date(1970, 1, 1)).days in Python.
        for (y, m, d, n) in [
            (1970, 1, 1, 0),
            (2019, 1, 1, 17_897),
            (2000, 2, 29, 11_016),
            (1900, 3, 1, -25_508),
            (2024, 12, 31, 20_088),
            (1979, 1, 1, 3_287),
        ] {
            assert_eq!(days(y, m, d), n, "{y}-{m}-{d}");
            assert_eq!(civil(n), (y, m, d));
        }
    }

    #[test]
    fn a_year_gets_months_with_the_year_once() {
        let start = days(2019, 1, 1) as f64 * 86_400.0;
        let end = days(2019, 12, 31) as f64 * 86_400.0 + 23.0 * 3600.0;
        let labels: Vec<String> = time_ticks(start, end, 12)
            .into_iter()
            .map(|t| t.label)
            .collect();
        assert_eq!(
            labels,
            [
                "Jan 2019", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                "Dec"
            ]
        );
        let labels: Vec<String> = time_ticks(start, end, 6)
            .into_iter()
            .map(|t| t.label)
            .collect();
        assert_eq!(labels, ["Jan 2019", "Mar", "May", "Jul", "Sep", "Nov"]);
        // Mid-month starts begin at the next boundary.
        let mid = days(2019, 1, 15) as f64 * 86_400.0;
        assert_eq!(time_ticks(mid, end, 12)[0].label, "Feb 2019");
    }

    #[test]
    fn shorter_spans_get_days_and_hours() {
        let start = days(2019, 7, 30) as f64 * 86_400.0;
        let labels: Vec<String> = time_ticks(start, start + 5.0 * 86_400.0, 8)
            .into_iter()
            .map(|t| t.label)
            .collect();
        assert_eq!(labels, ["Jul 30", "31", "Aug 1", "2", "3", "4"]);
        let labels: Vec<String> = time_ticks(start, start + 86_400.0, 5)
            .into_iter()
            .map(|t| t.label)
            .collect();
        assert_eq!(
            labels,
            ["Jul 30 00:00", "06:00", "12:00", "18:00", "Jul 31 00:00"]
        );
    }
}
