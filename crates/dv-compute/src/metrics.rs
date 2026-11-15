//! Verification metrics: how well a forecast matches observations.
//!
//! These are the methods behind comparison views, kept in the open engine so results can be
//! cited and checked. Missing values are NaN. Pairwise scores skip pairs where either side is
//! missing; [`crps_ensemble`] skips observations where the observation or any member is missing.
//! A score with no valid pairs is NaN. Definitions follow xskillscore, which the tests check
//! against.

/// Pairs where both values are present.
fn pairs<'a>(fc: &'a [f64], ob: &'a [f64]) -> impl Iterator<Item = (f64, f64)> + 'a {
    assert_eq!(
        fc.len(),
        ob.len(),
        "forecast and observations differ in length"
    );
    fc.iter()
        .zip(ob)
        .map(|(f, o)| (*f, *o))
        .filter(|(f, o)| !f.is_nan() && !o.is_nan())
}

fn mean(it: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = it.fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

/// Mean error, forecast minus observation.
pub fn bias(fc: &[f64], ob: &[f64]) -> f64 {
    mean(pairs(fc, ob).map(|(f, o)| f - o))
}

/// Mean absolute error.
pub fn mae(fc: &[f64], ob: &[f64]) -> f64 {
    mean(pairs(fc, ob).map(|(f, o)| (f - o).abs()))
}

/// Mean squared error.
pub fn mse(fc: &[f64], ob: &[f64]) -> f64 {
    mean(pairs(fc, ob).map(|(f, o)| (f - o).powi(2)))
}

/// Root mean squared error.
pub fn rmse(fc: &[f64], ob: &[f64]) -> f64 {
    mse(fc, ob).sqrt()
}

/// Pearson's correlation coefficient.
pub fn pearson_r(fc: &[f64], ob: &[f64]) -> f64 {
    let (f, o): (Vec<f64>, Vec<f64>) = pairs(fc, ob).unzip();
    pearson(&f, &o)
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    let mx = mean(x.iter().copied());
    let my = mean(y.iter().copied());
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        let (dx, dy) = (a - mx, b - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    sxy / (sxx * syy).sqrt()
}

/// Spearman's rank correlation: Pearson's on ranks, with tied values sharing their mean rank.
pub fn spearman_r(fc: &[f64], ob: &[f64]) -> f64 {
    let (f, o): (Vec<f64>, Vec<f64>) = pairs(fc, ob).unzip();
    pearson(&ranks(&f), &ranks(&o))
}

/// Ranks from 1, ties averaged.
fn ranks(x: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut out = vec![0.0; x.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        let rank = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            out[k] = rank;
        }
        i = j + 1;
    }
    out
}

/// The continuous ranked probability score of an ensemble, averaged over observations:
/// `mean |X - y| - ½ mean |X - X'|` for members X, X' and observation y.
pub fn crps_ensemble(members: &[Vec<f64>], ob: &[f64]) -> f64 {
    assert_eq!(
        members.len(),
        ob.len(),
        "members and observations differ in length"
    );
    mean(
        members
            .iter()
            .zip(ob)
            .filter(|(m, o)| !o.is_nan() && !m.is_empty() && m.iter().all(|x| !x.is_nan()))
            .map(|(m, &o)| {
                let n = m.len() as f64;
                let skill = m.iter().map(|x| (x - o).abs()).sum::<f64>() / n;
                // Sum of |xi - xj| over all ordered pairs, from sorted members in O(n log n).
                let mut s = m.clone();
                s.sort_by(f64::total_cmp);
                let spread: f64 = s
                    .iter()
                    .enumerate()
                    .map(|(i, x)| x * (2.0 * i as f64 - n + 1.0))
                    .sum::<f64>()
                    * 2.0;
                skill - 0.5 * spread / (n * n)
            }),
    )
}

/// The Brier score of event probabilities, where the event is an observation at or above
/// `threshold`.
pub fn brier_score(prob: &[f64], ob: &[f64], threshold: f64) -> f64 {
    mean(pairs(prob, ob).map(|(p, o)| {
        let event = if o >= threshold { 1.0 } else { 0.0 };
        (p - event).powi(2)
    }))
}

/// A 2×2 contingency table for an event: a value at or above a threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Contingency {
    pub hits: u64,
    pub misses: u64,
    pub false_alarms: u64,
    pub correct_negatives: u64,
}

impl Contingency {
    pub fn new(fc: &[f64], ob: &[f64], threshold: f64) -> Self {
        let mut t = Contingency::default();
        for (f, o) in pairs(fc, ob) {
            match (f >= threshold, o >= threshold) {
                (true, true) => t.hits += 1,
                (false, true) => t.misses += 1,
                (true, false) => t.false_alarms += 1,
                (false, false) => t.correct_negatives += 1,
            }
        }
        t
    }

    fn f(&self) -> (f64, f64, f64, f64) {
        (
            self.hits as f64,
            self.misses as f64,
            self.false_alarms as f64,
            self.correct_negatives as f64,
        )
    }

    pub fn total(&self) -> u64 {
        self.hits + self.misses + self.false_alarms + self.correct_negatives
    }

    /// Probability of detection: hits / observed events.
    pub fn hit_rate(&self) -> f64 {
        let (h, m, _, _) = self.f();
        h / (h + m)
    }

    /// False alarms / forecast events.
    pub fn false_alarm_ratio(&self) -> f64 {
        let (h, _, fa, _) = self.f();
        fa / (h + fa)
    }

    /// False alarms / observed non-events (probability of false detection).
    pub fn false_alarm_rate(&self) -> f64 {
        let (_, _, fa, cn) = self.f();
        fa / (fa + cn)
    }

    /// Critical success index: hits / (hits + misses + false alarms).
    pub fn threat_score(&self) -> f64 {
        let (h, m, fa, _) = self.f();
        h / (h + m + fa)
    }

    /// Gilbert skill score: the threat score corrected for hits expected by chance.
    pub fn equit_threat_score(&self) -> f64 {
        let (h, m, fa, _) = self.f();
        let random = (h + m) * (h + fa) / self.total() as f64;
        (h - random) / (h + m + fa - random)
    }

    /// Frequency bias: forecast events / observed events.
    pub fn bias_score(&self) -> f64 {
        let (h, m, fa, _) = self.f();
        (h + fa) / (h + m)
    }

    /// Peirce (Hanssen-Kuipers) skill score: hit rate minus false alarm rate.
    pub fn peirce_score(&self) -> f64 {
        self.hit_rate() - self.false_alarm_rate()
    }

    /// Heidke skill score: accuracy corrected for agreement expected by chance.
    pub fn heidke_score(&self) -> f64 {
        let (h, m, fa, cn) = self.f();
        2.0 * (h * cn - m * fa) / ((h + m) * (m + cn) + (h + fa) * (fa + cn))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_average_ties() {
        assert_eq!(ranks(&[3.0, 1.0, 3.0, 2.0]), [3.5, 1.0, 3.5, 2.0]);
    }

    #[test]
    fn crps_of_a_one_member_ensemble_is_its_absolute_error() {
        let c = crps_ensemble(&[vec![2.0], vec![-1.0]], &[1.0, 1.0]);
        assert_eq!(c, 1.5);
    }

    #[test]
    fn empty_input_is_nan() {
        assert!(rmse(&[f64::NAN], &[1.0]).is_nan());
    }
}
