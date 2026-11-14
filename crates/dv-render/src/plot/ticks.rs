//! Tick values from Wilkinson's extended algorithm (Talbot, Lin & Hanrahan, "An Extension of
//! Wilkinson's Algorithm for Positioning Tick Labels on Axes", InfoVis 2010).
//!
//! A search over step sizes `j · q · 10^z` (q from a preference-ordered list of nice numbers, j a
//! skip factor) and tick counts `k`, scoring simplicity, coverage, density and legibility, with
//! upper bounds that prune the search. Matches the authors' R implementation (`labeling::extended`,
//! the source of ggplot2's breaks), checked against it in tests.

/// Nice step multiples, most preferred first.
pub const Q: [f64; 6] = [1.0, 5.0, 2.0, 2.5, 4.0, 3.0];
/// Weights for simplicity, coverage, density and legibility.
const W: [f64; 4] = [0.25, 0.2, 0.5, 0.05];

/// About `m` tick values covering `dmin..=dmax`. With `only_loose`, the first and last ticks lie
/// outside or on the data range.
pub fn extended(dmin: f64, dmax: f64, m: usize, only_loose: bool) -> Vec<f64> {
    let eps = f64::EPSILON * 100.0;
    let (dmin, dmax) = if dmin > dmax {
        (dmax, dmin)
    } else {
        (dmin, dmax)
    };
    let m_f = m as f64;
    if dmax - dmin < eps {
        // R: seq(dmin, dmax, length.out = m).
        return (0..m)
            .map(|i| dmin + (dmax - dmin) * i as f64 / (m_f - 1.0).max(1.0))
            .collect();
    }
    let n = Q.len() as f64;
    let mut best: Option<(f64, f64, f64)> = None;
    let mut best_score = -2.0;

    let mut j = 1.0_f64;
    'outer: loop {
        for (qi, &q) in Q.iter().enumerate() {
            let sm = 1.0 - qi as f64 / (n - 1.0) - j + 1.0;
            if W[0] * sm + W[1] + W[2] + W[3] < best_score {
                break 'outer;
            }
            let mut k = 2.0_f64;
            loop {
                let dm = if k >= m_f {
                    2.0 - (k - 1.0) / (m_f - 1.0)
                } else {
                    1.0
                };
                if W[0] * sm + W[1] + W[2] * dm + W[3] < best_score {
                    break;
                }
                let delta = (dmax - dmin) / (k + 1.0) / j / q;
                let mut z = delta.log10().ceil();
                loop {
                    let step = j * q * 10f64.powf(z);
                    let cm = coverage_max(dmin, dmax, step * (k - 1.0));
                    if W[0] * sm + W[1] * cm + W[2] * dm + W[3] < best_score {
                        break;
                    }
                    let min_start = (dmax / step).floor() * j - (k - 1.0) * j;
                    let max_start = (dmin / step).ceil() * j;
                    if min_start <= max_start {
                        let mut start = min_start;
                        while start <= max_start {
                            let lmin = start * (step / j);
                            let lmax = lmin + step * (k - 1.0);
                            let s = simplicity(qi, n, j, lmin, lmax, step, eps);
                            let c = coverage(dmin, dmax, lmin, lmax);
                            let g = density(k, m_f, dmin, dmax, lmin, lmax);
                            let score = W[0] * s + W[1] * c + W[2] * g + W[3];
                            if score > best_score && (!only_loose || (lmin <= dmin && lmax >= dmax))
                            {
                                best_score = score;
                                best = Some((lmin, lmax, step));
                            }
                            start += 1.0;
                        }
                    }
                    z += 1.0;
                }
                k += 1.0;
            }
        }
        j += 1.0;
    }
    let Some((lmin, lmax, step)) = best else {
        return Vec::new();
    };
    // R: seq(lmin, lmax, by = step), which allows for rounding with a 1e-10 fuzz and clamps to lmax.
    let count = ((lmax - lmin) / step + 1e-10).floor() as usize;
    (0..=count)
        .map(|i| (lmin + i as f64 * step).min(lmax))
        .collect()
}

fn simplicity(qi: usize, n: f64, j: f64, lmin: f64, lmax: f64, step: f64, eps: f64) -> f64 {
    let r = r_mod(lmin, step);
    let v = if (r < eps || step - r < eps) && lmin <= 0.0 && lmax >= 0.0 {
        1.0
    } else {
        0.0
    };
    1.0 - qi as f64 / (n - 1.0) - j + v
}

/// R's `%%` on doubles (`myfmod` in R's arithmetic.c), whose rounding differs from `rem_euclid`.
fn r_mod(x1: f64, x2: f64) -> f64 {
    if x2.abs() * f64::EPSILON > 1.0 && x1.is_finite() && x1.abs() <= x2.abs() {
        return if x1.abs() == x2.abs() {
            0.0
        } else if (x1 < 0.0 && x2 > 0.0) || (x2 < 0.0 && x1 > 0.0) {
            x1 + x2
        } else {
            x1
        };
    }
    let q = x1 / x2;
    let tmp = x1 - q.floor() * x2;
    tmp - (tmp / x2).floor() * x2
}

fn coverage(dmin: f64, dmax: f64, lmin: f64, lmax: f64) -> f64 {
    let range = dmax - dmin;
    1.0 - 0.5 * ((dmax - lmax).powi(2) + (dmin - lmin).powi(2)) / (0.1 * range).powi(2)
}

fn coverage_max(dmin: f64, dmax: f64, span: f64) -> f64 {
    let range = dmax - dmin;
    if span > range {
        let half = (span - range) / 2.0;
        1.0 - 0.5 * (half.powi(2) + half.powi(2)) / (0.1 * range).powi(2)
    } else {
        1.0
    }
}

fn density(k: f64, m: f64, dmin: f64, dmax: f64, lmin: f64, lmax: f64) -> f64 {
    let r = (k - 1.0) / (lmax - lmin);
    let rt = (m - 1.0) / (lmax.max(dmax) - dmin.min(lmin));
    2.0 - (r / rt).max(rt / r)
}
