//! Colour-vision deficiency: simulation, and a check that flags categorical colors which become
//! hard to tell apart.
//!
//! Simulation uses Machado, Oliveira & Fernandes (2009): a 3 × 3 matrix on linear sRGB per
//! deficiency and severity, interpolated linearly between the authors' tenths, exactly as
//! colorspacious's `sRGB1+CVD` space does. The check simulates every color, clamps the result to
//! the sRGB gamut and flags each pair whose CIEDE2000 difference falls below a threshold.
//! `fixtures/reference/cvd.json` holds colorspacious and colour-science values for both steps.

mod machado;

use crate::color::{delta_e2000, linear_to_lab, linear_to_srgb, srgb, srgb_to_linear};
use crate::vector::Rgba;

/// Pairs closer than this (CIEDE2000) are flagged by [`check`]. Around 10 is where small
/// categorical marks, such as lines and points, stop being told apart at a glance; the
/// Okabe–Ito palette clears it under every simulated deficiency.
pub const MIN_DELTA_E: f64 = 10.0;

/// A deficiency of one cone type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Deficiency {
    /// Long-wavelength (red) cones.
    Protan,
    /// Medium-wavelength (green) cones; the most common.
    Deutan,
    /// Short-wavelength (blue) cones; rare.
    Tritan,
}

impl Deficiency {
    pub const ALL: [Self; 3] = [Self::Protan, Self::Deutan, Self::Tritan];

    /// Machado's matrix at `severity` in 0..=1, where 1 is dichromacy (protanopia and so on).
    pub fn matrix(self, severity: f64) -> [[f64; 3]; 3] {
        let table = match self {
            Self::Protan => &machado::PROTAN,
            Self::Deutan => &machado::DEUTAN,
            Self::Tritan => &machado::TRITAN,
        };
        let s = severity.clamp(0.0, 1.0) * 10.0;
        let lo = (s.floor() as usize).min(9);
        let f = s - lo as f64;
        std::array::from_fn(|i| {
            std::array::from_fn(|j| (1.0 - f) * table[lo][i][j] + f * table[lo + 1][i][j])
        })
    }

    /// How `color` looks with this deficiency at `severity`, in linear sRGB, not clamped.
    pub fn simulate_linear(self, color: Rgba, severity: f64) -> [f64; 3] {
        let m = self.matrix(severity);
        let rgb = srgb(color).map(srgb_to_linear);
        std::array::from_fn(|i| m[i][0] * rgb[0] + m[i][1] * rgb[1] + m[i][2] * rgb[2])
    }

    /// How `color` looks with this deficiency at `severity`, as sRGB components, not clamped (some
    /// simulated colors fall just outside the gamut).
    pub fn simulate(self, color: Rgba, severity: f64) -> [f64; 3] {
        self.simulate_linear(color, severity).map(linear_to_srgb)
    }
}

/// Two palette entries that are hard to tell apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Confusion {
    /// Indices into the palette, `a < b`.
    pub a: usize,
    pub b: usize,
    /// `None` for typical color vision.
    pub deficiency: Option<Deficiency>,
    /// CIEDE2000 between the two as seen.
    pub delta_e: f64,
}

/// The CIEDE2000 difference between `a` and `b` as seen with `deficiency` (or typical vision) at
/// `severity`. Simulated colors are clamped to the sRGB gamut first.
pub fn delta_e_seen(a: Rgba, b: Rgba, deficiency: Option<Deficiency>, severity: f64) -> f64 {
    let seen = |c: Rgba| {
        let rgb = match deficiency {
            Some(d) => d.simulate_linear(c, severity),
            None => srgb(c).map(srgb_to_linear),
        };
        linear_to_lab(rgb.map(|v| v.clamp(0.0, 1.0)))
    };
    delta_e2000(seen(a), seen(b))
}

/// Every pair of `palette` closer than `min_delta_e` with typical vision or any deficiency at
/// `severity`, ordered by pair then vision. Alpha is ignored.
pub fn check(palette: &[Rgba], severity: f64, min_delta_e: f64) -> Vec<Confusion> {
    let visions = [None]
        .into_iter()
        .chain(Deficiency::ALL.into_iter().map(Some));
    let mut out = Vec::new();
    for a in 0..palette.len() {
        for b in a + 1..palette.len() {
            for deficiency in visions.clone() {
                let delta_e = delta_e_seen(palette[a], palette[b], deficiency, severity);
                if delta_e < min_delta_e {
                    out.push(Confusion {
                        a,
                        b,
                        deficiency,
                        delta_e,
                    });
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_zero_is_identity() {
        for d in Deficiency::ALL {
            let m = d.matrix(0.0);
            for (i, row) in m.iter().enumerate() {
                for (j, v) in row.iter().enumerate() {
                    assert!((v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn red_and_green_collide_for_deutans() {
        let (red, green) = (Rgba::rgb(220, 50, 47), Rgba::rgb(110, 140, 30));
        assert!(delta_e_seen(red, green, None, 1.0) > 30.0);
        let flagged = check(&[red, green], 1.0, MIN_DELTA_E);
        assert!(
            flagged
                .iter()
                .any(|c| c.deficiency == Some(Deficiency::Deutan))
        );
    }
}
