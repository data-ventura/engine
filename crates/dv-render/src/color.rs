//! Colour science for checking palettes and comparing images: sRGB transfer functions, CIELAB and
//! the CIEDE2000 colour difference.
//!
//! sRGB follows IEC 61966-2-1: the piecewise transfer function and its 4-digit RGB-to-XYZ matrix,
//! with the D65 white point from its chromaticity (0.3127, 0.3290). CIEDE2000 follows Sharma, Wu &
//! Dalal (2005), "The CIEDE2000 Color-Difference Formula: Implementation Notes", with kL = kC = kH
//! = 1. Both agree with colour-science (`fixtures/reference/cvd.json`).

use crate::vector::Rgba;

/// IEC 61966-2-1's linear sRGB to CIE XYZ matrix.
const RGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
];
/// D65's XYZ, from its chromaticity with Y = 1.
const WHITE: [f64; 3] = [0.3127 / 0.3290, 1.0, (1.0 - 0.3127 - 0.3290) / 0.3290];

/// An sRGB component in 0..=1 to linear light.
pub fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// A linear-light component to sRGB. Not clamped.
pub fn linear_to_srgb(c: f64) -> f64 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// A color's sRGB components in 0..=1, ignoring alpha.
pub fn srgb(c: Rgba) -> [f64; 3] {
    std::array::from_fn(|i| f64::from(c.0[i]) / 255.0)
}

/// Linear sRGB (in gamut, 0..=1) to CIELAB, D65.
pub fn linear_to_lab(rgb: [f64; 3]) -> [f64; 3] {
    let xyz: [f64; 3] = std::array::from_fn(|i| {
        RGB_TO_XYZ[i][0] * rgb[0] + RGB_TO_XYZ[i][1] * rgb[1] + RGB_TO_XYZ[i][2] * rgb[2]
    });
    let f = |t: f64| {
        let d = 6.0 / 29.0;
        if t > d * d * d {
            t.cbrt()
        } else {
            t / (3.0 * d * d) + 4.0 / 29.0
        }
    };
    let [fx, fy, fz] = std::array::from_fn(|i| f(xyz[i] / WHITE[i]));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// An sRGB color to CIELAB, D65, ignoring alpha.
pub fn lab(c: Rgba) -> [f64; 3] {
    linear_to_lab(srgb(c).map(srgb_to_linear))
}

/// The CIEDE2000 difference between two CIELAB colors. About 1 is just noticeable side by side;
/// categorical marks in a chart want 10 or more.
pub fn delta_e2000(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    use std::f64::consts::PI;
    let [l1, a1, b1] = lab1;
    let [l2, a2, b2] = lab2;
    let c_bar = (a1.hypot(b1) + a2.hypot(b2)) / 2.0;
    let c7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c7 / (c7 + 25f64.powi(7))).sqrt());
    let (a1p, a2p) = ((1.0 + g) * a1, (1.0 + g) * a2);
    let (c1p, c2p) = (a1p.hypot(b1), a2p.hypot(b2));
    // Hue angles in degrees, 0..360; zero for achromatic colors.
    let hue = |b: f64, a: f64| {
        if a == 0.0 && b == 0.0 {
            0.0
        } else {
            b.atan2(a).to_degrees().rem_euclid(360.0)
        }
    };
    let (h1p, h2p) = (hue(b1, a1p), hue(b2, a2p));

    let dl = l2 - l1;
    let dc = c2p - c1p;
    let dh = if c1p * c2p == 0.0 {
        0.0
    } else if (h2p - h1p).abs() <= 180.0 {
        h2p - h1p
    } else if h2p - h1p > 180.0 {
        h2p - h1p - 360.0
    } else {
        h2p - h1p + 360.0
    };
    let dhh = 2.0 * (c1p * c2p).sqrt() * (dh.to_radians() / 2.0).sin();

    let l_bar = (l1 + l2) / 2.0;
    let c_bar_p = (c1p + c2p) / 2.0;
    let h_bar = if c1p * c2p == 0.0 {
        h1p + h2p
    } else if (h1p - h2p).abs() <= 180.0 {
        (h1p + h2p) / 2.0
    } else if h1p + h2p < 360.0 {
        (h1p + h2p + 360.0) / 2.0
    } else {
        (h1p + h2p - 360.0) / 2.0
    };
    let t = 1.0 - 0.17 * (h_bar - 30.0).to_radians().cos()
        + 0.24 * (2.0 * h_bar).to_radians().cos()
        + 0.32 * (3.0 * h_bar + 6.0).to_radians().cos()
        - 0.20 * (4.0 * h_bar - 63.0).to_radians().cos();
    let d_theta = 30.0 * (-((h_bar - 275.0) / 25.0).powi(2)).exp();
    let c7 = c_bar_p.powi(7);
    let rc = 2.0 * (c7 / (c7 + 25f64.powi(7))).sqrt();
    let l50 = (l_bar - 50.0).powi(2);
    let sl = 1.0 + 0.015 * l50 / (20.0 + l50).sqrt();
    let sc = 1.0 + 0.045 * c_bar_p;
    let sh = 1.0 + 0.015 * c_bar_p * t;
    let rt = -(2.0 * d_theta * PI / 180.0).sin() * rc;
    let (x, y, z) = (dl / sl, dc / sc, dhh / sh);
    (x * x + y * y + z * z + rt * y * z).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_round_trips() {
        for i in 0..=255 {
            let c = f64::from(i) / 255.0;
            assert!((linear_to_srgb(srgb_to_linear(c)) - c).abs() < 1e-12);
        }
    }

    /// Nearly neutral: the standard's rounded matrix puts sRGB white a hair off D65's chromaticity.
    #[test]
    fn white_is_l100() {
        let l = lab(Rgba::rgb(255, 255, 255));
        assert!(
            (l[0] - 100.0).abs() < 1e-3 && l[1].abs() < 0.05 && l[2].abs() < 0.05,
            "{l:?}"
        );
    }

    /// Pairs 1, 7 and 17 of Sharma, Wu & Dalal's test data.
    #[test]
    fn sharma_pairs() {
        let cases = [
            ([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485], 2.0425),
            ([50.0, 0.0, 0.0], [50.0, -1.0, 2.0], 2.3669),
            ([50.0, 2.5, 0.0], [73.0, 25.0, -18.0], 27.1492),
        ];
        for (a, b, want) in cases {
            assert!((delta_e2000(a, b) - want).abs() < 1e-4, "{a:?} {b:?}");
        }
    }
}
