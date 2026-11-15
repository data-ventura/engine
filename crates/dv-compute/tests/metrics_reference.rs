//! Checks verification metrics against xskillscore on synthetic data with missing values. The
//! data and reference come from `scripts/reference/verification.py`.

use std::path::PathBuf;

use dv_compute::metrics::*;
use serde_json::Value as Json;

fn values(v: &Json) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect()
}

#[test]
fn metrics_match_xskillscore() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/reference/verification.json");
    let r: Json = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let (ob, fc, prob) = (values(&r["obs"]), values(&r["fc"]), values(&r["prob"]));
    let ens: Vec<Vec<f64>> = r["ens"].as_array().unwrap().iter().map(values).collect();
    let thr = r["threshold"].as_f64().unwrap();
    let s = &r["scores"];
    let table = Contingency::new(&fc, &ob, thr);

    let check = |name: &str, got: f64| {
        let want = s[name].as_f64().unwrap();
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1.0),
            "{name}: {got} vs xskillscore {want}"
        );
    };
    check("bias", bias(&fc, &ob));
    check("mae", mae(&fc, &ob));
    check("mse", mse(&fc, &ob));
    check("rmse", rmse(&fc, &ob));
    check("pearson_r", pearson_r(&fc, &ob));
    check("spearman_r", spearman_r(&fc, &ob));
    check("crps_ensemble", crps_ensemble(&ens, &ob));
    check("brier_score", brier_score(&prob, &ob, thr));
    check("hits", table.hits as f64);
    check("misses", table.misses as f64);
    check("false_alarms", table.false_alarms as f64);
    check("correct_negatives", table.correct_negatives as f64);
    check("hit_rate", table.hit_rate());
    check("false_alarm_ratio", table.false_alarm_ratio());
    check("threat_score", table.threat_score());
    check("equit_threat_score", table.equit_threat_score());
    check("bias_score", table.bias_score());
    check("peirce_score", table.peirce_score());
    check("heidke_score", table.heidke_score());
}
