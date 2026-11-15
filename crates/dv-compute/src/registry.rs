//! The versioned registry of operations.
//!
//! Every operation whose result a workspace can depend on has a version here. Lock files record
//! the versions a result was computed with, so replay can tell when an operation has changed.
//! **Bump an operation's version whenever its results can change**, even by rounding.

/// One registered operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operation {
    pub name: &'static str,
    pub version: u32,
}

const fn op(name: &'static str, version: u32) -> Operation {
    Operation { name, version }
}

/// Every operation, by name.
pub const OPERATIONS: &[Operation] = &[
    // Formulas (dv_compute::formula): the language and each function.
    op("formula", 1),
    op("formula.window", 1),
    op("formula.units", 1),
    // Selections and joins (dv-model, dv-query).
    op("selection", 1),
    op("join.key", 1),
    // Profiles (dv-model).
    op("profile", 1),
    // Verification metrics (dv_compute::metrics).
    op("metric.bias", 1),
    op("metric.mae", 1),
    op("metric.mse", 1),
    op("metric.rmse", 1),
    op("metric.pearson_r", 1),
    op("metric.spearman_r", 1),
    op("metric.crps_ensemble", 1),
    op("metric.brier_score", 1),
    op("metric.contingency", 1),
];

/// The version of an operation, if it's registered.
pub fn version(name: &str) -> Option<u32> {
    OPERATIONS
        .iter()
        .find(|o| o.name == name)
        .map(|o| o.version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique() {
        let mut names: Vec<_> = OPERATIONS.iter().map(|o| o.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), OPERATIONS.len());
        assert_eq!(version("metric.rmse"), Some(1));
        assert_eq!(version("nope"), None);
    }
}
