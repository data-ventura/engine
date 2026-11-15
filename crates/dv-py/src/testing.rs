//! A Python environment for tests and benchmarks: a venv with the `dataventura` package (editable)
//! and the libraries the fixtures use, built once with uv under `target/`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// The Python interpreter to test with: `DV_PYTHON` if set, else a venv built with uv. `None` if
/// neither is available, so tests can skip with a message instead of failing on machines without
/// uv.
pub fn python() -> Option<PathBuf> {
    static PY: OnceLock<Option<PathBuf>> = OnceLock::new();
    PY.get_or_init(|| {
        if let Some(p) = std::env::var_os("DV_PYTHON") {
            return Some(p.into());
        }
        let root = repo_root();
        let venv = root.join("target/dv-py-venv");
        let py = venv.join("bin/python");
        let ok = |c: &mut Command| c.status().is_ok_and(|s| s.success());
        if !py.exists() {
            let made = ok(Command::new("uv")
                .args(["venv", "-q", "--python", "3.13"])
                .arg(&venv))
                && ok(Command::new("uv")
                    .args(["pip", "install", "-q", "--python"])
                    .arg(&py)
                    .args(["-e"])
                    .arg(root.join("python/dataventura"))
                    .args(["polars", "pandas", "numpy"]));
            if !made {
                let _ = std::fs::remove_dir_all(&venv);
                return None;
            }
        }
        Some(py)
    })
    .clone()
}

/// The engine repository's root.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/dv-py is two levels down")
        .to_path_buf()
}
