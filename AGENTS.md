# Data Ventura engine: instructions for agents

This is the **public, Apache-2.0** engine for Data Ventura. Everything committed here is public.

## Rules

- **Never** add code, names, paths or details from the private app repository (`data-ventura/app`), its packs, licensing or business plans.
- The engine is **read-only**: it never writes to or modifies source data.
- Keep the dependency direction: the app depends on the engine, never the reverse. No dependency on SourceRift's chassis.
- Dependencies must be permissively licensed (`deny.toml`).
- Every numeric result needs a reference test (polars, DuckDB, xarray, GDAL) on a small fixture.
- Run `cargo fmt --all`, `cargo clippy --all-targets -- -D warnings` and `cargo test` before reporting work as done.
- Ask before committing, pushing or tagging.

## Layout

See README.md for the crate map. Package names use the `dataventura-` prefix; library names use `dv_`.
