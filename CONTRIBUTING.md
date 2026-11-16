# Contributing

Thanks for your interest. The engine is early, so please open an issue to discuss anything larger than a small fix before writing code.

## Ground rules

- **Read-only.** The engine never modifies source data.
- **Correctness first.** Anything that computes a number needs a test against a reference (polars, DuckDB, xarray or GDAL) on a fixture.
- **No proprietary dependencies.** The engine must not depend on the Data Ventura apps, packs or licensing code. `cargo deny check` enforces this.
- **Permissive licenses only** for dependencies (see `deny.toml`).

## Before opening a pull request

```sh
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test
```

Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

## Golden images

`crates/dv-render/tests/golden.rs` draws reference charts in the light and dark themes and compares them with the PNGs and PDFs in `crates/dv-render/tests/golden/`: the SVG export rasterized by resvg, the PDF export rasterized by hayro, and (when a GPU is available) the vello screen path. Images are compared per 1 pt cell with a CIEDE2000 tolerance, so anti-aliasing differences between rasterizers pass and anything that moves does not. On a failure the drawn images and diffs (differing cells in magenta) are written to `target/golden/`; CI uploads them as an artifact.

After a change that's meant to alter the charts, update the goldens, look at every changed PNG, and regenerate the PDF text reference:

```sh
DV_UPDATE_GOLDENS=1 cargo test -p dataventura-render --test golden
uv run --with pypdfium2 python scripts/reference/pdf-text.py crates/dv-render/tests/golden fixtures/reference/pdf-text.json
cargo test -p dataventura-render --test golden
```

Commit the goldens with the change that caused them, and say why they changed.

## Developer Certificate of Origin

Every commit must be signed off (`git commit -s`), certifying the [Developer Certificate of Origin 1.1](https://developercertificate.org/). There is no CLA.

## Code of conduct

This project follows the [Contributor Covenant 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Report problems to conduct@dataventura.com.
