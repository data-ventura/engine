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

## Developer Certificate of Origin

Every commit must be signed off (`git commit -s`), certifying the [Developer Certificate of Origin 1.1](https://developercertificate.org/). There is no CLA.

## Code of conduct

This project follows the [Contributor Covenant 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Report problems to conduct@dataventura.com.
