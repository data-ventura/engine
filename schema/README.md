# .dvspec JSON Schema (draft)

A `.dvspec` file is a TOML workspace: sources, field role and unit fixes, derived fields and joins (as recipes), views and encodings, layout and selection. It never contains data or secrets: credentials stay in the keychain behind a named `connection`. This schema describes the TOML after parsing, so editors and agents can validate it. `dv-spec` reads the same format and also checks what a schema can't: that recipes refer to sources that exist, that units parse, and that nothing looks like a secret.

`fixtures/spec/example.dvspec` is a complete example.

Next to a workspace, `dvspec.lock` records what a result depended on: each source's size, modification time and SHA-256 of its metadata (a Parquet footer, Zarr metadata files, or the first 64 KiB of other files), optionally of its whole content, plus the engine and operation versions. Replaying a workspace compares the lock with the present: exact, best effort (with each difference listed), or unavailable.
