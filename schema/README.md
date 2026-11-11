# .dvspec JSON Schema (draft)

A `.dvspec` file is a TOML workspace: sources, field role fixes, derived fields and joins (as recipes), views and encodings, layout, selection and lenses. It never contains data or secrets. This schema describes the TOML after parsing, so editors and agents can validate it.
