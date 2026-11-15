//! The .dvspec workspace format, its JSON Schema and lock files.
//!
//! Part of the Data Ventura engine. See the repository README for the crate map.

pub mod lock;
pub mod workspace;

pub use lock::{Fingerprint, Lock, Replay, fingerprint, replay};
pub use workspace::{Workspace, parse_duration};
