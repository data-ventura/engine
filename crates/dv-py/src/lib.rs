//! The connected-runtime host: control channel and Arrow shared memory for Python (and later R and
//! Julia).
//!
//! - [`worker`]: a warm worker process per environment, its JSON-RPC control channel, memory cap
//!   and cancellation.
//! - [`shm`]: POSIX shared-memory segments for Arrow IPC data, and the registry that releases a
//!   crashed process's segments.
//!
//! The protocol is documented in `python/dataventura/PROTOCOL.md`, so any language with Arrow
//! support can implement a worker.

pub mod shm;
pub mod testing;
pub mod worker;

pub use worker::{Error, Output, Timings, TransformInfo, Worker, WorkerConfig};
