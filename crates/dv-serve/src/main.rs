//! `dv-serve`: the remote helper started by Data Ventura over SSH.
//!
//! It speaks a request/response protocol on stdin/stdout (Arrow IPC for tables,
//! compressed tiles for arrays). Not implemented yet.

fn main() {
    eprintln!(
        "dv-serve {}: not implemented yet",
        env!("CARGO_PKG_VERSION")
    );
    std::process::exit(2);
}
