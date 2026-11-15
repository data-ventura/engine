//! A warm Python worker: one process per environment, kept running between requests.
//!
//! The control channel is JSON-RPC 2.0 over the worker's stdin and stdout, one message per line.
//! Tables travel as Arrow IPC streams in POSIX shared memory: the host writes the input into a
//! segment it creates, and the worker writes its result into a segment the host named. The host
//! maps the result read-only and decodes it in place, so result columns point into the segment.
//! See `python/dataventura/PROTOCOL.md`.
//!
//! The worker is watched while it runs: its physical footprint is sampled every few milliseconds
//! and the process is stopped if it goes over its cap (SPEC §4.9). A request whose generation
//! passes gets a cancel message, which interrupts Python code at its next bytecode; a worker still
//! busy after the grace period (code in a long C call) is stopped. A stopped or crashed worker is
//! started again on the next request, with the same files loaded.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use arrow::array::RecordBatch;
use arrow::buffer::Buffer;
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow::ipc::reader::StreamDecoder;
use arrow::ipc::writer::{IpcWriteOptions, StreamWriter};
use dv_data::budget::{Budget, Consumer, Refused, Reservation};
use dv_data::request::Generation;
use serde_json::{Value, json};

use crate::shm::{Mapping, Registry};

/// The protocol version this host speaks.
pub const PROTOCOL: u64 = 0;

/// How to start a worker and the limits it runs under.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub env: Vec<(OsString, OsString)>,
    /// The worker is stopped when its physical footprint goes over this.
    pub memory_cap: u64,
    /// How long a cancelled request may keep running before the worker is stopped.
    pub cancel_grace: Duration,
    /// How often the worker's memory and the request's generation are checked.
    pub poll: Duration,
    /// How long to wait for a new worker to say it's ready.
    pub start_timeout: Duration,
}

impl WorkerConfig {
    /// `python -m dataventura.worker`, capped at a quarter of physical memory, with the 2 s grace
    /// period SPEC §4.9 gives cancelled Python code.
    pub fn python(python: impl Into<PathBuf>) -> Self {
        Self {
            program: python.into(),
            args: vec!["-m".into(), "dataventura.worker".into()],
            env: Vec::new(),
            memory_cap: dv_data::budget::physical_memory() / 4,
            cancel_grace: Duration::from_secs(2),
            poll: Duration::from_millis(5),
            start_timeout: Duration::from_secs(60),
        }
    }
}

/// Why a request failed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("couldn't start the Python worker: {0}")]
    Start(String),
    #[error("{message}")]
    Python {
        message: String,
        traceback: Option<String>,
    },
    #[error("cancelled{}", if *.stopped { "; the worker didn't stop in time and was restarted" } else { "" })]
    Cancelled { stopped: bool },
    #[error(
        "the Python worker used {} MB, over its {} MB memory cap, and was stopped",
        used / 1_000_000,
        cap / 1_000_000
    )]
    MemoryCap { used: u64, cap: u64 },
    #[error("the Python worker exited ({status}){}", stderr_hint(.stderr))]
    Exited { status: String, stderr: String },
    #[error(transparent)]
    Budget(#[from] Refused),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Arrow(#[from] ArrowError),
}

fn stderr_hint(s: &str) -> String {
    let tail: Vec<&str> = s.lines().rev().take(5).collect();
    if tail.is_empty() {
        String::new()
    } else {
        format!(
            ": {}",
            tail.into_iter().rev().collect::<Vec<_>>().join(" / ")
        )
    }
}

/// A transform the worker has registered.
#[derive(Debug, Clone)]
pub struct TransformInfo {
    pub name: String,
    pub input_type: String,
    /// The input columns it reads, if it said.
    pub columns: Option<Vec<String>>,
}

/// Where a request's time went, in milliseconds.
#[derive(Debug, Clone, Default)]
pub struct Timings {
    /// Host: encoding the input as Arrow IPC into shared memory.
    pub host_encode: f64,
    /// Worker: phases reported by the worker (attach, decode, to_input, user, to_table, encode).
    pub worker: Vec<(String, f64)>,
    /// Host: mapping and decoding the result.
    pub host_decode: f64,
    /// Wall time from the start of the call to a decoded result.
    pub total: f64,
}

impl Timings {
    /// Time spent in the user's function.
    pub fn user(&self) -> f64 {
        self.worker
            .iter()
            .find(|(k, _)| k == "user")
            .map_or(0.0, |(_, v)| *v)
    }

    /// Everything that isn't the user's function: what Data Ventura adds.
    pub fn overhead(&self) -> f64 {
        self.total - self.user()
    }
}

/// A transform's result. Its columns point into a shared-memory segment that stays mapped, and
/// counted in the memory budget, while any of them is alive.
#[derive(Debug)]
pub struct Output {
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
    pub timings: Timings,
}

/// The mapped result segment and its budget reservation, released together when the last buffer
/// pointing into it is dropped.
#[derive(Debug)]
struct Held {
    _mapping: Mapping,
    _reservation: Reservation,
}

struct Process {
    child: Child,
    stdin: ChildStdin,
    replies: Receiver<Value>,
    pid: u32,
    /// Footprint over the cap seen by the watcher (0 while under).
    over_cap: Arc<AtomicU64>,
    peak: Arc<AtomicU64>,
    stop_watch: Arc<AtomicBool>,
    info: Value,
}

impl Drop for Process {
    fn drop(&mut self) {
        self.stop_watch.store(true, Ordering::Release);
        let _ = writeln!(self.stdin, r#"{{"jsonrpc":"2.0","method":"shutdown"}}"#);
        let _ = self.stdin.flush();
        let deadline = Instant::now() + Duration::from_millis(500);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            thread::sleep(Duration::from_millis(5));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A warm worker. Not `Sync`: one request at a time, as SPEC §4.9 allows per view.
pub struct Worker {
    config: WorkerConfig,
    budget: Budget,
    registry: Arc<Registry>,
    process: Option<Process>,
    loaded: Vec<PathBuf>,
    transforms: Vec<TransformInfo>,
    stderr: Arc<Mutex<VecDeque<u8>>>,
    next_id: u64,
    starts: u64,
}

impl std::fmt::Debug for Worker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Worker")
            .field("program", &self.config.program)
            .field("pid", &self.process.as_ref().map(|p| p.pid))
            .field("starts", &self.starts)
            .finish_non_exhaustive()
    }
}

const STDERR_KEEP: usize = 64 * 1024;

impl Worker {
    /// Starts a worker. Its shared memory is counted against `budget` as [`Consumer::Python`].
    pub fn start(
        config: WorkerConfig,
        budget: Budget,
        registry: Arc<Registry>,
    ) -> Result<Self, Error> {
        let mut w = Self {
            config,
            budget,
            registry,
            process: None,
            loaded: Vec::new(),
            transforms: Vec::new(),
            stderr: Arc::new(Mutex::new(VecDeque::new())),
            next_id: 0,
            starts: 0,
        };
        w.ensure_running()?;
        Ok(w)
    }

    /// What the worker reported when it started: protocol, Python and pyarrow versions, pid.
    pub fn info(&self) -> Option<&Value> {
        self.process.as_ref().map(|p| &p.info)
    }

    /// How many times a worker process has been started, including restarts.
    pub fn starts(&self) -> u64 {
        self.starts
    }

    /// The worker's peak physical footprint, as sampled, in bytes.
    pub fn peak_footprint(&self) -> u64 {
        self.process
            .as_ref()
            .map_or(0, |p| p.peak.load(Ordering::Relaxed))
    }

    /// The last 64 KB the worker wrote to stderr, including anything the user's code printed.
    pub fn stderr_tail(&self) -> String {
        let buf = self.stderr.lock().expect("stderr lock");
        String::from_utf8_lossy(&buf.iter().copied().collect::<Vec<_>>()).into_owned()
    }

    /// Imports a `.py` file in the worker and returns every transform registered so far.
    pub fn load(&mut self, path: &Path) -> Result<Vec<TransformInfo>, Error> {
        self.ensure_running()?;
        let reply = self.call("load", json!({ "path": path }), None)?;
        if !self.loaded.iter().any(|p| p == path) {
            self.loaded.push(path.to_path_buf());
        }
        self.transforms = parse_transforms(&reply)?;
        Ok(self.transforms.clone())
    }

    pub fn transforms(&self) -> &[TransformInfo] {
        &self.transforms
    }

    /// Runs transform `name` on `batches`. If the transform declared the columns it reads, only
    /// those are sent. If `generation` passes while it runs, the worker is asked to cancel, and
    /// stopped if it hasn't within the grace period.
    pub fn transform(
        &mut self,
        name: &str,
        schema: &SchemaRef,
        batches: &[RecordBatch],
        generation: Option<&Generation>,
    ) -> Result<Output, Error> {
        let t0 = Instant::now();
        self.ensure_running()?;
        let columns = self
            .transforms
            .iter()
            .find(|t| t.name == name)
            .and_then(|t| t.columns.clone());
        let projected;
        let (schema, batches) = match columns {
            Some(cols) => {
                let idx = cols
                    .iter()
                    .map(|c| schema.index_of(c))
                    .collect::<Result<Vec<_>, _>>()?;
                projected = batches
                    .iter()
                    .map(|b| b.project(&idx))
                    .collect::<Result<Vec<_>, _>>()?;
                (&Arc::new(schema.project(&idx)?), projected.as_slice())
            }
            None => (schema, batches),
        };

        // The input, as an Arrow IPC stream in a segment of exactly its size.
        let mut ipc = Vec::with_capacity(
            batches
                .iter()
                .map(RecordBatch::get_array_memory_size)
                .sum::<usize>()
                + 4096,
        );
        {
            let opts = IpcWriteOptions::default();
            let mut w = StreamWriter::try_new_with_options(&mut ipc, schema, opts)?;
            for b in batches {
                w.write(b)?;
            }
            w.finish()?;
        }
        let _input_reservation = self.budget.reserve(Consumer::Python, ipc.len() as u64)?;
        let input_name = self.registry.name()?;
        let output_name = match self.registry.name() {
            Ok(n) => n,
            Err(e) => {
                self.registry.unlink(&input_name);
                return Err(e.into());
            }
        };
        let result = (|| {
            let input_len = ipc.len();
            let mut input = Mapping::create(&input_name, input_len)?;
            input.as_mut_slice().copy_from_slice(&ipc);
            drop(ipc);
            drop(input);
            let host_encode = ms(t0.elapsed());
            let reply = self.call(
                "transform",
                json!({
                    "name": name,
                    "generation": generation.map(Generation::value),
                    "input": { "segment": input_name, "size": input_len },
                    "output": { "segment": output_name },
                }),
                generation,
            )?;
            self.registry.unlink(&input_name);
            let t1 = Instant::now();
            let size = reply["output"]["size"]
                .as_u64()
                .ok_or_else(|| Error::Protocol("transform reply without output.size".into()))?;
            let mapping = Mapping::open(&output_name, size as usize)?;
            // The name can go now: the mapping keeps the memory until the result is dropped.
            self.registry.unlink(&output_name);
            let reservation = self.budget.reserve(Consumer::Python, size)?;
            let (schema, batches) = decode(mapping, reservation)?;
            let host_decode = ms(t1.elapsed());
            let worker = reply["timings"]
                .as_object()
                .map(|o| {
                    o.iter()
                        .map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0)))
                        .collect()
                })
                .unwrap_or_default();
            Ok(Output {
                schema,
                batches,
                timings: Timings {
                    host_encode,
                    worker,
                    host_decode,
                    total: ms(t0.elapsed()),
                },
            })
        })();
        // Unlinking twice is harmless; on success both are already gone.
        self.registry.unlink(&input_name);
        self.registry.unlink(&output_name);
        result
    }

    /// Starts the worker if it isn't running (first use, or after it was stopped or crashed), and
    /// loads the files the previous one had loaded.
    fn ensure_running(&mut self) -> Result<(), Error> {
        if let Some(p) = &mut self.process {
            match p.child.try_wait() {
                Ok(None) => return Ok(()),
                _ => self.process = None,
            }
        }
        self.process = Some(self.spawn()?);
        self.starts += 1;
        for path in self.loaded.clone() {
            let reply = self.call("load", json!({ "path": path }), None)?;
            self.transforms = parse_transforms(&reply)?;
        }
        Ok(())
    }

    fn spawn(&mut self) -> Result<Process, Error> {
        let mut cmd = Command::new(&self.config.program);
        cmd.args(&self.config.args)
            .envs(self.config.env.iter().map(|(k, v)| (k, v)))
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd
            .spawn()
            .map_err(|e| Error::Start(format!("{}: {e}", self.config.program.display())))?;
        let pid = child.id();
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let mut stderr = child.stderr.take().expect("piped stderr");

        let (tx, replies) = mpsc::channel();
        thread::Builder::new()
            .name("dv-py-stdout".into())
            .spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let Ok(line) = line else { break };
                    if let Ok(v) = serde_json::from_str::<Value>(&line)
                        && tx.send(v).is_err()
                    {
                        break;
                    }
                }
            })?;
        let ring = self.stderr.clone();
        thread::Builder::new()
            .name("dv-py-stderr".into())
            .spawn(move || {
                let mut buf = [0u8; 4096];
                while let Ok(n) = stderr.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    let mut r = ring.lock().expect("stderr lock");
                    r.extend(&buf[..n]);
                    let excess = r.len().saturating_sub(STDERR_KEEP);
                    r.drain(..excess);
                }
            })?;

        let (over_cap, peak, stop_watch) = (
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicBool::new(false)),
        );
        {
            let (over_cap, peak, stop) = (over_cap.clone(), peak.clone(), stop_watch.clone());
            let (cap, every) = (self.config.memory_cap, self.config.poll);
            thread::Builder::new()
                .name("dv-py-watch".into())
                .spawn(move || {
                    while !stop.load(Ordering::Acquire) {
                        let Some(used) = footprint(pid) else { break };
                        peak.fetch_max(used, Ordering::Relaxed);
                        if used > cap {
                            over_cap.store(used, Ordering::Release);
                            // SAFETY: kill has no memory-safety preconditions; pid is our child,
                            // which can't be reaped (and its pid reused) while we hold its Child.
                            unsafe {
                                libc::kill(pid as libc::pid_t, libc::SIGKILL);
                            }
                            break;
                        }
                        thread::sleep(every);
                    }
                })?;
        }

        let mut p = Process {
            child,
            stdin,
            replies,
            pid,
            over_cap,
            peak,
            stop_watch,
            info: Value::Null,
        };
        let ready = match p.replies.recv_timeout(self.config.start_timeout) {
            Ok(v) if v["method"] == "ready" => v,
            Ok(v) => return Err(Error::Protocol(format!("expected ready, got {v}"))),
            Err(_) => {
                let status = p.child.try_wait().ok().flatten();
                drop(p);
                return Err(Error::Start(format!(
                    "the worker didn't start{}{}",
                    status.map_or(String::new(), |s| format!(" ({s})")),
                    stderr_hint(&self.stderr_tail())
                )));
            }
        };
        let protocol = ready["params"]["protocol"].as_u64();
        if protocol != Some(PROTOCOL) {
            return Err(Error::Start(format!(
                "the worker speaks protocol {protocol:?}, this host {PROTOCOL}; update the dataventura package"
            )));
        }
        p.info = ready["params"].clone();
        Ok(p)
    }

    /// Sends one request and waits for its reply, watching the worker and the generation.
    fn call(
        &mut self,
        method: &str,
        params: Value,
        generation: Option<&Generation>,
    ) -> Result<Value, Error> {
        self.next_id += 1;
        let id = self.next_id;
        let (grace, poll) = (self.config.cancel_grace, self.config.poll);
        let p = self.process.as_mut().expect("running");
        let msg = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        writeln!(p.stdin, "{msg}")?;
        p.stdin.flush()?;

        let mut cancelled_at: Option<Instant> = None;
        loop {
            match p.replies.recv_timeout(poll) {
                Ok(v) if v["id"] == id => {
                    if let Some(err) = v.get("error") {
                        let code = err["code"].as_i64().unwrap_or(0);
                        if code == -32001 {
                            return Err(Error::Cancelled { stopped: false });
                        }
                        return Err(Error::Python {
                            message: err["message"].as_str().unwrap_or("error").to_string(),
                            traceback: err["data"].as_str().map(str::to_string),
                        });
                    }
                    return Ok(v["result"].clone());
                }
                Ok(_) => continue, // a reply to an earlier, abandoned request
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    let status = p.child.wait().ok();
                    let used = p.over_cap.load(Ordering::Acquire);
                    self.process = None;
                    if used > 0 {
                        return Err(Error::MemoryCap {
                            used,
                            cap: self.config.memory_cap,
                        });
                    }
                    return Err(Error::Exited {
                        status: status.map_or("unknown".into(), describe),
                        stderr: self.stderr_tail(),
                    });
                }
            }
            if generation.is_some_and(|g| !g.is_current()) {
                match cancelled_at {
                    None => {
                        let cancel =
                            json!({ "jsonrpc": "2.0", "method": "cancel", "params": { "id": id } });
                        let _ = writeln!(p.stdin, "{cancel}");
                        let _ = p.stdin.flush();
                        cancelled_at = Some(Instant::now());
                    }
                    Some(at) if at.elapsed() > grace => {
                        self.process = None; // Drop stops it
                        return Err(Error::Cancelled { stopped: true });
                    }
                    Some(_) => {}
                }
            }
        }
    }
}

fn parse_transforms(reply: &Value) -> Result<Vec<TransformInfo>, Error> {
    reply["transforms"]
        .as_array()
        .ok_or_else(|| Error::Protocol("load reply without transforms".into()))?
        .iter()
        .map(|t| {
            Ok(TransformInfo {
                name: t["name"].as_str().unwrap_or_default().to_string(),
                input_type: t["input_type"].as_str().unwrap_or_default().to_string(),
                columns: t["columns"].as_array().map(|c| {
                    c.iter()
                        .filter_map(|s| s.as_str().map(str::to_string))
                        .collect()
                }),
            })
        })
        .collect()
}

/// Decodes the IPC stream in `mapping` in place: the batches' buffers point into the segment.
fn decode(
    mapping: Mapping,
    reservation: Reservation,
) -> Result<(SchemaRef, Vec<RecordBatch>), Error> {
    let (ptr, len) = (mapping.as_ptr(), mapping.len());
    let owner = Arc::new(Held {
        _mapping: mapping,
        _reservation: reservation,
    });
    // SAFETY: `ptr..ptr+len` is the read-only mapping `owner` keeps alive for as long as any
    // buffer sliced from this one exists.
    let mut buf = unsafe { Buffer::from_custom_allocation(ptr, len, owner) };
    let mut decoder = StreamDecoder::new();
    let mut batches = Vec::new();
    while !buf.is_empty() {
        if let Some(b) = decoder.decode(&mut buf)? {
            batches.push(b);
        }
    }
    decoder.finish()?;
    let schema = decoder
        .schema()
        .ok_or_else(|| Error::Protocol("result stream without a schema".into()))?;
    Ok((schema, batches))
}

fn footprint(pid: u32) -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: proc_pid_rusage writes a rusage_info_v4 into the buffer we pass.
        unsafe {
            let mut info = std::mem::zeroed::<libc::rusage_info_v4>();
            let rc = libc::proc_pid_rusage(
                pid as libc::c_int,
                libc::RUSAGE_INFO_V4,
                (&raw mut info).cast::<libc::rusage_info_t>(),
            );
            (rc == 0).then_some(info.ri_phys_footprint)
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let statm = std::fs::read_to_string(format!("/proc/{pid}/statm")).ok()?;
        let rss_pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
        // SAFETY: sysconf has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        Some(rss_pages * page)
    }
}

fn describe(s: ExitStatus) -> String {
    use std::os::unix::process::ExitStatusExt;
    match (s.code(), s.signal()) {
        (Some(c), _) => format!("exit code {c}"),
        (None, Some(sig)) => format!("signal {sig}"),
        _ => s.to_string(),
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}
