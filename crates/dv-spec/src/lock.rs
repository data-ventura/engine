//! `dvspec.lock`: what a result depended on, and whether it can be replayed exactly.
//!
//! For each source the lock records a fingerprint: its identity, size, modification time and a
//! SHA-256 of its metadata (the Parquet footer, the Zarr metadata files, or the first 64 KiB of
//! other files, where NetCDF, HDF5 and text headers live), with an optional hash of the whole
//! content. It also records the engine version, each operation's version, and extension and
//! Python environment hashes.
//!
//! [`replay`] compares a lock with the present: **exact** when every fingerprint and version
//! matches, **best effort** when something changed (each difference is listed), and
//! **unavailable** when an input is missing. A modification time alone doesn't make a replay
//! inexact when size and hashes still match: copying or checking out a file changes it.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Bytes hashed from the start of files whose metadata is a header.
pub const HEADER_BYTES: u64 = 64 * 1024;

/// A lock file.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lock {
    /// The engine that wrote it.
    pub engine: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sources: BTreeMap<String, Fingerprint>,
    /// Operation versions from dv-compute's registry.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub operations: BTreeMap<String, u32>,
    /// Extension hashes, by extension name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub python: Option<PythonEnv>,
}

/// A source's fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fingerprint {
    /// The path or URL as the workspace names it.
    pub identity: String,
    pub size: u64,
    /// Seconds since the Unix epoch.
    pub modified: i64,
    pub metadata_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_sha256: Option<String>,
    /// Object store ETag or version ID, for remote sources.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub etag: Option<String>,
}

/// The Python environment a result used.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonEnv {
    pub interpreter: String,
    /// A hash of the environment's lock file or frozen package list.
    pub environment_sha256: String,
}

impl Lock {
    /// An empty lock for this engine version.
    pub fn new() -> Self {
        Self {
            engine: env!("CARGO_PKG_VERSION").to_string(),
            ..Default::default()
        }
    }

    pub fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("a lock always serializes")
    }
}

fn hex(d: impl AsRef<[u8]>) -> String {
    d.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

/// Fingerprints a local file or directory (a Zarr store). With `content`, also hashes every byte.
pub fn fingerprint(path: &Path, identity: &str, content: bool) -> std::io::Result<Fingerprint> {
    let meta = std::fs::metadata(path)?;
    let modified = meta
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    if meta.is_dir() {
        let files = dir_files(path)?;
        let size = files.iter().map(|(_, s)| s).sum();
        let mut h = Sha256::new();
        let mut all = Sha256::new();
        for (rel, _) in &files {
            let name = rel.rsplit('/').next().unwrap_or(rel);
            if matches!(
                name,
                ".zmetadata" | ".zarray" | ".zattrs" | ".zgroup" | "zarr.json"
            ) {
                h.update(rel.as_bytes());
                h.update([0]);
                h.update(std::fs::read(path.join(rel))?);
            }
            if content {
                all.update(rel.as_bytes());
                all.update([0]);
                all.update(std::fs::read(path.join(rel))?);
            }
        }
        return Ok(Fingerprint {
            identity: identity.into(),
            size,
            modified,
            metadata_sha256: hex(h.finalize()),
            content_sha256: content.then(|| hex(all.finalize())),
            etag: None,
        });
    }
    let size = meta.len();
    let mut f = File::open(path)?;
    let metadata_sha256 = hex(Sha256::digest(metadata_bytes(&mut f, size)?));
    let content_sha256 = if content {
        f.seek(SeekFrom::Start(0))?;
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        Some(hex(h.finalize()))
    } else {
        None
    };
    Ok(Fingerprint {
        identity: identity.into(),
        size,
        modified,
        metadata_sha256,
        content_sha256,
        etag: None,
    })
}

/// The bytes that describe a file: a Parquet footer (with its length and magic), or the header.
fn metadata_bytes(f: &mut File, size: u64) -> std::io::Result<Vec<u8>> {
    let mut magic = [0u8; 4];
    let is_parquet = size >= 12 && {
        f.read_exact(&mut magic)?;
        &magic == b"PAR1"
    };
    if is_parquet {
        let mut tail = [0u8; 8];
        f.seek(SeekFrom::End(-8))?;
        f.read_exact(&mut tail)?;
        if &tail[4..] == b"PAR1" {
            let len = u32::from_le_bytes(tail[..4].try_into().expect("4 bytes")) as u64 + 8;
            if len <= size {
                f.seek(SeekFrom::End(-(len as i64)))?;
                let mut footer = vec![0; len as usize];
                f.read_exact(&mut footer)?;
                return Ok(footer);
            }
        }
    }
    f.seek(SeekFrom::Start(0))?;
    let mut head = Vec::new();
    f.take(HEADER_BYTES).read_to_end(&mut head)?;
    Ok(head)
}

/// Files under a directory with sizes, as sorted `/`-separated relative paths.
fn dir_files(root: &Path) -> std::io::Result<Vec<(String, u64)>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir)? {
            let e = e?;
            let p = e.path();
            if e.file_type()?.is_dir() {
                stack.push(p);
            } else {
                let rel = p
                    .strip_prefix(root)
                    .expect("under root")
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                out.push((rel, e.metadata()?.len()));
            }
        }
    }
    out.sort();
    Ok(out)
}

/// How faithfully a locked result can be replayed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "level", content = "details", rename_all = "snake_case")]
pub enum Replay {
    Exact,
    /// Something changed; the workspace runs, and these differences are shown.
    BestEffort(Vec<String>),
    /// These inputs are missing; their views show placeholders.
    Unavailable(Vec<String>),
}

/// Compares a lock with the present. `current` holds fingerprints for the sources that still
/// exist (missing ones are left out) and the present operation and extension versions.
pub fn replay(locked: &Lock, current: &Lock) -> Replay {
    let missing: Vec<String> = locked
        .sources
        .keys()
        .filter(|k| !current.sources.contains_key(*k))
        .map(|k| format!("source {k} is missing"))
        .collect();
    if !missing.is_empty() {
        return Replay::Unavailable(missing);
    }
    let mut diffs = Vec::new();
    if locked.engine != current.engine {
        diffs.push(format!("engine {} → {}", locked.engine, current.engine));
    }
    for (name, was) in &locked.sources {
        let now = &current.sources[name];
        if was.size != now.size {
            diffs.push(format!("source {name}: size {} → {}", was.size, now.size));
        } else if was.metadata_sha256 != now.metadata_sha256 {
            diffs.push(format!("source {name}: metadata changed"));
        } else if let (Some(a), Some(b)) = (&was.content_sha256, &now.content_sha256)
            && a != b
        {
            diffs.push(format!("source {name}: content changed"));
        } else if let (Some(a), Some(b)) = (&was.etag, &now.etag)
            && a != b
        {
            diffs.push(format!("source {name}: object version changed"));
        }
    }
    for (op, v) in &locked.operations {
        match current.operations.get(op) {
            Some(now) if now == v => {}
            Some(now) => diffs.push(format!("operation {op}: version {v} → {now}")),
            None => diffs.push(format!("operation {op} no longer exists")),
        }
    }
    for (ext, h) in &locked.extensions {
        if current.extensions.get(ext) != Some(h) {
            diffs.push(format!("extension {ext} changed"));
        }
    }
    if locked.python != current.python {
        diffs.push("the Python environment changed".into());
    }
    if diffs.is_empty() {
        Replay::Exact
    } else {
        Replay::BestEffort(diffs)
    }
}
