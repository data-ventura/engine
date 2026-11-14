//! The M0 spike's benchmark harness.
//!
//! Every case records wall time, CPU time and peak resident memory separately, and appends one JSON
//! line per metric to `target/spike/results.jsonl`, so runs can be compared over time.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::Serialize;

pub mod era5;
#[cfg(target_os = "macos")]
pub mod live;
pub mod taxi;

/// Every spike binary gives freed large blocks back to the system, as the app will.
#[global_allocator]
static ALLOC: dv_data::alloc::ReturningAlloc = dv_data::alloc::ReturningAlloc;

/// CPU time (user + system) used by this process so far.
pub fn cpu_time() -> Duration {
    let ru = rusage();
    let tv = |t: libc::timeval| Duration::new(t.tv_sec as u64, t.tv_usec as u32 * 1000);
    tv(ru.ru_utime) + tv(ru.ru_stime)
}

/// Peak resident set size of this process so far, in bytes.
pub fn peak_rss() -> u64 {
    // macOS reports ru_maxrss in bytes; Linux reports kilobytes.
    let max = rusage().ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        max
    } else {
        max * 1024
    }
}

/// Physical footprint of this process now and at its peak, in bytes: what macOS charges the process
/// for memory, including GPU buffers in unified memory and compressed pages. This is the number
/// Activity Monitor shows and memory pressure acts on.
#[cfg(target_os = "macos")]
pub fn footprint() -> (u64, u64) {
    // SAFETY: proc_pid_rusage writes a rusage_info_v4 into the buffer we pass, which is plain old
    // data of the right size for RUSAGE_INFO_V4.
    unsafe {
        let mut info = std::mem::zeroed::<libc::rusage_info_v4>();
        let rc = libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            (&raw mut info).cast::<libc::rusage_info_t>(),
        );
        assert_eq!(rc, 0, "proc_pid_rusage");
        (info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint)
    }
}

/// The fraction of `path`'s pages in the OS file cache, to confirm a cold run is cold. Maps the
/// file read-only and asks the kernel which pages are resident, without reading any.
pub fn cached_fraction(path: &std::path::Path) -> f64 {
    use std::os::fd::AsRawFd;
    let file = fs::File::open(path).expect("open");
    let len = file.metadata().expect("metadata").len() as usize;
    if len == 0 {
        return 0.0;
    }
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let pages = len.div_ceil(page);
    let mut resident = vec![0 as libc::c_char; pages];
    // SAFETY: a read-only shared mapping of a file we hold open, of its length; mincore writes one
    // byte per page into a buffer of exactly that many bytes; the mapping is unmapped before
    // returning and never read.
    unsafe {
        let addr = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        );
        assert_ne!(addr, libc::MAP_FAILED, "mmap");
        let rc = libc::mincore(addr, len, resident.as_mut_ptr());
        libc::munmap(addr, len);
        assert_eq!(rc, 0, "mincore");
    }
    resident.iter().filter(|&&b| b & 1 != 0).count() as f64 / pages as f64
}

fn rusage() -> libc::rusage {
    // SAFETY: getrusage only writes into the struct we pass, which is plain old data.
    unsafe {
        let mut ru = std::mem::zeroed::<libc::rusage>();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        ru
    }
}

/// Wall and CPU time for one timed span.
#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub wall: Duration,
    pub cpu: Duration,
}

/// Runs `f` and measures its wall and CPU time.
pub fn time<T>(f: impl FnOnce() -> T) -> (T, Span) {
    let (c0, t0) = (cpu_time(), Instant::now());
    let out = f();
    let span = Span {
        wall: t0.elapsed(),
        cpu: cpu_time() - c0,
    };
    (out, span)
}

/// Summary statistics for repeated samples, in milliseconds.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Summary {
    pub n: usize,
    pub min: f64,
    pub p50: f64,
    pub p95: f64,
    pub max: f64,
}

impl Summary {
    pub fn of(samples: &[Duration]) -> Self {
        let mut ms: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1e3).collect();
        ms.sort_by(f64::total_cmp);
        let at = |q: f64| ms[((ms.len() - 1) as f64 * q).round() as usize];
        Self {
            n: ms.len(),
            min: ms[0],
            p50: at(0.5),
            p95: at(0.95),
            max: ms[ms.len() - 1],
        }
    }
}

/// Whether a case is measured with warm or cold caches (SPEC §9).
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Caches {
    Warm,
    Cold,
}

/// One recorded metric.
#[derive(Debug, Serialize)]
pub struct Record<'a> {
    pub case: &'a str,
    pub metric: &'a str,
    pub value: f64,
    pub unit: &'a str,
    /// The pass threshold, if this metric has one. Lower is better.
    pub limit: Option<f64>,
    pub pass: Option<bool>,
    pub caches: Caches,
    pub note: &'a str,
}

/// Collects a case's records, prints them, and appends them to the results file.
#[derive(Debug)]
pub struct Report {
    case: String,
    caches: Caches,
    lines: Vec<String>,
    failed: bool,
}

impl Report {
    pub fn new(case: &str, caches: Caches) -> Self {
        Self {
            case: case.into(),
            caches,
            lines: Vec::new(),
            failed: false,
        }
    }

    /// Records a metric. With a limit, the metric passes when `value < limit`.
    pub fn metric(&mut self, metric: &str, value: f64, unit: &str, limit: Option<f64>, note: &str) {
        let pass = limit.map(|l| value < l);
        self.failed |= pass == Some(false);
        let verdict = match pass {
            Some(true) => "PASS",
            Some(false) => "FAIL",
            None => "",
        };
        let lim = limit
            .map(|l| format!("< {} {unit}", (l * 1000.0).round() / 1000.0))
            .unwrap_or_default();
        println!(
            "{:<5} {metric:<28} {value:>12.2} {unit:<5} {lim:<14} {verdict} {note}",
            self.case
        );
        let rec = Record {
            case: &self.case,
            metric,
            value,
            unit,
            limit,
            pass,
            caches: self.caches,
            note,
        };
        self.lines
            .push(serde_json::to_string(&rec).expect("record serializes"));
    }

    /// Records the wall and CPU time of a span, with the limit on wall time.
    pub fn span(&mut self, metric: &str, span: Span, limit_ms: Option<f64>, note: &str) {
        self.metric(
            &format!("{metric}.wall"),
            span.wall.as_secs_f64() * 1e3,
            "ms",
            limit_ms,
            note,
        );
        self.metric(
            &format!("{metric}.cpu"),
            span.cpu.as_secs_f64() * 1e3,
            "ms",
            None,
            "",
        );
    }

    /// Records the p95 of repeated samples against a limit, and the rest of the summary as context.
    pub fn samples(&mut self, metric: &str, samples: &[Duration], limit_p95_ms: Option<f64>) {
        let s = Summary::of(samples);
        let note = format!("n={} p50={:.2} max={:.2}", s.n, s.p50, s.max);
        self.metric(&format!("{metric}.p95"), s.p95, "ms", limit_p95_ms, &note);
    }

    /// Records the current and peak physical footprint, labeled with the phase just finished.
    #[cfg(target_os = "macos")]
    pub fn memory(&mut self, phase: &str) {
        let (now, peak) = footprint();
        self.metric(
            &format!("footprint.{phase}"),
            now as f64 / 1e6,
            "MB",
            None,
            &format!("peak so far {:.0} MB", peak as f64 / 1e6),
        );
    }

    /// Writes peak RSS and appends everything to `target/spike/results.jsonl`.
    /// Returns false if any metric failed its limit.
    pub fn finish(mut self) -> bool {
        self.metric("peak_rss", peak_rss() as f64 / 1e6, "MB", None, "");
        #[cfg(target_os = "macos")]
        self.metric("peak_footprint", footprint().1 as f64 / 1e6, "MB", None, "");
        let dir = workspace_root().join("target/spike");
        fs::create_dir_all(&dir).expect("create target/spike");
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("results.jsonl"))
            .expect("open results.jsonl");
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        for line in &self.lines {
            writeln!(f, "{{\"t\":{stamp},{}", &line[1..]).expect("write results");
        }
        !self.failed
    }
}

/// The engine workspace root.
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("spike has a parent")
        .into()
}

/// A path under `fixtures/spike/data/`, with a hint to run the fetch script if it's missing.
pub fn spike_data(name: &str) -> PathBuf {
    let p = workspace_root().join("fixtures/spike/data").join(name);
    assert!(
        p.exists(),
        "{} is missing: run scripts/fetch-spike-data.sh",
        p.display()
    );
    p
}

/// Saves an RGBA8 texture as a PNG under `target/spike/`. Waits for the GPU.
pub fn save_png(gpu: &dv_render::gpu::Gpu, texture: &wgpu::Texture, size: [u32; 2], name: &str) {
    write_png(&read_rgba(gpu, texture, size), size, name);
}

/// Copies an `Rgba8Unorm` texture's pixels back to the CPU, row by row from the top.
pub fn read_rgba(gpu: &dv_render::gpu::Gpu, texture: &wgpu::Texture, size: [u32; 2]) -> Vec<u8> {
    let row = (size[0] * 4).div_ceil(256) * 256;
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row) * u64::from(size[1]),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([enc.finish()]);
    buf.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
    gpu.wait();
    let data = buf.get_mapped_range(..).expect("mapped");
    let mut pixels = Vec::with_capacity((size[0] * size[1] * 4) as usize);
    for y in 0..size[1] as usize {
        let start = y * row as usize;
        pixels.extend_from_slice(&data[start..start + size[0] as usize * 4]);
    }
    pixels
}

/// Writes RGBA pixels to `target/spike/<name>`.
pub fn write_png(pixels: &[u8], size: [u32; 2], name: &str) {
    let path = workspace_root().join("target/spike").join(name);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    let file = std::io::BufWriter::new(std::fs::File::create(&path).expect("create png"));
    let mut enc = png::Encoder::new(file, size[0], size[1]);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .expect("png header")
        .write_image_data(pixels)
        .expect("png data");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_percentiles() {
        let s: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        let sum = Summary::of(&s);
        assert_eq!((sum.n, sum.min, sum.max), (100, 1.0, 100.0));
        assert_eq!(sum.p95, 95.0);
    }

    #[test]
    fn rusage_is_live() {
        assert!(peak_rss() > 0);
        let (_, span) = time(|| (0..2_000_000u64).map(|x| x.wrapping_mul(x)).sum::<u64>());
        assert!(span.wall > Duration::ZERO);
    }
}
