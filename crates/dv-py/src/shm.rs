//! POSIX shared-memory segments, and a registry that releases them after a crash.
//!
//! The host names every segment, including the ones a worker creates for its results, and owns
//! their lifetime. Before a name is handed out it's recorded in this process's registry file; once
//! the segment is unlinked, that's recorded too. The process holds an exclusive `flock` on its
//! registry for as long as it lives, so on startup [`Registry::open`] can tell a dead process's
//! registry from a live one (a lock it can take) without trusting PIDs, and unlinks whatever the
//! dead process left behind. macOS can't list POSIX shared memory, so without the registry those
//! segments would hold memory until reboot.

use std::collections::BTreeSet;
use std::ffi::CString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// A mapped segment. Unmapped on drop; unlinking is separate (see [`Registry::unlink`]).
#[derive(Debug)]
pub struct Mapping {
    ptr: NonNull<u8>,
    len: usize,
    mapped: usize,
}

// SAFETY: the mapping is plain shared memory with no thread affinity; access is through `&self`
// (read) or `&mut self` (write), so Rust's aliasing rules still hold within this process.
unsafe impl Send for Mapping {}
// SAFETY: as above.
unsafe impl Sync for Mapping {}

impl Mapping {
    /// Creates a segment of `len` bytes named `name` (which must not exist) and maps it.
    pub fn create(name: &str, len: usize) -> io::Result<Self> {
        let fd = shm_open(name, libc::O_CREAT | libc::O_EXCL | libc::O_RDWR)?;
        let mapped = len.max(1);
        // SAFETY: fd is a shared-memory object we just created and own.
        if unsafe { libc::ftruncate(fd.as_raw_fd(), mapped as libc::off_t) } != 0 {
            let err = io::Error::last_os_error();
            let _ = shm_unlink(name);
            return Err(err);
        }
        Self::map(&fd, len, mapped, libc::PROT_READ | libc::PROT_WRITE)
    }

    /// Maps the first `len` bytes of an existing segment, read-only.
    pub fn open(name: &str, len: usize) -> io::Result<Self> {
        let fd = shm_open(name, libc::O_RDONLY)?;
        // SAFETY: fstat writes into the struct we pass.
        let size = unsafe {
            let mut st = std::mem::zeroed::<libc::stat>();
            if libc::fstat(fd.as_raw_fd(), &mut st) != 0 {
                return Err(io::Error::last_os_error());
            }
            st.st_size as usize
        };
        if size < len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("segment {name} holds {size} bytes, not the {len} announced"),
            ));
        }
        Self::map(&fd, len, len.max(1), libc::PROT_READ)
    }

    fn map(fd: &File, len: usize, mapped: usize, prot: libc::c_int) -> io::Result<Self> {
        // SAFETY: a shared mapping of an object we hold open, of at most its size.
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapped,
                prot,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            ptr: NonNull::new(addr.cast()).expect("mmap returned null"),
            len,
            mapped,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the mapping is at least `len` bytes and lives as long as `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// The bytes, writable. Only for segments made with [`Mapping::create`].
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above; `create` maps it writable, and `&mut self` makes this the only view.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: unmaps exactly what `map` mapped; nothing borrows it past `self`.
        unsafe {
            libc::munmap(self.ptr.as_ptr().cast(), self.mapped);
        }
    }
}

fn shm_open(name: &str, flags: libc::c_int) -> io::Result<File> {
    let c = CString::new(name).map_err(|_| io::Error::other("NUL in segment name"))?;
    // SAFETY: c is a valid C string; the returned descriptor is owned by the File.
    let fd = unsafe { libc::shm_open(c.as_ptr(), flags, 0o600 as libc::c_uint) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fd is a fresh descriptor nothing else owns.
    Ok(unsafe { std::os::fd::FromRawFd::from_raw_fd(fd) })
}

fn shm_unlink(name: &str) -> io::Result<()> {
    let c = CString::new(name).map_err(|_| io::Error::other("NUL in segment name"))?;
    // SAFETY: c is a valid C string.
    if unsafe { libc::shm_unlink(c.as_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// The segment names this process has handed out, recorded in a locked file so a later process
/// can release them if this one dies.
#[derive(Debug)]
pub struct Registry {
    file: Mutex<File>,
    path: PathBuf,
    pid: u32,
}

impl Registry {
    /// Opens this process's registry in `dir`, first releasing the segments of any process whose
    /// registry there is no longer locked. Returns the registry and how many segments it released.
    pub fn open(dir: &Path) -> io::Result<(Self, usize)> {
        fs::create_dir_all(dir)?;
        let released = sweep(dir)?;
        let pid = std::process::id();
        let path = dir.join(format!("{pid}.segments"));
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)?;
        lock(&file, false)?;
        Ok((
            Self {
                file: Mutex::new(file),
                path,
                pid,
            },
            released,
        ))
    }

    /// A fresh segment name, recorded before it's used. Names fit macOS's 31-byte limit, and are
    /// unique within the process even with several registries (as in tests).
    pub fn name(&self) -> io::Result<String> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("/dv{}.{n}", self.pid);
        // A segment of this name can only be left over from a dead process with the same PID.
        let _ = shm_unlink(&name);
        self.record('+', &name)?;
        Ok(name)
    }

    /// Unlinks `name` (if it exists) and records that it's gone.
    pub fn unlink(&self, name: &str) {
        let _ = shm_unlink(name);
        let _ = self.record('-', name);
    }

    fn record(&self, op: char, name: &str) -> io::Result<()> {
        let mut f = self.file.lock().expect("registry lock");
        writeln!(f, "{op}{name}")
    }
}

impl Drop for Registry {
    fn drop(&mut self) {
        // Everything should be unlinked by now; release stragglers and remove the file.
        if let Ok(names) = live_names(&self.path) {
            for n in names {
                let _ = shm_unlink(&n);
            }
        }
        let _ = fs::remove_file(&self.path);
    }
}

/// Releases the segments of every registry in `dir` whose owner is gone.
fn sweep(dir: &Path) -> io::Result<usize> {
    let mut released = 0;
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_none_or(|e| e != "segments") {
            continue;
        }
        let Ok(file) = OpenOptions::new().read(true).write(true).open(&path) else {
            continue;
        };
        if lock(&file, true).is_err() {
            continue; // its owner is alive
        }
        for name in live_names(&path)? {
            if shm_unlink(&name).is_ok() {
                released += 1;
            }
        }
        fs::remove_file(&path)?;
    }
    Ok(released)
}

fn live_names(path: &Path) -> io::Result<BTreeSet<String>> {
    let mut live = BTreeSet::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        match line.split_at_checked(1) {
            Some(("+", n)) => live.insert(n.to_string()),
            Some(("-", n)) => live.remove(n),
            _ => false,
        };
    }
    Ok(live)
}

/// An exclusive advisory lock, held until the file is closed.
fn lock(file: &File, nonblocking: bool) -> io::Result<()> {
    let op = libc::LOCK_EX | if nonblocking { libc::LOCK_NB } else { 0 };
    // SAFETY: flock on a descriptor we own.
    if unsafe { libc::flock(file.as_raw_fd(), op) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("dv-py-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn create_write_open_read() {
        let d = dir("rw");
        let (reg, _) = Registry::open(&d).unwrap();
        let name = reg.name().unwrap();
        let mut m = Mapping::create(&name, 5).unwrap();
        m.as_mut_slice().copy_from_slice(b"hello");
        let r = Mapping::open(&name, 5).unwrap();
        assert_eq!(r.as_slice(), b"hello");
        assert!(
            Mapping::open(&name, 1 << 20).is_err(),
            "longer than the segment"
        );
        reg.unlink(&name);
        assert!(Mapping::open(&name, 5).is_err(), "unlinked");
        // The mappings outlive the name.
        assert_eq!(r.as_slice(), b"hello");
    }

    #[test]
    fn a_dead_owners_segments_are_released() {
        let d = dir("sweep");
        fs::create_dir_all(&d).unwrap();
        // A registry no process holds a lock on, as a crashed process leaves it.
        let leaked = format!("/dvtest{}.leak", std::process::id());
        let gone = format!("/dvtest{}.gone", std::process::id());
        drop(Mapping::create(&leaked, 64).unwrap());
        fs::write(
            d.join("1.segments"),
            format!("+{leaked}\n+{gone}\n-{gone}\n"),
        )
        .unwrap();
        let (reg, released) = Registry::open(&d).unwrap();
        assert_eq!(released, 1);
        assert!(Mapping::open(&leaked, 64).is_err());
        assert!(!d.join("1.segments").exists());
        // A live registry (ours) is left alone by a second sweep.
        let name = reg.name().unwrap();
        drop(Mapping::create(&name, 8).unwrap());
        assert_eq!(sweep(&d).unwrap(), 0);
        assert!(Mapping::open(&name, 8).is_ok());
        drop(reg);
        assert!(
            Mapping::open(&name, 8).is_err(),
            "dropping the registry releases stragglers"
        );
    }
}
