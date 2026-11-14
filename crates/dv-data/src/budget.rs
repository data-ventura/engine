//! One memory budget shared by every consumer: query execution, decoded array chunks, the RAM
//! cache, GPU buffers and Python shared memory.
//!
//! Consumers reserve memory before they allocate and hold a [`Reservation`] for as long as the
//! memory is in use; dropping it gives the bytes back. When a reservation doesn't fit, the budget
//! first asks registered [`Evict`]ors (caches) to free enough, then refuses. A refused consumer
//! degrades on its own terms: a coarser level of detail, or failing that one request with a clear
//! message. The budget never allocates anything itself.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// Who holds memory, for accounting and messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Consumer {
    Query,
    Arrays,
    Cache,
    Gpu,
    Python,
}

impl Consumer {
    pub const ALL: [Consumer; 5] = [
        Consumer::Query,
        Consumer::Arrays,
        Consumer::Cache,
        Consumer::Gpu,
        Consumer::Python,
    ];

    fn slot(self) -> usize {
        self as usize
    }
}

/// Something that can free memory it holds under the budget, such as a cache.
pub trait Evict: Send + Sync {
    /// Frees at least `bytes` if it can, by dropping reservations, and returns how many it freed.
    fn evict(&self, bytes: u64) -> u64;
}

/// A reservation the budget couldn't make room for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "{consumer:?} needs {} MB, but only {} MB of the {} MB memory budget is free",
    wanted / 1_000_000,
    free / 1_000_000,
    limit / 1_000_000
)]
pub struct Refused {
    pub consumer: Consumer,
    pub wanted: u64,
    pub free: u64,
    pub limit: u64,
}

/// The shared memory budget. Cheap to clone; clones share one budget.
#[derive(Clone)]
pub struct Budget {
    inner: Arc<Inner>,
}

struct Inner {
    limit: AtomicU64,
    used: AtomicU64,
    peak: AtomicU64,
    by_consumer: [AtomicU64; 5],
    evictors: Mutex<Vec<Weak<dyn Evict>>>,
}

impl fmt::Debug for Budget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Budget")
            .field("limit", &self.limit())
            .field("used", &self.used())
            .field("peak", &self.peak())
            .finish()
    }
}

impl Budget {
    /// A budget of `limit` bytes.
    pub fn new(limit: u64) -> Self {
        Self {
            inner: Arc::new(Inner {
                limit: AtomicU64::new(limit),
                used: AtomicU64::new(0),
                peak: AtomicU64::new(0),
                by_consumer: Default::default(),
                evictors: Mutex::new(Vec::new()),
            }),
        }
    }

    /// The default budget: 60% of physical memory (SPEC §4.9).
    pub fn for_this_machine() -> Self {
        Self::new(physical_memory() / 10 * 6)
    }

    pub fn limit(&self) -> u64 {
        self.inner.limit.load(Ordering::Relaxed)
    }

    /// Changes the limit, for example when the system reports memory pressure. Memory already
    /// reserved stays reserved; caches are asked to shed anything over the new limit.
    pub fn set_limit(&self, limit: u64) {
        self.inner.limit.store(limit, Ordering::Relaxed);
        let over = self.used().saturating_sub(limit);
        if over > 0 {
            self.evict(over);
        }
    }

    /// Bytes reserved now.
    pub fn used(&self) -> u64 {
        self.inner.used.load(Ordering::Relaxed)
    }

    /// Bytes reserved now by one consumer.
    pub fn used_by(&self, consumer: Consumer) -> u64 {
        self.inner.by_consumer[consumer.slot()].load(Ordering::Relaxed)
    }

    /// The most bytes ever reserved at once.
    pub fn peak(&self) -> u64 {
        self.inner.peak.load(Ordering::Relaxed)
    }

    /// Registers a cache that can free memory when a reservation doesn't fit. The budget holds it
    /// weakly, so dropping the cache unregisters it.
    pub fn register(&self, evictor: Weak<dyn Evict>) {
        let mut list = self.inner.evictors.lock().expect("evictors");
        list.retain(|e| e.strong_count() > 0);
        list.push(evictor);
    }

    /// Reserves `bytes` for `consumer`, evicting from caches if needed.
    pub fn reserve(&self, consumer: Consumer, bytes: u64) -> Result<Reservation, Refused> {
        if self.try_add(consumer, bytes) {
            return Ok(self.reservation(consumer, bytes));
        }
        let short = (self.used() + bytes).saturating_sub(self.limit());
        self.evict(short);
        if self.try_add(consumer, bytes) {
            return Ok(self.reservation(consumer, bytes));
        }
        Err(Refused {
            consumer,
            wanted: bytes,
            free: self.limit().saturating_sub(self.used()),
            limit: self.limit(),
        })
    }

    /// Reserves `bytes` only if they fit without evicting anything. Caches use this, so a cache
    /// never pushes out another cache's entries.
    pub fn try_reserve(&self, consumer: Consumer, bytes: u64) -> Option<Reservation> {
        self.try_add(consumer, bytes)
            .then(|| self.reservation(consumer, bytes))
    }

    fn reservation(&self, consumer: Consumer, bytes: u64) -> Reservation {
        Reservation {
            budget: self.clone(),
            consumer,
            bytes,
        }
    }

    fn try_add(&self, consumer: Consumer, bytes: u64) -> bool {
        let limit = self.limit();
        let added = self
            .inner
            .used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                (used + bytes <= limit).then_some(used + bytes)
            });
        match added {
            Ok(before) => {
                self.inner.peak.fetch_max(before + bytes, Ordering::Relaxed);
                self.inner.by_consumer[consumer.slot()].fetch_add(bytes, Ordering::Relaxed);
                true
            }
            Err(_) => false,
        }
    }

    fn release(&self, consumer: Consumer, bytes: u64) {
        self.inner.used.fetch_sub(bytes, Ordering::AcqRel);
        self.inner.by_consumer[consumer.slot()].fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Asks caches to free `bytes`, in registration order. Returns how many they freed.
    fn evict(&self, bytes: u64) -> u64 {
        // Evictors drop reservations, which only touch atomics, so calling them without the lock
        // can't deadlock.
        let list: Vec<_> = self
            .inner
            .evictors
            .lock()
            .expect("evictors")
            .iter()
            .filter_map(Weak::upgrade)
            .collect();
        let mut freed = 0;
        for e in list {
            if freed >= bytes {
                break;
            }
            freed += e.evict(bytes - freed);
        }
        freed
    }
}

/// Memory reserved under a [`Budget`]. Dropping it gives the bytes back.
#[derive(Debug)]
pub struct Reservation {
    budget: Budget,
    consumer: Consumer,
    bytes: u64,
}

impl Reservation {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn consumer(&self) -> Consumer {
        self.consumer
    }

    /// Reserves `more` bytes on top of this reservation, evicting from caches if needed.
    pub fn grow(&mut self, more: u64) -> Result<(), Refused> {
        let extra = self.budget.reserve(self.consumer, more)?;
        self.bytes += extra.bytes;
        std::mem::forget(extra);
        Ok(())
    }

    /// Gives back all but `bytes` of this reservation.
    pub fn shrink_to(&mut self, bytes: u64) {
        let back = self.bytes.saturating_sub(bytes);
        self.budget.release(self.consumer, back);
        self.bytes -= back;
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.release(self.consumer, self.bytes);
    }
}

/// Physical memory in bytes.
pub fn physical_memory() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut bytes: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: hw.memsize is a u64, and we pass a buffer of exactly that size.
        let rc = unsafe {
            libc::sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&raw mut bytes).cast(),
                &raw mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 {
            return bytes;
        }
    }
    #[cfg(unix)]
    {
        // SAFETY: sysconf has no preconditions.
        let (pages, size) = unsafe {
            (
                libc::sysconf(libc::_SC_PHYS_PAGES),
                libc::sysconf(libc::_SC_PAGESIZE),
            )
        };
        if pages > 0 && size > 0 {
            return pages as u64 * size as u64;
        }
    }
    8 << 30
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cache holding reservations it gives up oldest first.
    struct TestCache(Mutex<Vec<Reservation>>);

    impl Evict for TestCache {
        fn evict(&self, bytes: u64) -> u64 {
            let mut held = self.0.lock().unwrap();
            let mut freed = 0;
            while freed < bytes && !held.is_empty() {
                freed += held.remove(0).bytes();
            }
            freed
        }
    }

    #[test]
    fn reservations_add_up_and_are_returned() {
        let b = Budget::new(100);
        let r1 = b.reserve(Consumer::Arrays, 60).unwrap();
        let mut r2 = b.reserve(Consumer::Gpu, 30).unwrap();
        assert_eq!((b.used(), b.used_by(Consumer::Gpu)), (90, 30));
        let refused = b.reserve(Consumer::Query, 20).unwrap_err();
        assert_eq!((refused.wanted, refused.free, refused.limit), (20, 10, 100));
        assert!(r2.grow(20).is_err());
        r2.grow(10).unwrap();
        assert_eq!(b.used(), 100);
        r2.shrink_to(5);
        assert_eq!(b.used(), 65);
        drop((r1, r2));
        assert_eq!((b.used(), b.peak()), (0, 100));
    }

    #[test]
    fn refused_reservations_evict_caches_first() {
        let b = Budget::new(100);
        let cache = Arc::new(TestCache(Mutex::new(Vec::new())));
        b.register(Arc::downgrade(&cache) as Weak<dyn Evict>);
        for _ in 0..8 {
            let r = b.try_reserve(Consumer::Cache, 10).unwrap();
            cache.0.lock().unwrap().push(r);
        }
        // Caches don't evict each other.
        assert!(b.try_reserve(Consumer::Cache, 30).is_none());
        // Other consumers do: 30 bytes need one 10-byte entry evicted.
        let r = b.reserve(Consumer::Arrays, 30).unwrap();
        assert_eq!(cache.0.lock().unwrap().len(), 7);
        assert_eq!(b.used(), 100);
        // A lower limit sheds cache entries down to it.
        b.set_limit(50);
        assert!(b.used() <= 50, "{}", b.used());
        assert_eq!(b.used_by(Consumer::Arrays), 30);
        drop(r);
        // Anything bigger than the limit is refused even with the cache empty.
        assert!(b.reserve(Consumer::Arrays, 60).is_err());
    }

    #[test]
    fn concurrent_reservations_never_exceed_the_limit() {
        let b = Budget::new(1000);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..10_000 {
                        if let Ok(r) = b.reserve(Consumer::Arrays, 7) {
                            assert!(b.used() <= 1000);
                            drop(r);
                        }
                    }
                });
            }
        });
        assert_eq!(b.used(), 0);
        assert!(b.peak() <= 1000);
    }

    #[test]
    fn this_machine_has_memory() {
        assert!(physical_memory() >= 1 << 30);
    }
}
