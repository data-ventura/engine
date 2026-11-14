//! Steps of a series (such as the hourly fields of a climate dataset) kept in RAM under the memory
//! budget, with the next few steps in the direction of travel fetched ahead.
//!
//! The view says where it is with [`Prefetch::go_to`] and reads steps with [`Prefetch::get`]. A
//! background task fetches the missing steps of the window `[step, step + dir, …, step +
//! depth·dir]`, nearest first, a few at a time: one read at a time leaves most of a connection's
//! throughput unused (ERA5 chunks: about 16 MB/s alone, 64 MB/s with 32 in flight), but each read
//! in flight slows the others, including the one the view is waiting for. A read already under way
//! for a step the view then asks for is shared, not repeated. When the view moves so that a step
//! being read leaves the window, its read is cancelled.
//!
//! Cached steps reserve [`Consumer::Cache`] memory. The cache makes room by dropping its own least
//! recently used steps outside the window, and it gives steps back when another consumer needs the
//! memory.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use tokio::sync::{OnceCell, watch};
use tokio::task::{AbortHandle, JoinSet};

use crate::Result;
use crate::budget::{Budget, Consumer, Evict, Reservation};

/// Fetches one step's values.
pub type Fetch =
    Arc<dyn Fn(u64) -> Pin<Box<dyn Future<Output = Result<Vec<f32>>> + Send>> + Send + Sync>;

/// A step cache with read-ahead. Dropping it stops the read-ahead task.
pub struct Prefetch {
    inner: Arc<Inner>,
    window: watch::Sender<Window>,
}

impl std::fmt::Debug for Prefetch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prefetch")
            .field("window", &*self.window.borrow())
            .field("stats", &self.stats())
            .finish()
    }
}

/// Where the view is, and which way it's moving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Window {
    step: u64,
    dir: i64,
}

/// Counts for reports and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stats {
    /// Reads of a step that completed.
    pub fetched: u64,
    /// [`Prefetch::get`] calls answered from RAM without waiting.
    pub hits: u64,
    /// Steps dropped to make room.
    pub evicted: u64,
}

struct Inner {
    budget: Budget,
    fetch: Fetch,
    steps: u64,
    depth: u64,
    slots: Mutex<HashMap<u64, Arc<Slot>>>,
    /// The window as of the last [`Prefetch::go_to`], for eviction.
    window: Mutex<Window>,
    clock: AtomicU64,
    fetched: AtomicU64,
    hits: AtomicU64,
    evicted: AtomicU64,
}

#[derive(Default)]
struct Slot {
    values: OnceCell<Arc<Vec<f32>>>,
    reservation: Mutex<Option<Reservation>>,
    used: AtomicU64,
}

impl Prefetch {
    /// A cache over steps `0..steps` that reads up to `depth` steps ahead, `parallel` at a time.
    /// Must be called inside a Tokio runtime: it spawns the read-ahead task.
    pub fn new(budget: Budget, steps: u64, depth: u64, parallel: usize, fetch: Fetch) -> Self {
        let inner = Arc::new(Inner {
            budget: budget.clone(),
            fetch,
            steps,
            depth,
            slots: Mutex::new(HashMap::new()),
            window: Mutex::new(Window { step: 0, dir: 1 }),
            clock: AtomicU64::new(0),
            fetched: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            evicted: AtomicU64::new(0),
        });
        budget.register(Arc::downgrade(&inner) as Weak<dyn Evict>);
        let (window, rx) = watch::channel(Window { step: 0, dir: 1 });
        tokio::spawn(read_ahead(Arc::downgrade(&inner), rx, parallel.max(1)));
        Self { inner, window }
    }

    /// Moves the view to `step`, travelling in the direction of `dir` (positive, negative, or 0
    /// for a jump with no direction, which reads ahead forward).
    pub fn go_to(&self, step: u64, dir: i64) {
        let w = Window {
            step,
            dir: if dir < 0 { -1 } else { 1 },
        };
        *self.inner.window.lock().expect("window") = w;
        self.window.send_replace(w);
    }

    /// The values of `step`, from RAM if they're there, or else from a read already under way, or
    /// else from a new read.
    pub async fn get(&self, step: u64) -> Result<Arc<Vec<f32>>> {
        if let Some(v) = self.cached(step) {
            self.inner.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(v);
        }
        self.inner.get(step).await
    }

    /// The values of `step` if they're in RAM now.
    pub fn cached(&self, step: u64) -> Option<Arc<Vec<f32>>> {
        let slot = self
            .inner
            .slots
            .lock()
            .expect("slots")
            .get(&step)
            .cloned()?;
        let v = slot.values.get()?.clone();
        slot.used.store(self.inner.tick(), Ordering::Relaxed);
        Some(v)
    }

    pub fn stats(&self) -> Stats {
        Stats {
            fetched: self.inner.fetched.load(Ordering::Relaxed),
            hits: self.inner.hits.load(Ordering::Relaxed),
            evicted: self.inner.evicted.load(Ordering::Relaxed),
        }
    }
}

impl Inner {
    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    fn window_steps(&self, w: Window) -> impl Iterator<Item = u64> + use<> {
        let (steps, depth) = (self.steps, self.depth);
        (0..=depth)
            .map_while(move |k| w.step.checked_add_signed(w.dir * k as i64))
            .take_while(move |&s| s < steps)
    }

    fn in_window(&self, step: u64) -> bool {
        let w = *self.window.lock().expect("window");
        self.window_steps(w).any(|s| s == step)
    }

    async fn get(&self, step: u64) -> Result<Arc<Vec<f32>>> {
        let slot = self
            .slots
            .lock()
            .expect("slots")
            .entry(step)
            .or_default()
            .clone();
        let values = slot
            .values
            .get_or_try_init(|| async {
                let v = (self.fetch)(step).await?;
                self.fetched.fetch_add(1, Ordering::Relaxed);
                Ok::<_, crate::Error>(Arc::new(v))
            })
            .await?
            .clone();
        slot.used.store(self.tick(), Ordering::Relaxed);
        let mut held = slot.reservation.lock().expect("reservation");
        if held.is_none() {
            let bytes = values.len() as u64 * 4;
            let r = self.budget.try_reserve(Consumer::Cache, bytes).or_else(|| {
                self.make_room(bytes, step);
                self.budget.try_reserve(Consumer::Cache, bytes)
            });
            match r {
                Some(r) => *held = Some(r),
                // No room: the caller gets the values, but they aren't kept.
                None => {
                    drop(held);
                    self.forget(step, &slot);
                }
            }
        }
        Ok(values)
    }

    /// Drops this cache's least recently used steps outside the window, other than `keep`, until
    /// `bytes` are free or none are left. Returns how many bytes it freed.
    fn make_room(&self, bytes: u64, keep: u64) -> u64 {
        let w = *self.window.lock().expect("window");
        let window: Vec<u64> = self.window_steps(w).collect();
        let mut candidates: Vec<(u64, u64, Arc<Slot>)> = self
            .slots
            .lock()
            .expect("slots")
            .iter()
            .filter(|(s, _)| **s != keep && !window.contains(s))
            .map(|(s, slot)| (slot.used.load(Ordering::Relaxed), *s, slot.clone()))
            .collect();
        candidates.sort_unstable_by_key(|c| c.0);
        let mut freed = 0;
        for (_, step, slot) in candidates {
            if freed >= bytes {
                break;
            }
            let r = slot.reservation.lock().expect("reservation").take();
            if let Some(r) = r {
                freed += r.bytes();
                self.forget(step, &slot);
                self.evicted.fetch_add(1, Ordering::Relaxed);
            }
        }
        freed
    }

    /// Removes `step`'s slot if it's still `slot`.
    fn forget(&self, step: u64, slot: &Arc<Slot>) {
        let mut slots = self.slots.lock().expect("slots");
        if slots.get(&step).is_some_and(|s| Arc::ptr_eq(s, slot)) {
            slots.remove(&step);
        }
    }
}

impl Evict for Inner {
    fn evict(&self, bytes: u64) -> u64 {
        let freed = self.make_room(bytes, u64::MAX);
        if freed >= bytes {
            return freed;
        }
        // Another consumer needs the memory more than read-ahead does: give up the window too.
        let mut rest: Vec<(u64, Arc<Slot>)> = self
            .slots
            .lock()
            .expect("slots")
            .iter()
            .map(|(s, slot)| (*s, slot.clone()))
            .collect();
        rest.sort_unstable_by_key(|(_, slot)| slot.used.load(Ordering::Relaxed));
        let mut freed = freed;
        for (step, slot) in rest {
            if freed >= bytes {
                break;
            }
            let r = slot.reservation.lock().expect("reservation").take();
            if let Some(r) = r {
                freed += r.bytes();
                self.forget(step, &slot);
                self.evicted.fetch_add(1, Ordering::Relaxed);
            }
        }
        freed
    }
}

/// Fetches the window's missing steps nearest first, up to `parallel` at once, until the
/// [`Prefetch`] is dropped. A read whose step leaves the window is cancelled.
async fn read_ahead(inner: Weak<Inner>, mut rx: watch::Receiver<Window>, parallel: usize) {
    let mut reads = JoinSet::new();
    let mut running: HashMap<u64, AbortHandle> = HashMap::new();
    // Steps whose read failed at this window position: left to the view's own read, so a failing
    // store isn't retried in a loop.
    let mut failed = HashSet::new();
    let mut at = None;
    loop {
        let w = *rx.borrow_and_update();
        if at != Some(w) {
            at = Some(w);
            failed.clear();
        }
        {
            let Some(this) = inner.upgrade() else { return };
            running.retain(|&s, read| {
                let keep = this.in_window(s);
                if !keep {
                    read.abort();
                }
                keep
            });
            let missing: Vec<u64> = {
                let slots = this.slots.lock().expect("slots");
                this.window_steps(w)
                    .filter(|s| !running.contains_key(s) && !failed.contains(s))
                    .filter(|s| slots.get(s).is_none_or(|slot| slot.values.get().is_none()))
                    .take(parallel.saturating_sub(running.len()))
                    .collect()
            };
            for step in missing {
                let this = this.clone();
                let read = reads.spawn(async move { (step, this.get(step).await.is_ok()) });
                running.insert(step, read);
            }
        }
        tokio::select! {
            Some(done) = reads.join_next(), if !reads.is_empty() => {
                // A cancelled read was already taken out of `running`.
                if let Ok((step, ok)) = done {
                    running.remove(&step);
                    if !ok {
                        failed.insert(step);
                    }
                }
            }
            changed = rx.changed() => {
                if changed.is_err() {
                    reads.abort_all();
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// Steps of `len` floats whose values are the step number, taking `ms` each, one at a time
    /// like a shared connection. Records which steps were read.
    fn slow_fetch(len: usize, ms: u64, log: Arc<Mutex<Vec<u64>>>) -> Fetch {
        let line = Arc::new(tokio::sync::Mutex::new(()));
        Arc::new(move |step| {
            let (log, line) = (log.clone(), line.clone());
            Box::pin(async move {
                let _one_at_a_time = line.lock().await;
                tokio::time::sleep(Duration::from_millis(ms)).await;
                log.lock().unwrap().push(step);
                Ok(vec![step as f32; len])
            })
        })
    }

    #[tokio::test(start_paused = true)]
    async fn stepping_forward_hits_ram() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let p = Prefetch::new(
            Budget::new(1 << 20),
            100,
            3,
            1,
            slow_fetch(10, 150, log.clone()),
        );
        p.go_to(10, 1);
        assert_eq!(p.get(10).await.unwrap()[0], 10.0);
        // The user steps forward every 200 ms; after the first step everything is already there.
        for s in 11..20 {
            tokio::time::sleep(Duration::from_millis(200)).await;
            assert!(p.cached(s).is_some(), "step {s} read ahead");
            p.go_to(s, 1);
            assert_eq!(p.get(s).await.unwrap()[0], s as f32);
        }
        assert_eq!(p.stats().hits, 9);
        // Each step was read once, in order.
        let read = log.lock().unwrap().clone();
        assert_eq!(read[..10], (10..20).collect::<Vec<_>>()[..]);
        assert!(read.windows(2).all(|w| w[0] < w[1]), "{read:?}");
    }

    #[tokio::test(start_paused = true)]
    async fn a_jump_cancels_read_ahead_and_shares_the_read() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let p = Prefetch::new(
            Budget::new(1 << 20),
            100,
            3,
            1,
            slow_fetch(10, 150, log.clone()),
        );
        p.go_to(10, 1);
        p.get(10).await.unwrap();
        // Read-ahead is now on step 11; jump backward to 50 before it finishes.
        tokio::time::sleep(Duration::from_millis(50)).await;
        p.go_to(50, -1);
        let t = tokio::time::Instant::now();
        assert_eq!(p.get(50).await.unwrap()[0], 50.0);
        // The jump didn't wait behind the cancelled read, and its read wasn't repeated.
        assert_eq!(t.elapsed(), Duration::from_millis(150));
        tokio::time::sleep(Duration::from_secs(2)).await;
        let read = log.lock().unwrap().clone();
        assert_eq!(
            read,
            vec![10, 50, 49, 48, 47],
            "11 cancelled, then backward"
        );
    }

    /// Steps that take `ms` each however many run at once, like a connection limited by latency,
    /// not throughput. Records the most reads in flight.
    fn latency_fetch(ms: u64, most: Arc<Mutex<(usize, usize)>>) -> Fetch {
        Arc::new(move |step| {
            let most = most.clone();
            Box::pin(async move {
                {
                    let mut m = most.lock().unwrap();
                    m.0 += 1;
                    m.1 = m.1.max(m.0);
                }
                tokio::time::sleep(Duration::from_millis(ms)).await;
                most.lock().unwrap().0 -= 1;
                Ok(vec![step as f32; 10])
            })
        })
    }

    #[tokio::test(start_paused = true)]
    async fn parallel_reads_keep_up_when_one_cant() {
        // Steps every 100 ms, reads take 250 ms.
        let mut hits = Vec::new();
        for parallel in [1, 3] {
            let most = Arc::new(Mutex::new((0, 0)));
            let p = Prefetch::new(
                Budget::new(1 << 20),
                100,
                4,
                parallel,
                latency_fetch(250, most.clone()),
            );
            p.go_to(0, 1);
            p.get(0).await.unwrap();
            for s in 1..30 {
                tokio::time::sleep(Duration::from_millis(100)).await;
                p.go_to(s, 1);
                p.get(s).await.unwrap();
            }
            // The view's own read can add one to read-ahead's.
            assert!(most.lock().unwrap().1 <= parallel + 1);
            hits.push(p.stats().hits);
        }
        assert!(hits[0] < 20, "one at a time falls behind: {hits:?}");
        assert_eq!(hits[1], 29, "three at a time keep up: {hits:?}");
    }

    #[tokio::test(start_paused = true)]
    async fn the_cache_stays_inside_the_budget() {
        // Room for 5 steps of 10 floats.
        let budget = Budget::new(200);
        let log = Arc::new(Mutex::new(Vec::new()));
        let p = Prefetch::new(budget.clone(), 100, 2, 2, slow_fetch(10, 10, log.clone()));
        for s in 0..20 {
            p.go_to(s, 1);
            p.get(s).await.unwrap();
            assert!(budget.used() <= 200);
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
        assert!(p.stats().evicted > 0);
        // The newest steps are kept; the oldest went first.
        assert!(p.cached(19).is_some() && p.cached(0).is_none());
        // Another consumer takes everything back.
        let r = budget.reserve(Consumer::Arrays, 200).unwrap();
        assert_eq!(budget.used_by(Consumer::Cache), 0);
        drop(r);
        drop(p);
        assert_eq!(budget.used(), 0);
    }
}
