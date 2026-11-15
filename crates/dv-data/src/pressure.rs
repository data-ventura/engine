//! System memory pressure shrinks the memory budget (SPEC §4.9).
//!
//! On macOS, [`PressureWatcher`] listens for the kernel's memory-pressure notifications through a
//! libdispatch source, the same mechanism AppKit apps use. At *warning* the budget drops to a
//! fraction of its normal limit, and further at *critical*; when pressure returns to *normal* the
//! limit is restored. Shrinking the limit asks caches to shed whatever is over it, and consumers
//! whose reservations are then refused draw at coarser detail and say so, as under any tight
//! budget. Elsewhere the watcher does nothing, and [`PressureWatcher::apply`] can drive it by hand.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use crate::budget::Budget;

/// The system's memory-pressure level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Level {
    Normal,
    Warning,
    Critical,
}

/// The budget limit at each level, as a fraction of the normal limit.
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    pub warning: f64,
    pub critical: f64,
}

impl Default for Policy {
    /// Half at warning and a quarter at critical: at warning macOS is compressing memory, and
    /// giving back half our caches is cheap (they refill from disk or the network); at critical
    /// it's about to kill processes, and only what's on screen should stay.
    fn default() -> Self {
        Self {
            warning: 0.5,
            critical: 0.25,
        }
    }
}

struct State {
    budget: Budget,
    normal: u64,
    policy: Policy,
    level: AtomicU8,
    log: Mutex<Vec<Level>>,
}

impl State {
    fn apply(&self, level: Level) {
        self.level.store(level as u8, Ordering::Release);
        self.log.lock().expect("log").push(level);
        let f = match level {
            Level::Normal => 1.0,
            Level::Warning => self.policy.warning,
            Level::Critical => self.policy.critical,
        };
        self.budget.set_limit((self.normal as f64 * f) as u64);
    }
}

/// Keeps a budget's limit in step with system memory pressure while it lives.
pub struct PressureWatcher {
    state: Arc<State>,
    #[cfg(target_os = "macos")]
    source: Option<macos::Source>,
}

impl std::fmt::Debug for PressureWatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PressureWatcher")
            .field("level", &self.level())
            .field("normal_limit", &self.state.normal)
            .finish_non_exhaustive()
    }
}

impl PressureWatcher {
    /// Watches system memory pressure, taking `budget`'s current limit as its normal limit.
    pub fn start(budget: Budget, policy: Policy) -> Self {
        let state = Arc::new(State {
            normal: budget.limit(),
            budget,
            policy,
            level: AtomicU8::new(Level::Normal as u8),
            log: Mutex::new(Vec::new()),
        });
        #[cfg(target_os = "macos")]
        {
            let source = macos::Source::new(state.clone());
            if let Some(level) = macos::current_level()
                && level != Level::Normal
            {
                state.apply(level);
            }
            Self { state, source }
        }
        #[cfg(not(target_os = "macos"))]
        Self { state }
    }

    /// Whether the system's notifications are connected (macOS, and the source was created).
    pub fn is_listening(&self) -> bool {
        #[cfg(target_os = "macos")]
        return self.source.is_some();
        #[cfg(not(target_os = "macos"))]
        false
    }

    pub fn level(&self) -> Level {
        match self.state.level.load(Ordering::Acquire) {
            1 => Level::Warning,
            2 => Level::Critical,
            _ => Level::Normal,
        }
    }

    /// Every level applied so far, in order.
    pub fn history(&self) -> Vec<Level> {
        self.state.log.lock().expect("log").clone()
    }

    /// Applies `level` as if the system had reported it. For tests and the spike.
    pub fn apply(&self, level: Level) {
        self.state.apply(level);
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::c_void;
    use std::sync::Arc;

    use super::{Level, State};

    #[repr(C)]
    struct SourceType {
        _private: [u8; 0],
    }

    // libdispatch, part of libSystem.
    unsafe extern "C" {
        static _dispatch_source_type_memorypressure: SourceType;
        fn dispatch_get_global_queue(identifier: isize, flags: usize) -> *mut c_void;
        fn dispatch_source_create(
            kind: *const SourceType,
            handle: usize,
            mask: usize,
            queue: *mut c_void,
        ) -> *mut c_void;
        fn dispatch_set_context(object: *mut c_void, context: *mut c_void);
        fn dispatch_source_set_event_handler_f(
            source: *mut c_void,
            handler: extern "C" fn(*mut c_void),
        );
        fn dispatch_source_set_cancel_handler_f(
            source: *mut c_void,
            handler: extern "C" fn(*mut c_void),
        );
        fn dispatch_source_get_data(source: *mut c_void) -> usize;
        fn dispatch_resume(object: *mut c_void);
        fn dispatch_source_cancel(source: *mut c_void);
        fn dispatch_release(object: *mut c_void);
    }

    const NORMAL: usize = 0x1;
    const WARN: usize = 0x2;
    const CRITICAL: usize = 0x4;
    const QOS_CLASS_UTILITY: isize = 0x11;

    struct Context {
        source: *mut c_void,
        state: Arc<State>,
    }

    /// A memory-pressure dispatch source. Cancelled and released on drop; its context is freed by
    /// the cancel handler, after the last event handler has run.
    pub(super) struct Source(*mut c_void);

    // SAFETY: dispatch objects are thread-safe, reference-counted handles.
    unsafe impl Send for Source {}
    // SAFETY: as above; `Source` exposes no methods that touch the object.
    unsafe impl Sync for Source {}

    impl Source {
        pub(super) fn new(state: Arc<State>) -> Option<Self> {
            // SAFETY: the calls follow libdispatch's documented protocol: create the source on a
            // global queue, set its context and handlers before resuming it, and free the
            // context only in the cancel handler.
            unsafe {
                let queue = dispatch_get_global_queue(QOS_CLASS_UTILITY, 0);
                let source = dispatch_source_create(
                    &raw const _dispatch_source_type_memorypressure,
                    0,
                    NORMAL | WARN | CRITICAL,
                    queue,
                );
                if source.is_null() {
                    return None;
                }
                let ctx = Box::into_raw(Box::new(Context { source, state }));
                dispatch_set_context(source, ctx.cast());
                dispatch_source_set_event_handler_f(source, on_event);
                dispatch_source_set_cancel_handler_f(source, on_cancel);
                dispatch_resume(source);
                Some(Self(source))
            }
        }
    }

    impl Drop for Source {
        fn drop(&mut self) {
            // SAFETY: the source is valid until this release, which drops our reference;
            // libdispatch keeps it alive until the cancel handler has run.
            unsafe {
                dispatch_source_cancel(self.0);
                dispatch_release(self.0);
            }
        }
    }

    extern "C" fn on_event(ctx: *mut c_void) {
        // SAFETY: ctx is the Context set in `new`, freed only by `on_cancel`, which libdispatch
        // never runs concurrently with an event handler.
        let ctx = unsafe { &*ctx.cast::<Context>() };
        // SAFETY: the source is alive while its handlers run.
        let data = unsafe { dispatch_source_get_data(ctx.source) };
        let level = if data & CRITICAL != 0 {
            Level::Critical
        } else if data & WARN != 0 {
            Level::Warning
        } else {
            Level::Normal
        };
        ctx.state.apply(level);
    }

    extern "C" fn on_cancel(ctx: *mut c_void) {
        // SAFETY: the Box made in `new`; the cancel handler runs once, after all event handlers.
        drop(unsafe { Box::from_raw(ctx.cast::<Context>()) });
    }

    /// The level now, from `kern.memorystatus_vm_pressure_level` (1 normal, 2 warning, 4 critical).
    pub(super) fn current_level() -> Option<Level> {
        let mut v: u32 = 0;
        let mut len = std::mem::size_of::<u32>();
        // SAFETY: the sysctl is a 32-bit integer and we pass a buffer of exactly that size.
        let rc = unsafe {
            libc::sysctlbyname(
                c"kern.memorystatus_vm_pressure_level".as_ptr(),
                (&raw mut v).cast(),
                &raw mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0).then_some(match v {
            4 => Level::Critical,
            2 => Level::Warning,
            _ => Level::Normal,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Weak;

    use super::*;
    use crate::budget::{Consumer, Evict};

    /// A cache holding reservations that it drops when asked.
    struct Cache(Mutex<Vec<crate::budget::Reservation>>);

    impl Evict for Cache {
        fn evict(&self, bytes: u64) -> u64 {
            let mut held = self.0.lock().unwrap();
            let mut freed = 0;
            while freed < bytes {
                match held.pop() {
                    Some(r) => freed += r.bytes(),
                    None => break,
                }
            }
            freed
        }
    }

    #[test]
    fn pressure_shrinks_the_budget_and_caches_shed() {
        let budget = Budget::new(1000);
        let cache = Arc::new(Cache(Mutex::new(Vec::new())));
        budget.register(Arc::downgrade(&cache) as Weak<dyn Evict>);
        for _ in 0..9 {
            let r = budget.try_reserve(Consumer::Cache, 100).unwrap();
            cache.0.lock().unwrap().push(r);
        }
        let w = PressureWatcher::start(budget.clone(), Policy::default());
        #[cfg(target_os = "macos")]
        assert!(w.is_listening(), "the dispatch source was created");

        w.apply(Level::Warning);
        assert_eq!(budget.limit(), 500);
        assert!(budget.used() <= 500, "the cache shed down to the new limit");
        w.apply(Level::Critical);
        assert_eq!(budget.limit(), 250);
        assert!(budget.used() <= 250);
        // A request that no longer fits is refused, so its view degrades and says so.
        assert!(budget.reserve(Consumer::Arrays, 300).is_err());
        w.apply(Level::Normal);
        assert_eq!(budget.limit(), 1000);
        assert!(budget.reserve(Consumer::Arrays, 300).is_ok());
        assert_eq!(
            w.history(),
            [Level::Warning, Level::Critical, Level::Normal]
        );
        // Dropping the watcher cancels the source cleanly.
        drop(w);
    }
}
