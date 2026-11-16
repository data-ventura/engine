//! DataFusion's memory pool, backed by the engine's shared memory budget.
//!
//! Every operator that buffers (hash joins, aggregates, sorts) reserves through this pool, so
//! query memory counts against the same budget as array chunks, caches and GPU buffers. Queries
//! may also have a limit of their own, below the budget, so one large query can't push every
//! cache out. When the pool refuses, the operator spills if it can and otherwise fails that one
//! query with the pool's message.
//!
//! Operators that can spill share what operators that can't have left, in equal parts, like
//! DataFusion's `FairSpillPool`. Without that, the first partition of a sort would take the whole
//! limit and the others would fail before they had anything to spill. Unlike `FairSpillPool`, a
//! quarter of the limit stays free for operators that can't spill (merges, exchanges, hash join
//! build sides), so a full set of spillers can't starve the merge that reads their spills back.

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;

use datafusion::error::{DataFusionError, Result};
use datafusion::execution::memory_pool::{
    MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation,
};
use dv_data::budget::{Budget, Consumer, Reservation};

/// A [`MemoryPool`] that reserves from a [`Budget`] as [`Consumer::Query`].
pub struct BudgetPool {
    budget: Budget,
    query_limit: Option<u64>,
    state: Mutex<State>,
}

struct State {
    held: Reservation,
    /// Consumers registered as able to spill.
    registered: usize,
    /// Bytes held by each consumer that can spill and holds any, by consumer id. A consumer can
    /// have several reservations.
    spillers: HashMap<usize, u64>,
    /// Bytes held by consumers that can't spill.
    unspillable: u64,
}

impl BudgetPool {
    /// A pool with no limit of its own: queries may use the whole budget.
    pub fn new(budget: Budget) -> Self {
        Self::with_limit(budget, None)
    }

    /// A pool whose queries hold at most `query_limit` bytes at once, or the whole budget if
    /// that's smaller.
    pub fn with_limit(budget: Budget, query_limit: Option<u64>) -> Self {
        let held = budget
            .reserve(Consumer::Query, 0)
            .expect("an empty reservation always fits");
        Self {
            budget,
            query_limit,
            state: Mutex::new(State {
                held,
                registered: 0,
                spillers: HashMap::new(),
                unspillable: 0,
            }),
        }
    }

    /// The most bytes queries may hold at once now.
    pub fn limit(&self) -> u64 {
        let budget = self.budget.limit();
        self.query_limit.map_or(budget, |l| l.min(budget))
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("query reservation")
    }
}

impl fmt::Debug for BudgetPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BudgetPool")
            .field("reserved", &self.reserved())
            .field("limit", &self.limit())
            .finish()
    }
}

impl fmt::Display for BudgetPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "budget pool ({} bytes reserved)", self.reserved())
    }
}

impl State {
    fn add(&mut self, consumer: &MemoryConsumer, bytes: u64) {
        if bytes == 0 {
            return;
        }
        if consumer.can_spill() {
            *self.spillers.entry(consumer.id()).or_default() += bytes;
        } else {
            self.unspillable += bytes;
        }
    }
}

fn refuse(consumer: &MemoryConsumer, wanted: usize, free: u64, limit: u64) -> DataFusionError {
    DataFusionError::ResourcesExhausted(format!(
        "{} needs {} more, but only {} of the {} query memory limit is free",
        consumer.name(),
        crate::spill::size(wanted as u64),
        crate::spill::size(free),
        crate::spill::size(limit)
    ))
}

impl MemoryPool for BudgetPool {
    fn name(&self) -> &str {
        "dv-budget"
    }

    fn register(&self, consumer: &MemoryConsumer) {
        if consumer.can_spill() {
            self.state().registered += 1;
        }
    }

    fn unregister(&self, consumer: &MemoryConsumer) {
        if consumer.can_spill() {
            let mut state = self.state();
            state.registered = state.registered.saturating_sub(1);
        }
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        let mut state = self.state();
        state.held.grow_unchecked(additional as u64);
        state.add(reservation.consumer(), additional as u64);
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        let mut state = self.state();
        let keep = state.held.bytes().saturating_sub(shrink as u64);
        state.held.shrink_to(keep);
        let consumer = reservation.consumer();
        if consumer.can_spill() {
            if let Some(held) = state.spillers.get_mut(&consumer.id()) {
                *held = held.saturating_sub(shrink as u64);
                if *held == 0 {
                    state.spillers.remove(&consumer.id());
                }
            }
        } else {
            state.unspillable = state.unspillable.saturating_sub(shrink as u64);
        }
    }

    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        let limit = self.limit();
        let mut state = self.state();
        let consumer = reservation.consumer();
        let more = additional as u64;
        if consumer.can_spill() {
            // Each spiller gets an equal share of what the others can't give back, less a
            // quarter of the limit kept for them, since they have no way to make room.
            let spillable = limit.saturating_sub(state.unspillable.max(limit / 4));
            let share = spillable / state.registered.max(1) as u64;
            let mine = state.spillers.get(&consumer.id()).copied().unwrap_or(0);
            if mine + more > share {
                return Err(refuse(
                    consumer,
                    additional,
                    share.saturating_sub(mine),
                    limit,
                ));
            }
        } else if state.held.bytes() + more > limit {
            let free = limit.saturating_sub(state.held.bytes());
            return Err(refuse(consumer, additional, free, limit));
        }
        state
            .held
            .grow(more)
            .map_err(|refused| DataFusionError::ResourcesExhausted(refused.to_string()))?;
        state.add(consumer, more);
        Ok(())
    }

    fn reserved(&self) -> usize {
        self.state().held.bytes() as usize
    }

    fn memory_limit(&self) -> MemoryLimit {
        MemoryLimit::Finite(self.limit() as usize)
    }
}
