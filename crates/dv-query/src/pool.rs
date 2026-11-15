//! DataFusion's memory pool, backed by the engine's shared memory budget.
//!
//! Every operator that buffers (hash joins, aggregates, sorts) reserves through this pool, so
//! query memory counts against the same budget as array chunks, caches and GPU buffers. When the
//! budget refuses, the operator spills if it can and otherwise fails that one query with the
//! budget's message.

use std::fmt;
use std::sync::Mutex;

use datafusion::error::{DataFusionError, Result};
use datafusion::execution::memory_pool::{MemoryLimit, MemoryPool, MemoryReservation};
use dv_data::budget::{Budget, Consumer, Reservation};

/// A [`MemoryPool`] that reserves from a [`Budget`] as [`Consumer::Query`].
pub struct BudgetPool {
    budget: Budget,
    held: Mutex<Reservation>,
}

impl BudgetPool {
    pub fn new(budget: Budget) -> Self {
        let held = budget
            .reserve(Consumer::Query, 0)
            .expect("an empty reservation always fits");
        Self {
            budget,
            held: Mutex::new(held),
        }
    }

    fn held(&self) -> std::sync::MutexGuard<'_, Reservation> {
        self.held.lock().expect("query reservation")
    }
}

impl fmt::Debug for BudgetPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BudgetPool")
            .field("reserved", &self.reserved())
            .field("limit", &self.budget.limit())
            .finish()
    }
}

impl fmt::Display for BudgetPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "budget pool ({} bytes reserved)", self.reserved())
    }
}

impl MemoryPool for BudgetPool {
    fn name(&self) -> &str {
        "dv-budget"
    }

    fn grow(&self, _reservation: &MemoryReservation, additional: usize) {
        self.held().grow_unchecked(additional as u64);
    }

    fn shrink(&self, _reservation: &MemoryReservation, shrink: usize) {
        let mut held = self.held();
        let keep = held.bytes().saturating_sub(shrink as u64);
        held.shrink_to(keep);
    }

    fn try_grow(&self, _reservation: &MemoryReservation, additional: usize) -> Result<()> {
        self.held()
            .grow(additional as u64)
            .map_err(|refused| DataFusionError::ResourcesExhausted(refused.to_string()))
    }

    fn reserved(&self) -> usize {
        self.held().bytes() as usize
    }

    fn memory_limit(&self) -> MemoryLimit {
        MemoryLimit::Finite(self.budget.limit() as usize)
    }
}
