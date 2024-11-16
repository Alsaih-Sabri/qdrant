use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::hardware_counter::HardwareCounterCell;

/// A "slow" but thread-safe accumulator for measurement results of `HardwareCounterCell` values.
/// This type is completely reference counted and clones of this type will read/write the same values as their origin structure.
pub struct HwMeasurementAcc {
    cpu_counter: Arc<AtomicUsize>,
    drain: Option<Arc<Self>>,
}

impl HwMeasurementAcc {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_values(cpu: usize) -> Self {
        Self {
            cpu_counter: Arc::new(AtomicUsize::new(cpu)),
            drain: None,
        }
    }

    pub fn new_with_drain(drain: Arc<Self>) -> Self {
        Self {
            cpu_counter: Arc::new(AtomicUsize::new(0)),
            drain: Some(drain),
        }
    }

    /// Creates a new instance of `HwMeasurementAcc` we call "collector" with self as a "parent".
    /// All values collected by the collector will be applied to the parents counters
    /// when the collector drops.
    /// This allows using `HwMeasurementAcc` in multi-threaded or async code without cloning it.
    pub fn new_collector(&self) -> HwMeasurementAcc {
        HwMeasurementAcc {
            cpu_counter: Arc::new(AtomicUsize::new(0)),
            drain: Some(Arc::new(Self {
                cpu_counter: self.cpu_counter.clone(),
                drain: None,
            })),
        }
    }

    /// Creates completely independent copy of the current `HwMeasurementAcc`'s values.
    pub fn deep_copy(&self) -> Self {
        Self::new_with_values(self.get_cpu())
    }

    pub fn get_cpu(&self) -> usize {
        self.cpu_counter.load(Ordering::Relaxed)
    }

    /// Discards all values of the `HwMeasurementAcc`.
    ///
    /// This function explicitly states that we don't care about the measurement result and therefore want
    /// to disable the check on `drop`.
    pub fn discard(&self) {
        let HwMeasurementAcc {
            cpu_counter,
            drain: _,
        } = self;

        cpu_counter.store(0, Ordering::Relaxed);
    }

    /// Merge, but for internal use only.
    fn merge_from_ref(&self, other: &Self) {
        let HwMeasurementAcc {
            ref cpu_counter,
            drain: _,
        } = other;
        // Discard of the drain is not a problem, as no counters are lost.
        let cpu = cpu_counter.swap(0, Ordering::Relaxed);
        self.cpu_counter.fetch_add(cpu, Ordering::Relaxed);
    }

    /// Consumes and accumulates the values from `other` into the accumulator.
    pub fn merge(&self, other: Self) {
        self.merge_from_ref(&other);
    }

    /// Consumes and accumulates the values from `hw_counter_cell` into the accumulator.
    pub fn merge_from_cell(&self, hw_counter_cell: impl Into<HardwareCounterCell>) {
        let HardwareCounterCell { ref cpu_counter } = hw_counter_cell.into();

        self.cpu_counter
            .fetch_add(cpu_counter.take(), Ordering::Relaxed);
    }

    /// Returns `true` if all values of the `HwMeasurementAcc` are zero.
    pub fn is_zero(&self) -> bool {
        let HwMeasurementAcc {
            ref cpu_counter,
            drain: _,
        } = self;
        cpu_counter.load(Ordering::Relaxed) == 0
    }
}

impl Drop for HwMeasurementAcc {
    // `HwMeasurementAcc` holds collected hardware measurements for certain operations. To not accidentally lose measured values, we have
    // this custom drop() function, panicking if it gets dropped while still holding values (in debug/test builds).
    //
    // If you encountered it panicking here, it means that you probably don't have propagated or handled some collected measurements properly,
    // or didn't discard them when unneeded using `discard()`.
    //
    // You can apply values by utilizing `merge_from_cell(other_cell)` or `merge(other)`, consuming the other counter, which then can be dropped safely.
    fn drop(&mut self) {
        if let Some(drain) = &self.drain {
            drain.merge_from_ref(self);
        }

        #[cfg(any(debug_assertions, test))] // Fail in both, release and debug tests.
        {
            if !self.is_zero() {
                panic!("HwMeasurementAcc dropped while still holding values!")
            }
        }
    }
}

impl Default for HwMeasurementAcc {
    fn default() -> Self {
        Self {
            cpu_counter: Arc::new(AtomicUsize::new(0)),
            drain: None,
        }
    }
}
