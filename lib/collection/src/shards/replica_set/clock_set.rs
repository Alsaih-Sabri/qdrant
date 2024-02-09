use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub struct ClockSet {
    clocks: Vec<Arc<Clock>>,
}

impl ClockSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the first available clock from the set, or create a new one.
    pub fn get_clock(&mut self) -> ClockGuard {
        for (id, clock) in self.clocks.iter().enumerate() {
            if clock.try_lock() {
                return ClockGuard::new(id, clock.clone());
            }
        }

        let id = self.clocks.len();
        let clock = Arc::new(Clock::new_locked());

        self.clocks.push(clock.clone());

        ClockGuard::new(id, clock)
    }
}

#[derive(Debug)]
pub struct ClockGuard {
    id: usize,
    clock: Arc<Clock>,
}

impl ClockGuard {
    fn new(id: usize, clock: Arc<Clock>) -> Self {
        Self { id, clock }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    /// Advance clock by a single tick and return current tick.
    #[must_use = "new clock value must be used"]
    pub fn tick_once(&mut self) -> u64 {
        self.clock.tick_once()
    }

    /// Advance clock to `new_tick`, if `new_tick` is newer than current tick.
    pub fn advance_to(&mut self, new_tick: u64) {
        self.clock.advance_to(new_tick)
    }
}

impl Drop for ClockGuard {
    fn drop(&mut self) {
        self.clock.release();
    }
}

#[derive(Debug)]
struct Clock {
    /// Tracks the *next* clock tick
    next_tick: AtomicU64,
    available: AtomicBool,
}

impl Clock {
    fn new_locked() -> Self {
        Self {
            next_tick: 0.into(),
            available: false.into(),
        }
    }

    /// Advance clock by a single tick and return current tick.
    ///
    /// # Thread safety
    ///
    /// Clock *has to* be locked (using [`Clock::lock`]) before calling `tick_once`!
    #[must_use = "new clock tick value must be used"]
    fn tick_once(&self) -> u64 {
        // `Clock` tracks *next* tick, so we increment `next_tick` by 1 and return *previous* value
        // of `next_tick` (which is *exactly* what `fetch_add(1)` does)
        let current_tick = self.next_tick.fetch_add(1, Ordering::Relaxed);

        // If `current_tick` is `0`, then "revert" `next_tick` back to `0`.
        // We expect that `advance_to` would be used to advance "past" the initial `0`.
        //
        // Executing multiple atomic operations sequentially is not strictly thread-safe,
        // but we expect that `Clock` would be "locked" before calling `tick_once`.
        if current_tick == 0 {
            self.next_tick.store(0, Ordering::Relaxed);
        }

        current_tick
    }

    /// Advance clock to `new_tick`, if `new_tick` is newer than current tick.
    fn advance_to(&self, new_tick: u64) {
        // `Clock` tracks *next* tick, so if we want to advance *current* tick to `new_tick`,
        // we have to advance `next_tick` to `new_tick + 1`
        self.next_tick.fetch_max(new_tick + 1, Ordering::Relaxed);
    }

    /// Try to acquire exclusive lock over this clock.
    ///
    /// Returns `true` if the lock was successfully acquired, or `false` if the clock is already
    /// locked.
    fn try_lock(&self) -> bool {
        self.available.swap(false, Ordering::Relaxed)
    }

    /// Release the exclusive lock over this clock.
    ///
    /// No-op if the clock is not locked.
    fn release(&self) {
        self.available.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tick a single clock, it should increment after we advance it from 0 (or higher).
    #[test]
    fn test_clock_set_single_tick() {
        let mut clock_set = ClockSet::new();

        // Don't tick from 0 unless we explicitly advance
        assert_eq!(clock_set.get_clock().tick_once(), 0);
        assert_eq!(clock_set.get_clock().tick_once(), 0);
        assert_eq!(clock_set.get_clock().tick_once(), 0);

        clock_set.get_clock().advance_to(0);

        // Following ticks should increment
        assert_eq!(clock_set.get_clock().tick_once(), 1);
        assert_eq!(clock_set.get_clock().tick_once(), 2);
        assert_eq!(clock_set.get_clock().tick_once(), 3);
        assert_eq!(clock_set.get_clock().tick_once(), 4);
    }

    /// Test a single clock, but tick it multiple times on the same guard.
    #[test]
    fn test_clock_set_single_tick_twice() {
        let mut clock_set = ClockSet::new();

        // Don't tick from 0 unless we explicitly advance
        {
            let mut clock = clock_set.get_clock();
            assert_eq!(clock.tick_once(), 0);
            assert_eq!(clock.tick_once(), 0);
            clock.advance_to(0);
        }

        // Following ticks should increment
        {
            let mut clock = clock_set.get_clock();
            assert_eq!(clock.tick_once(), 1);
            assert_eq!(clock.tick_once(), 2);
            assert_eq!(clock.tick_once(), 3);
            assert_eq!(clock.tick_once(), 4);
        }
    }

    #[test]
    fn test_clock_set_single_advance_high() {
        let mut clock_set = ClockSet::new();

        // Bring the clock up to 4
        assert_eq!(clock_set.get_clock().tick_once(), 0);
        clock_set.get_clock().advance_to(0);
        assert_eq!(clock_set.get_clock().tick_once(), 1);
        assert_eq!(clock_set.get_clock().tick_once(), 2);
        assert_eq!(clock_set.get_clock().tick_once(), 3);
        assert_eq!(clock_set.get_clock().tick_once(), 4);

        // If we advance to 100, we should continue from 101
        clock_set.get_clock().advance_to(100);
        assert_eq!(clock_set.get_clock().tick_once(), 101);
    }

    #[test]
    fn test_clock_set_single_advance_low() {
        let mut clock_set = ClockSet::new();

        // Bring the clock up to 4
        assert_eq!(clock_set.get_clock().tick_once(), 0);
        clock_set.get_clock().advance_to(0);
        assert_eq!(clock_set.get_clock().tick_once(), 1);
        assert_eq!(clock_set.get_clock().tick_once(), 2);
        assert_eq!(clock_set.get_clock().tick_once(), 3);
        assert_eq!(clock_set.get_clock().tick_once(), 4);

        // If we advance to a low number, just continue
        clock_set.get_clock().advance_to(0);
        assert_eq!(clock_set.get_clock().tick_once(), 5);

        clock_set.get_clock().advance_to(1);
        assert_eq!(clock_set.get_clock().tick_once(), 6);
    }

    #[test]
    fn test_clock_multi_tick() {
        let mut clock_set = ClockSet::new();

        // 2 parallel operations, that fails and doesn't advance
        {
            let mut clock1 = clock_set.get_clock();
            let mut clock2 = clock_set.get_clock();
            assert_eq!(clock1.tick_once(), 0);
            assert_eq!(clock2.tick_once(), 0);
        }

        // 2 parallel operations
        {
            let mut clock1 = clock_set.get_clock();
            let mut clock2 = clock_set.get_clock();
            assert_eq!(clock1.tick_once(), 0);
            assert_eq!(clock2.tick_once(), 0);
            clock1.advance_to(0);
            clock2.advance_to(0);
        }

        // 1 operation
        {
            let mut clock1 = clock_set.get_clock();
            assert_eq!(clock1.tick_once(), 1);
            clock1.advance_to(1);
        }

        // 1 operation, without advancing should still tick
        {
            let mut clock1 = clock_set.get_clock();
            assert_eq!(clock1.tick_once(), 2);
        }

        // 2 parallel operations
        {
            let mut clock1 = clock_set.get_clock();
            let mut clock2 = clock_set.get_clock();
            assert_eq!(clock1.tick_once(), 3);
            assert_eq!(clock2.tick_once(), 1);
            clock1.advance_to(3);
            clock2.advance_to(1);
        }

        // 3 parallel operations, but clock 2 was much newer on some node
        {
            let mut clock1 = clock_set.get_clock();
            let mut clock2 = clock_set.get_clock();
            let mut clock3 = clock_set.get_clock();
            assert_eq!(clock1.tick_once(), 4);
            assert_eq!(clock2.tick_once(), 2);
            assert_eq!(clock3.tick_once(), 0);
            clock1.advance_to(4);
            clock2.advance_to(10);
            clock3.advance_to(0);
        }

        // 3 parallel operations, advancing in a different order should not matter
        {
            let mut clock1 = clock_set.get_clock();
            let mut clock2 = clock_set.get_clock();
            let mut clock3 = clock_set.get_clock();
            assert_eq!(clock1.tick_once(), 5);
            assert_eq!(clock2.tick_once(), 11);
            assert_eq!(clock3.tick_once(), 1);
            clock3.advance_to(1);
            clock2.advance_to(11);
            clock1.advance_to(5);
        }

        // 3 parallel operations, advancing just some should still tick all
        {
            let mut clock1 = clock_set.get_clock();
            let mut clock2 = clock_set.get_clock();
            let mut clock3 = clock_set.get_clock();
            assert_eq!(clock1.tick_once(), 6);
            assert_eq!(clock2.tick_once(), 12);
            assert_eq!(clock3.tick_once(), 2);
            clock2.advance_to(12);
        }

        // 1 operation
        {
            let mut clock1 = clock_set.get_clock();
            assert_eq!(clock1.tick_once(), 7);
        }

        // Test final state of all clocks
        {
            let mut clock1 = clock_set.get_clock();
            let mut clock2 = clock_set.get_clock();
            let mut clock3 = clock_set.get_clock();
            let mut clock4 = clock_set.get_clock();
            assert_eq!(clock1.tick_once(), 8);
            assert_eq!(clock2.tick_once(), 13);
            assert_eq!(clock3.tick_once(), 3);
            assert_eq!(clock4.tick_once(), 0);
        }
    }
}
