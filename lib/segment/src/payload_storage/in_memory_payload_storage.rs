use crate::types::{Payload, PointOffsetType};
use std::collections::HashMap;
use crate::entry::entry_point::OperationResult;

/// Same as `SimplePayloadStorage` but without persistence
/// Warn: for tests only
#[derive(Default)]
pub struct InMemoryPayloadStorage {
    pub(crate) payload: HashMap<PointOffsetType, Payload>,
}

impl InMemoryPayloadStorage {
    pub fn payload_ptr(&self, point_id: PointOffsetType) -> Option<&Payload> {
        self.payload.get(&point_id)
    }

    pub fn iter<F>(&self, mut callback: F) -> OperationResult<()>
        where F: FnMut(PointOffsetType, &Payload) -> bool
    {
        for (key, val) in self.payload.iter() {
            let do_continue = callback(*key, val);
            if !do_continue {
                return Ok(());
            }
        }
        Ok(())
    }
}
