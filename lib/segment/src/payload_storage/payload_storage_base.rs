use crate::entry::entry_point::OperationResult;
use crate::types::{Filter, PayloadKeyTypeRef, PointOffsetType};

/// Trait for payload data storage. Should allow filter checks
pub trait PayloadStorage {
    /// Assign payload to a concrete point with a concrete payload value
    fn assign(
        &mut self,
        point_id: PointOffsetType,
        payload: &serde_json::Value,
    ) -> OperationResult<()>;

    /// Get payload for point
    fn payload(&self, point_id: PointOffsetType) -> serde_json::Value;

    /// Delete payload by key
    fn delete(
        &mut self,
        point_id: PointOffsetType,
        key: PayloadKeyTypeRef,
    ) -> OperationResult<Option<serde_json::Value>>;

    /// Drop all payload of the point
    fn drop(&mut self, point_id: PointOffsetType) -> OperationResult<Option<serde_json::Value>>;

    /// Completely drop payload. Pufff!
    fn wipe(&mut self) -> OperationResult<()>;

    /// Force persistence of current storage state.
    fn flush(&self) -> OperationResult<()>;

    /// Iterate all point ids with payload
    fn iter_ids(&self) -> Box<dyn Iterator<Item = PointOffsetType> + '_>;
}

pub trait ConditionChecker {
    /// Check if point satisfies filter condition. Return true if satisfies
    fn check(&self, point_id: PointOffsetType, query: &Filter) -> bool;
}

pub type PayloadStorageSS = dyn PayloadStorage + Sync + Send;
pub type ConditionCheckerSS = dyn ConditionChecker + Sync + Send;
