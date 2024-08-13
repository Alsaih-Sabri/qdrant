use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use common::types::TelemetryDetail;

use crate::common::operation_error::{OperationResult, SegmentFailedState};
use crate::data_types::facets::{FacetRequestInternal, FacetValue};
use crate::data_types::named_vectors::NamedVectors;
use crate::data_types::order_by::{OrderBy, OrderValue};
use crate::data_types::query_context::{QueryContext, SegmentQueryContext};
use crate::data_types::vectors::{QueryVector, Vector};
use crate::index::field_index::CardinalityEstimation;
use crate::json_path::JsonPath;
use crate::telemetry::SegmentTelemetry;
use crate::types::{
    Filter, Payload, PayloadFieldSchema, PayloadKeyType, PayloadKeyTypeRef, PointIdType,
    ScoredPoint, SearchParams, SegmentConfig, SegmentInfo, SegmentType, SeqNumberType, WithPayload,
    WithVector,
};

/// Define all operations which can be performed with Segment or Segment-like entity.
///
/// Assume all operations are idempotent - which means that no matter how many times an operation
/// is executed - the storage state will be the same.
pub trait SegmentEntry {
    /// Get current update version of the segment
    fn version(&self) -> SeqNumberType;

    /// Get version of specified point
    fn point_version(&self, point_id: PointIdType) -> Option<SeqNumberType>;

    #[allow(clippy::too_many_arguments)]
    fn search_batch(
        &self,
        vector_name: &str,
        query_vectors: &[&QueryVector],
        with_payload: &WithPayload,
        with_vector: &WithVector,
        filter: Option<&Filter>,
        top: usize,
        params: Option<&SearchParams>,
        query_context: SegmentQueryContext,
    ) -> OperationResult<Vec<Vec<ScoredPoint>>>;

    fn upsert_point(
        &mut self,
        op_num: SeqNumberType,
        point_id: PointIdType,
        vectors: NamedVectors,
    ) -> OperationResult<bool>;

    fn delete_point(
        &mut self,
        op_num: SeqNumberType,
        point_id: PointIdType,
    ) -> OperationResult<bool>;

    fn update_vectors(
        &mut self,
        op_num: SeqNumberType,
        point_id: PointIdType,
        vectors: NamedVectors,
    ) -> OperationResult<bool>;

    fn delete_vector(
        &mut self,
        op_num: SeqNumberType,
        point_id: PointIdType,
        vector_name: &str,
    ) -> OperationResult<bool>;

    fn set_payload(
        &mut self,
        op_num: SeqNumberType,
        point_id: PointIdType,
        payload: &Payload,
        key: &Option<JsonPath>,
    ) -> OperationResult<bool>;

    fn set_full_payload(
        &mut self,
        op_num: SeqNumberType,
        point_id: PointIdType,
        full_payload: &Payload,
    ) -> OperationResult<bool>;

    fn delete_payload(
        &mut self,
        op_num: SeqNumberType,
        point_id: PointIdType,
        key: PayloadKeyTypeRef,
    ) -> OperationResult<bool>;

    fn clear_payload(
        &mut self,
        op_num: SeqNumberType,
        point_id: PointIdType,
    ) -> OperationResult<bool>;

    fn vector(&self, vector_name: &str, point_id: PointIdType) -> OperationResult<Option<Vector>>;

    fn all_vectors(&self, point_id: PointIdType) -> OperationResult<NamedVectors>;

    /// Retrieve payload for the point
    /// If not found, return empty payload
    fn payload(&self, point_id: PointIdType) -> OperationResult<Payload>;

    /// Iterator over all points in segment in ascending order.
    fn iter_points(&self) -> Box<dyn Iterator<Item = PointIdType> + '_>;

    /// Paginate over points which satisfies filtering condition starting with `offset` id including.
    ///
    /// Cancelled by `is_stopped` flag.
    fn read_filtered<'a>(
        &'a self,
        offset: Option<PointIdType>,
        limit: Option<usize>,
        filter: Option<&'a Filter>,
        is_stopped: &AtomicBool,
    ) -> Vec<PointIdType>;

    /// Return points which satisfies filtering condition ordered by the `order_by.key` field,
    /// starting with `order_by.start_from` value including.
    ///
    /// Will fail if there is no index for the order_by key.
    /// Cancelled by `is_stopped` flag.
    fn read_ordered_filtered<'a>(
        &'a self,
        limit: Option<usize>,
        filter: Option<&'a Filter>,
        order_by: &'a OrderBy,
        is_stopped: &AtomicBool,
    ) -> OperationResult<Vec<(OrderValue, PointIdType)>>;

    /// Return random points which satisfies filtering condition.
    ///
    /// Cancelled by `is_stopped` flag.
    fn read_random_filtered(
        &self,
        limit: usize,
        filter: Option<&Filter>,
        is_stopped: &AtomicBool,
    ) -> Vec<PointIdType>;

    /// Read points in [from; to) range
    fn read_range(&self, from: Option<PointIdType>, to: Option<PointIdType>) -> Vec<PointIdType>;

    /// Return the largest counts for the given facet request.
    fn facet(
        &self,
        request: &FacetRequestInternal,
        is_stopped: &AtomicBool,
    ) -> OperationResult<HashMap<FacetValue, usize>>;

    /// Check if there is point with `point_id` in this segment.
    fn has_point(&self, point_id: PointIdType) -> bool;

    /// Estimate available point count in this segment for given filter.
    fn estimate_point_count<'a>(&'a self, filter: Option<&'a Filter>) -> CardinalityEstimation;

    fn vector_names(&self) -> HashSet<String>;

    /// Number of available points
    ///
    /// - excludes soft deleted points
    fn available_point_count(&self) -> usize;

    /// Number of deleted points
    fn deleted_point_count(&self) -> usize;

    /// Size of all available vectors in storage
    fn available_vectors_size_in_bytes(&self, vector_name: &str) -> OperationResult<usize>;

    /// Max value from all `available_vectors_size_in_bytes`
    fn max_available_vectors_size_in_bytes(&self) -> OperationResult<usize> {
        self.vector_names()
            .into_iter()
            .map(|vector_name| self.available_vectors_size_in_bytes(&vector_name))
            .collect::<OperationResult<Vec<_>>>()
            .map(|sizes| sizes.into_iter().max().unwrap_or_default())
    }

    /// Get segment type
    fn segment_type(&self) -> SegmentType;

    /// Get current stats of the segment
    fn info(&self) -> SegmentInfo;

    /// Get segment configuration
    fn config(&self) -> &SegmentConfig;

    /// Get current stats of the segment
    fn is_appendable(&self) -> bool;

    /// Flushes current segment state into a persistent storage, if possible
    /// if sync == true, block current thread while flushing
    ///
    /// Returns maximum version number which is guaranteed to be persisted.
    fn flush(&self, sync: bool, force: bool) -> OperationResult<SeqNumberType>;

    /// Removes all persisted data and forces to destroy segment
    fn drop_data(self) -> OperationResult<()>;

    /// Path to data, owned by segment
    fn data_path(&self) -> PathBuf;

    /// Delete field index, if exists
    fn delete_field_index(
        &mut self,
        op_num: SeqNumberType,
        key: PayloadKeyTypeRef,
    ) -> OperationResult<bool>;

    /// Create index for a payload field, if not exists
    fn create_field_index(
        &mut self,
        op_num: SeqNumberType,
        key: PayloadKeyTypeRef,
        field_schema: Option<&PayloadFieldSchema>,
    ) -> OperationResult<bool>;

    /// Get indexed fields
    fn get_indexed_fields(&self) -> HashMap<PayloadKeyType, PayloadFieldSchema>;

    /// Checks if segment errored during last operations
    fn check_error(&self) -> Option<SegmentFailedState>;

    /// Delete points by the given filter
    fn delete_filtered<'a>(
        &'a mut self,
        op_num: SeqNumberType,
        filter: &'a Filter,
    ) -> OperationResult<usize>;

    /// Take a snapshot of the segment.
    ///
    /// Creates a tar archive of the segment directory into `snapshot_dir_path`.
    /// Uses `temp_path` to prepare files to archive.
    fn take_snapshot(&self, temp_path: &Path, snapshot_dir_path: &Path)
        -> OperationResult<PathBuf>;

    // Get collected telemetry data of segment
    fn get_telemetry_data(&self, detail: TelemetryDetail) -> SegmentTelemetry;

    fn fill_query_context(&self, query_context: &mut QueryContext);
}
