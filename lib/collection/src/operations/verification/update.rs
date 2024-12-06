use api::rest::{PointInsertOperations, UpdateVectors};
use segment::types::{Filter, StrictModeConfig};

use super::{check_limit_opt, StrictModeVerification};
use crate::collection::Collection;
use crate::common::local_data_stats::LocalDataAtomicStats;
use crate::operations::payload_ops::{DeletePayload, SetPayload};
use crate::operations::point_ops::PointsSelector;
use crate::operations::types::CollectionError;
use crate::operations::vector_ops::DeleteVectors;

impl StrictModeVerification for PointsSelector {
    fn indexed_filter_write(&self) -> Option<&Filter> {
        match self {
            PointsSelector::FilterSelector(filter) => Some(&filter.filter),
            PointsSelector::PointIdsSelector(_) => None,
        }
    }

    fn query_limit(&self) -> Option<usize> {
        None
    }

    fn indexed_filter_read(&self) -> Option<&Filter> {
        None
    }

    fn request_exact(&self) -> Option<bool> {
        None
    }

    fn request_search_params(&self) -> Option<&segment::types::SearchParams> {
        None
    }
}

impl StrictModeVerification for DeleteVectors {
    fn query_limit(&self) -> Option<usize> {
        None
    }

    fn indexed_filter_read(&self) -> Option<&Filter> {
        None
    }

    fn indexed_filter_write(&self) -> Option<&Filter> {
        self.filter.as_ref()
    }

    fn request_exact(&self) -> Option<bool> {
        None
    }

    fn request_search_params(&self) -> Option<&segment::types::SearchParams> {
        None
    }
}

impl StrictModeVerification for SetPayload {
    async fn check_custom(
        &self,
        collection: &Collection,
        strict_mode_config: &StrictModeConfig,
    ) -> Result<(), CollectionError> {
        if let Some(payload_size_limit_bytes) = strict_mode_config.max_collection_payload_size_bytes
        {
            let local_stats = collection.local_stats_estimations().await;
            check_collection_payload_size_limit(payload_size_limit_bytes, local_stats)?;
        }

        Ok(())
    }

    fn indexed_filter_write(&self) -> Option<&Filter> {
        self.filter.as_ref()
    }

    fn query_limit(&self) -> Option<usize> {
        None
    }

    fn indexed_filter_read(&self) -> Option<&Filter> {
        None
    }

    fn request_exact(&self) -> Option<bool> {
        None
    }

    fn request_search_params(&self) -> Option<&segment::types::SearchParams> {
        None
    }
}

impl StrictModeVerification for DeletePayload {
    fn indexed_filter_write(&self) -> Option<&Filter> {
        self.filter.as_ref()
    }

    fn query_limit(&self) -> Option<usize> {
        None
    }

    fn indexed_filter_read(&self) -> Option<&Filter> {
        None
    }

    fn request_exact(&self) -> Option<bool> {
        None
    }

    fn request_search_params(&self) -> Option<&segment::types::SearchParams> {
        None
    }
}

impl StrictModeVerification for PointInsertOperations {
    async fn check_custom(
        &self,
        collection: &Collection,
        strict_mode_config: &StrictModeConfig,
    ) -> Result<(), CollectionError> {
        check_limit_opt(
            Some(self.len()),
            strict_mode_config.upsert_max_batchsize,
            "upsert limit",
        )?;

        check_collection_size_limit(collection, strict_mode_config).await?;

        Ok(())
    }

    fn query_limit(&self) -> Option<usize> {
        None
    }

    fn indexed_filter_read(&self) -> Option<&Filter> {
        None
    }

    fn indexed_filter_write(&self) -> Option<&Filter> {
        None
    }

    fn request_exact(&self) -> Option<bool> {
        None
    }

    fn request_search_params(&self) -> Option<&segment::types::SearchParams> {
        None
    }
}

impl StrictModeVerification for UpdateVectors {
    async fn check_custom(
        &self,
        collection: &Collection,
        strict_mode_config: &StrictModeConfig,
    ) -> Result<(), CollectionError> {
        check_limit_opt(
            Some(self.points.len()),
            strict_mode_config.upsert_max_batchsize,
            "update limit",
        )?;

        check_collection_size_limit(collection, strict_mode_config).await?;

        Ok(())
    }

    fn query_limit(&self) -> Option<usize> {
        None
    }

    fn indexed_filter_read(&self) -> Option<&Filter> {
        None
    }

    fn indexed_filter_write(&self) -> Option<&Filter> {
        None
    }

    fn request_exact(&self) -> Option<bool> {
        None
    }

    fn request_search_params(&self) -> Option<&segment::types::SearchParams> {
        None
    }
}

async fn check_collection_size_limit(
    collection: &Collection,
    strict_mode_config: &StrictModeConfig,
) -> Result<(), CollectionError> {
    let vector_limit = strict_mode_config.max_collection_vector_size_bytes;
    let payload_limit = strict_mode_config.max_collection_payload_size_bytes;

    // All configs to test are disabled/unset so we don't need to check anything nor update cache!.
    if (vector_limit, payload_limit) == (None, None) {
        return Ok(());
    }

    let stats = collection.local_stats_estimations().await;

    if let Some(vector_storage_size_limit_bytes) = vector_limit {
        check_collection_vector_size_limit(vector_storage_size_limit_bytes, stats)?;
    }

    if let Some(payload_storage_size_limit_bytes) = payload_limit {
        check_collection_payload_size_limit(payload_storage_size_limit_bytes, stats)?;
    }

    Ok(())
}

fn check_collection_vector_size_limit(
    max_vec_storage_size_bytes: usize,
    stats: &LocalDataAtomicStats,
) -> Result<(), CollectionError> {
    let vec_storage_size_bytes = stats.get_vector_storage_size();
    if vec_storage_size_bytes >= max_vec_storage_size_bytes {
        let size_in_mb = max_vec_storage_size_bytes as f32 / (1024.0 * 1024.0);
        return Err(CollectionError::bad_request(format!(
            "Max vector storage size limit of {size_in_mb}MB reached!",
        )));
    }

    Ok(())
}

fn check_collection_payload_size_limit(
    max_payload_storage_size_bytes: usize,
    stats: &LocalDataAtomicStats,
) -> Result<(), CollectionError> {
    let payload_storage_size_bytes = stats.get_payload_storage_size();
    if payload_storage_size_bytes >= max_payload_storage_size_bytes {
        let size_in_mb = max_payload_storage_size_bytes as f32 / (1024.0 * 1024.0);
        return Err(CollectionError::bad_request(format!(
            "Max payload storage size limit of {size_in_mb}MB reached!",
        )));
    }

    Ok(())
}
