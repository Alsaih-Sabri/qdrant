use segment::types::{Filter, SearchParams};

use super::StrictModeVerification;
use crate::collection::Collection;
use crate::operations::config_diff::StrictModeConfig;
use crate::operations::types::{CollectionError, DiscoverRequest, DiscoverRequestBatch};

impl StrictModeVerification for DiscoverRequest {
    fn query_limit(&self) -> Option<usize> {
        Some(self.discover_request.limit)
    }

    fn indexed_filter_read(&self) -> Option<&Filter> {
        self.discover_request.filter.as_ref()
    }

    fn request_search_params(&self) -> Option<&SearchParams> {
        self.discover_request.params.as_ref()
    }

    fn timeout(&self) -> Option<usize> {
        None
    }

    fn indexed_filter_write(&self) -> Option<&Filter> {
        None
    }

    fn request_exact(&self) -> Option<bool> {
        None
    }
}

impl StrictModeVerification for DiscoverRequestBatch {
    fn check_strict_mode(
        &self,
        collection: &Collection,
        strict_mode_config: &StrictModeConfig,
    ) -> Result<(), CollectionError> {
        for i in self.searches.iter() {
            i.check_strict_mode(collection, strict_mode_config)?;
        }

        Ok(())
    }

    fn query_limit(&self) -> Option<usize> {
        None
    }

    fn timeout(&self) -> Option<usize> {
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

    fn request_search_params(&self) -> Option<&SearchParams> {
        None
    }
}
