use std::collections::HashMap;

use bitvec::prelude::BitSlice;
use sparse::common::types::{DimId, DimWeight};

use crate::data_types::tiny_map;

pub struct QueryContext {
    /// Total amount of available points in the segment.
    available_point_count: usize,

    /// Parameter, which defines how big a plain segment can be to be considered
    /// small enough to be searched with `indexed_only` option.
    search_optimized_threshold_kb: usize,

    /// Statistics of the element frequency,
    /// collected over all segments.
    /// Required for processing sparse vector search with `idf-dot` similarity.
    #[allow(dead_code)]
    idf: tiny_map::TinyMap<String, HashMap<DimId, usize>>,
}

impl QueryContext {
    pub fn new(search_optimized_threshold_kb: usize) -> Self {
        Self {
            available_point_count: 0,
            search_optimized_threshold_kb,
            idf: tiny_map::TinyMap::new(),
        }
    }

    pub fn available_point_count(&self) -> usize {
        self.available_point_count
    }

    pub fn search_optimized_threshold_kb(&self) -> usize {
        self.search_optimized_threshold_kb
    }

    pub fn add_available_point_count(&mut self, count: usize) {
        self.available_point_count += count;
    }

    /// Fill indices of sparse vectors, which are required for `idf-dot` similarity
    /// with zeros, so the statistics can be collected.
    pub fn init_idf(&mut self, vector_name: &str, indices: &[DimId]) {
        // ToDo: Would be nice to have an implementation of `entry` for `TinyMap`.
        let idf = if let Some(idf) = self.idf.get_mut(vector_name) {
            idf
        } else {
            self.idf.insert(vector_name.to_string(), HashMap::default());
            self.idf.get_mut(vector_name).unwrap()
        };

        for index in indices {
            idf.insert(*index, 0);
        }
    }

    pub fn mut_idf(&mut self) -> &mut tiny_map::TinyMap<String, HashMap<DimId, usize>> {
        &mut self.idf
    }

    pub fn get_segment_query_context(&self) -> SegmentQueryContext {
        SegmentQueryContext {
            query_context: Some(self),
            deleted_points: None,
        }
    }
}

impl Default for QueryContext {
    fn default() -> Self {
        Self::new(usize::MAX) // Search optimized threshold won't affect the search.
    }
}

/// Defines context of the search query on the segment level
#[derive(Default)]
pub struct SegmentQueryContext<'a> {
    query_context: Option<&'a QueryContext>,
    deleted_points: Option<&'a BitSlice>,
}

impl<'a> SegmentQueryContext<'a> {
    pub fn get_vector_context(&self, vector_name: &str) -> VectorQueryContext {
        let query_context = if let Some(query_context) = self.query_context {
            query_context
        } else {
            return VectorQueryContext::default();
        };
        VectorQueryContext {
            available_point_count: query_context.available_point_count,
            search_optimized_threshold_kb: query_context.search_optimized_threshold_kb,
            idf: query_context.idf.get(vector_name),
            deleted_points: self.deleted_points,
        }
    }

    pub fn with_deleted_points(&self, deleted_points: &'a BitSlice) -> Self {
        SegmentQueryContext {
            query_context: self.query_context,
            deleted_points: Some(deleted_points),
        }
    }
}

/// Query context related to a specific vector
pub struct VectorQueryContext<'a> {
    /// Total amount of available points in the segment.
    available_point_count: usize,

    /// Parameter, which defines how big a plain segment can be to be considered
    /// small enough to be searched with `indexed_only` option.
    search_optimized_threshold_kb: usize,

    idf: Option<&'a HashMap<DimId, usize>>,

    deleted_points: Option<&'a BitSlice>,
}

impl VectorQueryContext<'_> {
    pub fn available_point_count(&self) -> usize {
        self.available_point_count
    }

    pub fn search_optimized_threshold_kb(&self) -> usize {
        self.search_optimized_threshold_kb
    }

    pub fn deleted_points(&self) -> Option<&BitSlice> {
        self.deleted_points
    }

    /// Compute advanced formula for Inverse Document Frequency (IDF) according to wikipedia.
    /// This should account for corner cases when `df` and `n` are small or zero.
    #[inline]
    fn fancy_idf(n: DimWeight, df: DimWeight) -> DimWeight {
        ((n - df + 0.5) / (df + 0.5) + 1.).ln()
    }

    pub fn remap_idf_weights(&self, indices: &[DimId], weights: &mut [DimWeight]) {
        // Number of documents
        let n = self.available_point_count as DimWeight;
        for (weight, index) in weights.iter_mut().zip(indices) {
            // Document frequency
            let df = self
                .idf
                .and_then(|idf| idf.get(index))
                .copied()
                .unwrap_or(0);

            *weight *= Self::fancy_idf(n, df as DimWeight);
        }
    }

    pub fn is_require_idf(&self) -> bool {
        self.idf.is_some()
    }
}

impl Default for VectorQueryContext<'_> {
    fn default() -> Self {
        VectorQueryContext {
            available_point_count: 0,
            search_optimized_threshold_kb: usize::MAX,
            idf: None,
            deleted_points: None,
        }
    }
}
