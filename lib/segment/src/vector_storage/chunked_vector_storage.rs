use std::path::PathBuf;

use crate::common::operation_error::OperationResult;
use crate::common::Flusher;

/// In case of simple vector storage, vector offset is the same as PointOffsetType.
/// But in case of multivectors, it requires an additional lookup.
pub type VectorOffsetType = usize;

#[allow(clippy::len_without_is_empty)]
pub trait ChunkedVectorStorage<T> {
    fn len(&self) -> usize;

    fn dim(&self) -> usize;

    fn get(&self, key: VectorOffsetType) -> Option<&[T]>;

    fn files(&self) -> Vec<PathBuf>;

    fn flusher(&self) -> Flusher;

    fn push(&mut self, vector: &[T]) -> OperationResult<VectorOffsetType>;

    fn insert(&mut self, key: VectorOffsetType, vector: &[T]) -> OperationResult<()>;

    fn insert_many(
        &mut self,
        start_key: VectorOffsetType,
        vectors: &[T],
        count: usize,
    ) -> OperationResult<()>;

    /// Returns `count` flattened vectors starting from key. if chunk boundary is crossed, returns None
    fn get_many(&self, key: VectorOffsetType, count: usize) -> Option<&[T]>;

    /// Returns batch of vectors by keys.
    /// Keys are expected to be consecutive.
    fn get_batch(&self, keys: &[VectorOffsetType]) -> Vec<&[T]> {
        // TODO replace blanket implementation with optimized one for each storage
        keys.iter()
            .map(|&key| self.get(key).expect("mmap vector not found"))
            .collect()
    }

    fn get_remaining_chunk_keys(&self, start_key: VectorOffsetType) -> usize;

    fn max_vector_size_bytes(&self) -> usize;

    /// True, if this storage is on-disk by default.
    fn is_on_disk(&self) -> bool;
}
