use std::cmp;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use common::cpu::CpuPermit;
use parking_lot::RwLock;
use rocksdb::DB;

use super::{
    create_id_tracker, create_payload_storage, get_vector_storage_path, new_segment_path,
    open_segment_db, open_vector_storage,
};
use crate::common::error_logging::LogError;
use crate::common::operation_error::{check_process_stopped, OperationError, OperationResult};
use crate::entry::entry_point::SegmentEntry;
use crate::id_tracker::{IdTracker, IdTrackerEnum};
use crate::index::hnsw_index::num_rayon_threads;
use crate::index::{PayloadIndex, VectorIndex};
use crate::payload_storage::payload_storage_enum::PayloadStorageEnum;
use crate::payload_storage::PayloadStorage;
use crate::segment::Segment;
use crate::segment_constructor::{build_segment, load_segment};
use crate::types::{Indexes, PayloadFieldSchema, PayloadKeyType, SegmentConfig, SeqNumberType};
use crate::vector_storage::quantized::quantized_vectors::QuantizedVectors;
use crate::vector_storage::{VectorStorage, VectorStorageEnum};

/// Structure for constructing segment out of several other segments
pub struct SegmentBuilder {
    version: SeqNumberType,
    _database: Arc<RwLock<DB>>,
    id_tracker: IdTrackerEnum,
    payload_storage: PayloadStorageEnum,
    vector_storages: HashMap<String, VectorStorageEnum>,
    segment_config: SegmentConfig,

    destination_path: PathBuf,
    temp_path: PathBuf,
    indexed_fields: HashMap<PayloadKeyType, PayloadFieldSchema>,
}

impl SegmentBuilder {
    pub fn new(
        segment_path: &Path,
        temp_dir: &Path,
        segment_config: &SegmentConfig,
    ) -> OperationResult<Self> {
        // When we build a new segment, it is empty at first,
        // so we can ignore the `stopped` flag
        let stopped = AtomicBool::new(false);

        let temp_path = new_segment_path(temp_dir);

        let database = open_segment_db(&temp_path, segment_config)?;

        let id_tracker = create_id_tracker(database.clone())?;

        let payload_storage = create_payload_storage(database.clone(), segment_config)?;

        let mut vector_storages = HashMap::new();

        for (vector_name, vector_config) in &segment_config.vector_data {
            let vector_storage_path = get_vector_storage_path(segment_path, vector_name);
            let vector_storage = open_vector_storage(
                &database,
                vector_config,
                &stopped,
                &vector_storage_path,
                vector_name,
            )?;

            vector_storages.insert(vector_name.to_owned(), vector_storage);
        }

        let destination_path = segment_path.join(temp_path.file_name().unwrap());

        Ok(SegmentBuilder {
            version: Default::default(), // default version is 0
            _database: database,
            id_tracker,
            payload_storage,
            vector_storages,
            segment_config: segment_config.clone(),

            destination_path,
            temp_path,
            indexed_fields: Default::default(),
        })
    }

    pub fn remove_indexed_field(&mut self, field: &PayloadKeyType) {
        self.indexed_fields.remove(field);
    }

    pub fn add_indexed_field(&mut self, field: PayloadKeyType, schema: PayloadFieldSchema) {
        self.indexed_fields.insert(field, schema);
    }

    /// Update current segment builder with all (not deleted) vectors and payload form `other` segment
    /// Perform index building at the end of update
    ///
    /// # Arguments
    ///
    /// * `other` - segment to add into construction
    ///
    /// # Result
    ///
    /// * `bool` - if `true` - data successfully added, if `false` - process was interrupted
    ///
    pub fn update_from(&mut self, other: &Segment, stopped: &AtomicBool) -> OperationResult<bool> {
        self.version = cmp::max(self.version, other.version());

        let other_id_tracker = other.id_tracker.borrow();
        let other_vector_storages: HashMap<_, _> = other
            .vector_data
            .iter()
            .map(|(vector_name, vector_data)| {
                (vector_name.to_owned(), vector_data.vector_storage.borrow())
            })
            .collect();
        let other_payload_index = other.payload_index.borrow();

        let id_tracker = &mut self.id_tracker;

        if self.vector_storages.len() != other_vector_storages.len() {
            return Err(OperationError::service_error(
                format!("Self and other segments have different vector names count. Self count: {}, other count: {}", self.vector_storages.len(), other_vector_storages.len()),
            ));
        }

        let mut new_internal_range = None;
        for (vector_name, vector_storage) in &mut self.vector_storages {
            check_process_stopped(stopped)?;
            let other_vector_storage = other_vector_storages.get(vector_name).ok_or_else(|| {
                OperationError::service_error(format!(
                    "Cannot update from other segment because if missing vector name {vector_name}"
                ))
            })?;
            let internal_range = vector_storage.update_from(
                other_vector_storage,
                &mut other_id_tracker.iter_ids(),
                stopped,
            )?;
            match new_internal_range.clone() {
                Some(new_internal_range) => {
                    if new_internal_range != internal_range {
                        return Err(OperationError::service_error(
                            "Internal ids range mismatch between self segment vectors and other segment vectors",
                        ));
                    }
                }
                None => new_internal_range = Some(internal_range.clone()),
            }
        }

        if let Some(new_internal_range) = new_internal_range {
            let internal_id_iter = new_internal_range.zip(other_id_tracker.iter_ids());

            for (new_internal_id, old_internal_id) in internal_id_iter {
                check_process_stopped(stopped)?;

                let external_id =
                    if let Some(external_id) = other_id_tracker.external_id(old_internal_id) {
                        external_id
                    } else {
                        log::warn!(
                            "Cannot find external id for internal id {old_internal_id}, skipping"
                        );
                        continue;
                    };

                let other_version = other_id_tracker
                    .internal_version(old_internal_id)
                    .unwrap_or_else(|| {
                        log::debug!(
                            "Internal version not found for internal id {old_internal_id}, using 0"
                        );
                        0
                    });

                match id_tracker.internal_id(external_id) {
                    None => {
                        // New point, just insert
                        id_tracker.set_link(external_id, new_internal_id)?;
                        id_tracker.set_internal_version(new_internal_id, other_version)?;
                        let other_payload = other_payload_index.payload(old_internal_id)?;
                        // Propagate payload to new segment
                        if !other_payload.is_empty() {
                            self.payload_storage
                                .assign(new_internal_id, &other_payload)?;
                        }
                    }
                    Some(existing_internal_id) => {
                        // Point exists in both: newly constructed and old segments, so we need to merge them
                        // Based on version
                        let existing_version =
                            id_tracker.internal_version(existing_internal_id).unwrap();
                        let remove_id = if existing_version < other_version {
                            // Other version is the newest, remove the existing one and replace
                            id_tracker.drop(external_id)?;
                            id_tracker.set_link(external_id, new_internal_id)?;
                            id_tracker.set_internal_version(new_internal_id, other_version)?;
                            self.payload_storage.drop(existing_internal_id)?;
                            let other_payload = other_payload_index.payload(old_internal_id)?;
                            // Propagate payload to new segment
                            if !other_payload.is_empty() {
                                self.payload_storage
                                    .assign(new_internal_id, &other_payload)?;
                            }
                            existing_internal_id
                        } else {
                            // Old version is still good, do not move anything else
                            // Mark newly added vector as removed
                            new_internal_id
                        };
                        for vector_storage in self.vector_storages.values_mut() {
                            vector_storage.delete_vector(remove_id)?;
                        }
                    }
                }
            }
        }

        for (field, payload_schema) in other.payload_index.borrow().indexed_fields() {
            self.indexed_fields.insert(field, payload_schema);
        }

        Ok(true)
    }

    pub fn build(
        mut self,
        permit: CpuPermit,
        stopped: &AtomicBool,
    ) -> Result<Segment, OperationError> {
        {
            // Arc permit to share it with each vector store
            let permit = Arc::new(permit);

            for (field, payload_schema) in &self.indexed_fields {
                segment.create_field_index(segment.version(), field, Some(payload_schema))?;
                check_process_stopped(stopped)?;
            }

            Self::update_quantization(&mut segment, stopped)?;

            for vector_data in segment.vector_data.values_mut() {
                vector_data
                    .vector_index
                    .borrow_mut()
                    .build_index(permit.clone(), stopped)?;
            }

            // We're done with CPU-intensive tasks, release CPU permit
            debug_assert_eq!(
                Arc::strong_count(&permit),
                1,
                "Must release CPU permit Arc everywhere",
            );
            drop(permit);

            segment.flush(true)?;
            drop(segment);
            // Now segment is evicted from RAM
        }

        // Move fully constructed segment into collection directory and load back to RAM
        std::fs::rename(&self.temp_path, &self.destination_path)
            .describe("Moving segment data after optimization")?;

        let loaded_segment = load_segment(&self.destination_path, stopped)?.ok_or_else(|| {
            OperationError::service_error(format!(
                "Segment loading error: {}",
                self.destination_path.display()
            ))
        })?;
        Ok(loaded_segment)
    }

    fn update_quantization(segment: &mut Segment, stopped: &AtomicBool) -> OperationResult<()> {
        let config = segment.config().clone();

        for (vector_name, vector_data) in &mut segment.vector_data {
            let max_threads = if let Some(config) = config.vector_data.get(vector_name) {
                match &config.index {
                    Indexes::Hnsw(hnsw) => num_rayon_threads(hnsw.max_indexing_threads),
                    _ => 1,
                }
            } else {
                // quantization is applied only for dense vectors
                continue;
            };

            if let Some(quantization) = config.quantization_config(vector_name) {
                let segment_path = segment.current_path.as_path();
                check_process_stopped(stopped)?;

                let vector_storage_path = get_vector_storage_path(segment_path, vector_name);

                let vector_storage = vector_data.vector_storage.borrow();

                let quantized_vectors = QuantizedVectors::create(
                    &vector_storage,
                    quantization,
                    &vector_storage_path,
                    max_threads,
                    stopped,
                )?;

                *vector_data.quantized_vectors.borrow_mut() = Some(quantized_vectors);
            }
        }
        Ok(())
    }
}
