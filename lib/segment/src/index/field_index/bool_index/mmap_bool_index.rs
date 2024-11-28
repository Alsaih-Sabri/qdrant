use std::ops::DerefMut;
use std::path::{Path, PathBuf};

use common::types::PointOffsetType;
use memory::madvise::AdviceSetting;
use memory::mmap_ops::{self};
use memory::mmap_type::MmapType;

use crate::common::error_logging::LogError;
use crate::common::operation_error::{OperationError, OperationResult};
use crate::index::field_index::{
    CardinalityEstimation, FieldIndexBuilderTrait, PayloadBlockCondition, PayloadFieldIndex,
    PrimaryCondition, ValueIndexer,
};
use crate::types::{FieldCondition, Match, MatchValue, PayloadKeyType, ValueVariants};
use crate::vector_storage::dense::dynamic_mmap_flags::DynamicMmapFlags;

const BASE_DIR_PREFIX: &str = "bool-";
const METADATA_FILE: &str = "metadata.dat";
const TRUES_DIRNAME: &str = "trues";
const FALSES_DIRNAME: &str = "falses";

/// When bitslices are extended to fit a certain id, they get this much extra space to avoid resizing too often.
const BITSLICE_GROWTH_SLACK: usize = 1028;

#[repr(C)]
struct BoolIndexMetadata {
    /// The amount of points which have either a true or false value
    indexed_count: PointOffsetType,

    /// Reserved space for extensibility
    _reserved: [u8; 128],
}

/// Payload index for boolean values, stored in memory-mapped files.
pub struct MmapBoolIndex {
    base_dir: PathBuf,
    metadata: MmapType<BoolIndexMetadata>,
    trues_slice: DynamicMmapFlags,
    falses_slice: DynamicMmapFlags,
}

impl MmapBoolIndex {
    /// Returns the directory name for the given field name, where the files should live.
    fn dirname(field_name: &str) -> String {
        format!("{BASE_DIR_PREFIX}{field_name}")
    }

    /// Creates a new boolean index at the given path. If it already exists, loads the index.
    ///
    /// # Arguments
    /// - `parent_path`: The path to the directory containing the rest of the indexes.
    pub fn open_or_create(indexes_path: &Path, field_name: String) -> OperationResult<Self> {
        let path = indexes_path.join(Self::dirname(&field_name));

        let header_path = path.join(METADATA_FILE);
        if header_path.exists() {
            Self::open(&path)
        } else {
            std::fs::create_dir_all(&path).map_err(|err| {
                OperationError::service_error(format!(
                    "Failed to create mmap bool index directory: {err}"
                ))
            })?;
            Self::create(&path)
        }
    }

    fn create(path: &Path) -> OperationResult<Self> {
        if !path.is_dir() {
            return Err(OperationError::service_error("Path is not a directory"));
        }

        // Trues bitslice
        let trues_path = path.join(TRUES_DIRNAME);
        let trues_slice = DynamicMmapFlags::open(&trues_path)?;

        // Falses bitslice
        let falses_path = path.join(FALSES_DIRNAME);
        let falses_slice = DynamicMmapFlags::open(&falses_path)?;

        // Metadata
        let meta_path = path.join(METADATA_FILE);
        mmap_ops::create_and_ensure_length(&meta_path, size_of::<BoolIndexMetadata>())?;
        let meta_mmap = mmap_ops::open_write_mmap(&meta_path, AdviceSetting::Global, false)?;
        let mut metadata: MmapType<BoolIndexMetadata> = unsafe { MmapType::try_from(meta_mmap)? };

        *metadata.deref_mut() = BoolIndexMetadata {
            indexed_count: 0,
            _reserved: [0; 128],
        };

        Ok(Self {
            base_dir: path.to_path_buf(),
            trues_slice,
            falses_slice,
            metadata,
        })
    }

    fn open(path: &Path) -> OperationResult<Self> {
        // Trues bitslice
        let trues_path = path.join(TRUES_DIRNAME);
        let trues_slice = DynamicMmapFlags::open(&trues_path)?;

        // Falses bitslice
        let falses_path = path.join(FALSES_DIRNAME);
        let falses_slice = DynamicMmapFlags::open(&falses_path)?;

        // Metadata
        let meta_path = path.join(METADATA_FILE);
        let meta_mmap = mmap_ops::open_write_mmap(&meta_path, AdviceSetting::Global, false)
            .describe("Open header mmap for writing")?;
        let metadata = unsafe { MmapType::try_from(meta_mmap)? };

        Ok(Self {
            base_dir: path.to_path_buf(),
            metadata,
            trues_slice,
            falses_slice,
        })
    }

    fn set_or_insert(&mut self, id: u32, has_true: bool, has_false: bool) -> OperationResult<()> {
        let prev_true = set_or_insert_flag(&mut self.trues_slice, id as usize, has_true)?;
        let prev_false = set_or_insert_flag(&mut self.falses_slice, id as usize, has_false)?;

        let was_indexed = prev_true || prev_false;
        let gets_indexed = has_true || has_false;

        match (was_indexed, gets_indexed) {
            (false, true) => {
                self.metadata.indexed_count += 1;
            }
            (true, false) => {
                self.metadata.indexed_count = self.metadata.indexed_count.saturating_sub(1);
            }
            _ => {}
        }

        Ok(())
    }
}

/// Set or insert a flag in the given flags. Returns previous value.
///
/// Resizes if needed to set to true.
fn set_or_insert_flag(
    flags: &mut DynamicMmapFlags,
    key: usize,
    value: bool,
) -> OperationResult<bool> {
    // Set to true
    if value {
        // Make sure the key fits
        if key >= flags.len() {
            flags.set_len(key + BITSLICE_GROWTH_SLACK)?;
        }
        return Ok(flags.set(key, value));
    }

    // Set to false
    if key < flags.len() {
        return Ok(flags.set(key, value));
    }

    Ok(false)
}

pub struct BoolIndexBuilder(MmapBoolIndex);

impl FieldIndexBuilderTrait for BoolIndexBuilder {
    type FieldIndexType = MmapBoolIndex;

    fn init(&mut self) -> OperationResult<()> {
        // After Self is created, it is already initialized
        Ok(())
    }

    fn add_point(
        &mut self,
        id: PointOffsetType,
        payload: &[&serde_json::Value],
    ) -> OperationResult<()> {
        self.0.add_point(id, payload)
    }

    fn finalize(self) -> OperationResult<Self::FieldIndexType> {
        Ok(self.0)
    }
}

impl ValueIndexer for MmapBoolIndex {
    type ValueType = bool;

    fn add_many(
        &mut self,
        id: PointOffsetType,
        values: Vec<Self::ValueType>,
    ) -> OperationResult<()> {
        if values.is_empty() {
            return Ok(());
        }

        let has_true = values.iter().any(|v| *v);
        let has_false = values.iter().any(|v| !*v);

        self.set_or_insert(id, has_true, has_false)?;

        Ok(())
    }

    fn get_value(value: &serde_json::Value) -> Option<Self::ValueType> {
        value.as_bool()
    }

    fn remove_point(&mut self, id: PointOffsetType) -> OperationResult<()> {
        self.set_or_insert(id, false, false)?;
        Ok(())
    }
}
impl PayloadFieldIndex for MmapBoolIndex {
    fn count_indexed_points(&self) -> usize {
        self.metadata.indexed_count as usize
    }

    fn load(&mut self) -> OperationResult<bool> {
        // No extra loading needed for mmap
        Ok(true)
    }

    fn cleanup(self) -> OperationResult<()> {
        std::fs::remove_dir_all(self.base_dir)?;
        Ok(())
    }

    fn flusher(&self) -> crate::common::Flusher {
        let Self {
            base_dir: _,
            metadata,
            trues_slice,
            falses_slice,
        } = self;

        let metadata_flusher = metadata.flusher();
        let trues_flusher = trues_slice.flusher();
        let falses_flusher = falses_slice.flusher();

        Box::new(move || {
            metadata_flusher()?;
            trues_flusher()?;
            falses_flusher()?;
            Ok(())
        })
    }

    fn files(&self) -> Vec<std::path::PathBuf> {
        let mut files = self.trues_slice.files();
        files.extend(self.falses_slice.files());
        files.push(self.base_dir.join(METADATA_FILE));

        files
    }

    fn filter<'a>(
        &'a self,
        condition: &'a FieldCondition,
    ) -> Option<Box<dyn Iterator<Item = PointOffsetType> + 'a>> {
        match &condition.r#match {
            Some(Match::Value(MatchValue {
                value: ValueVariants::Bool(value),
            })) => {
                if *value {
                    Some(Box::new(
                        self.trues_slice
                            .get_bitslice()
                            .iter_ones()
                            .map(|x| x as PointOffsetType),
                    ))
                } else {
                    Some(Box::new(
                        self.falses_slice
                            .get_bitslice()
                            .iter_ones()
                            .map(|x| x as PointOffsetType),
                    ))
                }
            }
            _ => None,
        }
    }

    fn estimate_cardinality(&self, condition: &FieldCondition) -> Option<CardinalityEstimation> {
        match &condition.r#match {
            Some(Match::Value(MatchValue {
                value: ValueVariants::Bool(value),
            })) => {
                let count = match *value {
                    true => self.trues_slice.count_flags(),
                    false => self.falses_slice.count_flags(),
                };

                let estimation = CardinalityEstimation::exact(count)
                    .with_primary_clause(PrimaryCondition::Condition(Box::new(condition.clone())));

                Some(estimation)
            }
            _ => None,
        }
    }

    fn payload_blocks(
        &self,
        threshold: usize,
        key: PayloadKeyType,
    ) -> Box<dyn Iterator<Item = PayloadBlockCondition> + '_> {
        let make_block = |count, value, key: PayloadKeyType| {
            if count > threshold {
                Some(PayloadBlockCondition {
                    condition: FieldCondition::new_match(
                        key,
                        Match::Value(MatchValue {
                            value: ValueVariants::Bool(value),
                        }),
                    ),
                    cardinality: count,
                })
            } else {
                None
            }
        };

        // just two possible blocks: true and false
        let iter = [
            make_block(self.trues_slice.count_flags(), true, key.clone()),
            make_block(self.falses_slice.count_flags(), false, key),
        ]
        .into_iter()
        .flatten();

        Box::new(iter)
    }
}
