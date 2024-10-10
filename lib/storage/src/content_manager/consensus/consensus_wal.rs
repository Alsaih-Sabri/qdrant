use std::path::Path;
use std::{cmp, fs};

use prost_for_raft::Message;
use protobuf::Message as _;
use raft::eraftpb::Entry as RaftEntry;
use wal::Wal;

use crate::content_manager::consensus_manager;
use crate::content_manager::consensus_ops::ConsensusOperations;
use crate::StorageError;

const COLLECTIONS_META_WAL_DIR: &str = "collections_meta_wal";

#[derive(Debug)]
pub struct ConsensusOpWal {
    wal: Wal,
}

impl ConsensusOpWal {
    pub fn new(storage_path: &str) -> Self {
        let collections_meta_wal_path = Path::new(storage_path).join(COLLECTIONS_META_WAL_DIR);

        fs::create_dir_all(&collections_meta_wal_path)
            .expect("Can't create consensus WAL directory");

        let wal = Wal::open(collections_meta_wal_path).expect("Can't open consensus WAL");
        Self { wal }
    }

    pub fn clear(&mut self) -> Result<(), StorageError> {
        self.wal.clear()?;
        Ok(())
    }

    pub fn entry(&self, raft_index: u64) -> raft::Result<RaftEntry> {
        // Raft entries are expected to have index starting from 1
        if raft_index < 1 {
            return Err(raft::Error::Store(raft::StorageError::Unavailable));
        }

        let wal_index = self
            .index_offset()?
            .try_raft_to_wal(raft_index)
            .ok_or_else(|| raft::Error::Store(raft::StorageError::Compacted))?;

        self.entry_by_wal_index(wal_index)
    }

    pub fn entries(
        &self,
        from_raft_index: u64,
        until_raft_index: u64,
        max_size_bytes: Option<u64>,
    ) -> raft::Result<Vec<RaftEntry>> {
        let offset = self.index_offset()?;

        // Map requested Raft indices to WAL indices
        let from_wal_index = offset.raft_to_wal(from_raft_index);
        let until_wal_index = offset.raft_to_wal(until_raft_index);

        // Bound mapped WAL indices between first/last WAL indices
        let from_wal_index = cmp::max(from_wal_index, offset.wal_index);
        let until_wal_index = cmp::min(until_wal_index, offset.wal_index + self.wal.num_entries());

        // `Some(u64::MAX)` and `None` are treated as "no max size"
        // `Some(0)` is treated as "return single entry"
        //
        // See:
        // - https://docs.rs/raft/latest/raft/storage/trait.Storage.html#tymethod.entries
        // - https://github.com/tikv/raft-rs/blob/v0.7.0/src/util.rs#L56-L71
        let max_size_bytes = match max_size_bytes {
            Some(u64::MAX) | None => None,
            Some(max_size_bytes) => Some(max_size_bytes),
        };

        let mut size_bytes = 0;
        let mut entries = Vec::with_capacity(until_wal_index.saturating_sub(from_wal_index) as _);

        for wal_index in from_wal_index..until_wal_index {
            let entry = self.entry_by_wal_index(wal_index)?;

            size_bytes += u64::from(entry.compute_size());
            entries.push(entry);

            if let Some(max_size_bytes) = max_size_bytes {
                if size_bytes >= max_size_bytes {
                    break;
                }
            }
        }

        Ok(entries)
    }

    pub fn first_entry(&self) -> Result<Option<RaftEntry>, StorageError> {
        self.entry_by_wal_index_impl(self.wal.first_index())
    }

    pub fn last_entry(&self) -> Result<Option<RaftEntry>, StorageError> {
        self.entry_by_wal_index_impl(self.wal.last_index())
    }

    pub fn append_entries(&mut self, new_entries: Vec<RaftEntry>) -> Result<(), StorageError> {
        if new_entries.is_empty() {
            return Ok(());
        }

        // Calculate WAL to Raft index offset
        let mut current_index_offset = self.index_offset_impl()?;

        // Use single buffer to encode all new entries to reduce allocations
        let mut buf = Vec::new();

        for new_entry in new_entries {
            // If WAL is not empty, check that new entry index is within WAL bounds
            if let Some(offset) = current_index_offset {
                // Check that new entry index was not already compacted (it's not less than first WAL index)
                let Some(new_entry_wal_index) = offset.try_raft_to_wal(new_entry.index) else {
                    return Err(StorageError::service_error(format!(
                        "Can't append entry with Raft index {}, \
                         because WAL is already compacted at Raft index {}",
                        new_entry.index, offset.raft_index,
                    )));
                };

                let next_wal_index = self.wal.last_index() + 1;

                // Check that new entry index is sequential (it's not greater than next WAL index),
                // or truncate entries at the tail of WAL, if it overwrites some
                if new_entry_wal_index > next_wal_index {
                    return Err(StorageError::service_error(format!(
                        "Can't append entry with Raft index {} (expected WAL index {}), \
                         because last entry in WAL is at WAL index {}, \
                         and all entries have to be sequential",
                        new_entry.index,
                        new_entry_wal_index,
                        self.wal.last_index(),
                    )));
                } else if new_entry_wal_index < next_wal_index {
                    log::debug!(
                        "Truncating conflicting WAL entries from Raft index {} \
                         (WAL index {new_entry_wal_index})",
                        new_entry.index,
                    );

                    self.wal.truncate(new_entry_wal_index)?;
                }
            }

            if log::log_enabled!(log::Level::Debug) {
                if let Ok(op) = ConsensusOperations::try_from(&new_entry) {
                    log::debug!(
                        "Appending operation, term: {}, index: {}, entry: {op:?}",
                        new_entry.term,
                        new_entry.index,
                    );
                } else {
                    log::debug!("Appending entry: {new_entry:?}");
                }
            }

            buf.clear();
            new_entry.encode(&mut buf)?;

            let new_entry_wal_index = self.wal.append(&buf)?;

            #[cfg(debug_assertions)]
            {
                // Assert that we calculated indices (and truncated WAL) correctly, and new entry
                // was inserted at expected WAL index

                let expected_new_entry_wal_index = current_index_offset
                    .map_or(Some(0), |offset| offset.try_raft_to_wal(new_entry.index))
                    .expect("new entry can't overwrite already compacted WAL entries");

                debug_assert_eq!(
                    new_entry_wal_index, expected_new_entry_wal_index,
                    "WAL index of inserted entry does not match its expected WAL index, \
                     Raft index: {}, inserted at WAL index: {}, expected WAL index: {}",
                    new_entry.index, new_entry_wal_index, expected_new_entry_wal_index,
                );
            }

            // Calculate WAL to Raft index offset, if we inserted first entry into empty WAL
            if current_index_offset.is_none() {
                current_index_offset = self.index_offset_impl()?;
            }
        }

        // Flush WAL to disk
        self.wal.flush_open_segment()?;

        Ok(())
    }

    pub fn index_offset(&self) -> raft::Result<IndexOffset> {
        let res = self.index_offset_impl();
        into_raft_result(res)
    }

    pub fn index_offset_impl(&self) -> Result<Option<IndexOffset>, StorageError> {
        let wal_index = self.wal.first_index();

        let raft_index = self
            .entry_by_wal_index_impl(wal_index)?
            .map(|entry| entry.index);

        let Some(raft_index) = raft_index else {
            return Ok(None);
        };

        // WAL index of an entry should always be *less* than its Raft index, but this is already
        // asserted in `entry_by_wal_index_impl`
        let wal_to_raft_offset = raft_index - wal_index;

        let offset = IndexOffset {
            wal_index,
            raft_index,
            wal_to_raft_offset,
        };

        Ok(Some(offset))
    }

    fn entry_by_wal_index(&self, wal_index: u64) -> raft::Result<RaftEntry> {
        let res = self.entry_by_wal_index_impl(wal_index);
        into_raft_result(res)
    }

    fn entry_by_wal_index_impl(&self, wal_index: u64) -> Result<Option<RaftEntry>, StorageError> {
        let entry: Option<RaftEntry> = self
            .wal
            .entry(wal_index)
            .map(|entry| prost_for_raft::Message::decode(entry.as_ref()))
            .transpose()?;

        if let Some(entry) = &entry {
            // WAL index of an entry should always be *less* than its Raft index.
            //
            // - WAL index starts with 0
            // - Raft index starts with 1
            // - if WAL is compacted, then difference between Raft index and WAL index may be even greater
            // - Raft indices in the WAL should always be *sequential*
            //
            // So, if at any point WAL index of an entry is *greater-or-equal* than its Raft index,
            // it's an error in `ConsensusOpWal` logic.

            debug_assert!(
                wal_index < entry.index,
                "WAL index of an entry is greater than (or equal to) its Raft index, \
                 WAL index: {wal_index}, Raft index: {}",
                entry.index,
            );
        }

        Ok(entry)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct IndexOffset {
    pub wal_index: u64,
    pub raft_index: u64,
    pub wal_to_raft_offset: u64,
}

impl IndexOffset {
    pub fn try_raft_to_wal(&self, raft_index: u64) -> Option<u64> {
        raft_index.checked_sub(self.wal_to_raft_offset)
    }

    pub fn raft_to_wal(&self, raft_index: u64) -> u64 {
        raft_index.saturating_sub(self.wal_to_raft_offset)
    }

    pub fn wal_to_raft(&self, wal_index: u64) -> u64 {
        wal_index + self.wal_to_raft_offset
    }
}

fn into_raft_result<T>(result: Result<Option<T>, StorageError>) -> raft::Result<T> {
    result
        .map_err(consensus_manager::raft_error_other)?
        .ok_or(raft::Error::Store(raft::StorageError::Unavailable))
}

#[cfg(test)]
mod tests {
    use raft::eraftpb::Entry;

    use super::*;

    fn init_logger() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_log_compaction_rewrite() {
        init_logger();
        let entries_orig = vec![
            Entry {
                entry_type: 0,
                term: 1,
                index: 1,
                data: vec![1, 2, 3],
                context: vec![],
                sync_log: false,
            },
            Entry {
                entry_type: 0,
                term: 1,
                index: 2,
                data: vec![1, 2, 3],
                context: vec![],
                sync_log: false,
            },
            Entry {
                entry_type: 0,
                term: 1,
                index: 3,
                data: vec![1, 2, 3],
                context: vec![],
                sync_log: false,
            },
        ];

        let entries_new = vec![
            Entry {
                entry_type: 0,
                term: 1,
                index: 2,
                data: vec![2, 2, 2],
                context: vec![],
                sync_log: false,
            },
            Entry {
                entry_type: 0,
                term: 1,
                index: 3,
                data: vec![3, 3, 3],
                context: vec![],
                sync_log: false,
            },
        ];

        let temp_dir = tempfile::tempdir().unwrap();

        let mut wal = ConsensusOpWal::new(temp_dir.path().to_str().unwrap());
        wal.append_entries(entries_orig).unwrap();
        wal.append_entries(entries_new.clone()).unwrap();

        let result_entries = wal.entries(1, 4, None).unwrap();
        assert_eq!(result_entries.len(), 3);

        assert_eq!(result_entries[0].data, vec![1, 2, 3]);
        assert_eq!(result_entries[1].data, vec![2, 2, 2]);
        assert_eq!(result_entries[2].data, vec![3, 3, 3]);

        wal.clear().unwrap();
        wal.append_entries(entries_new).unwrap();
        assert_eq!(wal.index_offset().unwrap().wal_to_raft_offset, 2);

        let broken_entry = vec![Entry {
            entry_type: 0,
            term: 1,
            index: 1, // Index 1 can't be overwritten, because it is already compacted
            data: vec![5, 5, 5],
            context: vec![],
            sync_log: false,
        }];

        // Some errors can't be corrected
        assert!(matches!(
            wal.append_entries(broken_entry),
            Err(StorageError::ServiceError { .. })
        ));
    }

    #[test]
    fn test_log_rewrite() {
        init_logger();
        let entries_orig = vec![
            Entry {
                entry_type: 0,
                term: 1,
                index: 1,
                data: vec![1, 1, 1],
                context: vec![],
                sync_log: false,
            },
            Entry {
                entry_type: 0,
                term: 1,
                index: 2,
                data: vec![1, 1, 1],
                context: vec![],
                sync_log: false,
            },
            Entry {
                entry_type: 0,
                term: 1,
                index: 3,
                data: vec![1, 1, 1],
                context: vec![],
                sync_log: false,
            },
        ];

        let entries_new = vec![
            Entry {
                entry_type: 0,
                term: 1,
                index: 2,
                data: vec![2, 2, 2],
                context: vec![],
                sync_log: false,
            },
            Entry {
                entry_type: 0,
                term: 1,
                index: 3,
                data: vec![2, 2, 2],
                context: vec![],
                sync_log: false,
            },
            Entry {
                entry_type: 0,
                term: 1,
                index: 4,
                data: vec![2, 2, 2],
                context: vec![],
                sync_log: false,
            },
        ];

        let temp_dir = tempfile::tempdir().unwrap();
        let mut wal = ConsensusOpWal::new(temp_dir.path().to_str().unwrap());

        // append original entries
        wal.append_entries(entries_orig).unwrap();
        assert_eq!(wal.wal.num_segments(), 1);
        assert_eq!(wal.wal.num_entries(), 3);
        assert_eq!(wal.index_offset().unwrap().wal_to_raft_offset, 1);
        assert_eq!(wal.first_entry().unwrap().unwrap().index, 1);
        assert_eq!(wal.last_entry().unwrap().unwrap().index, 3);

        let result_entries = wal.entries(1, 4, None).unwrap();
        assert_eq!(result_entries.len(), 3);
        assert_eq!(result_entries[0].data, vec![1, 1, 1]);
        assert_eq!(result_entries[1].data, vec![1, 1, 1]);
        assert_eq!(result_entries[2].data, vec![1, 1, 1]);

        // drop wal to check persistence
        drop(wal);
        let mut wal = ConsensusOpWal::new(temp_dir.path().to_str().unwrap());

        // append overlapping entries
        wal.append_entries(entries_new).unwrap();
        assert_eq!(wal.wal.num_segments(), 1);
        assert_eq!(wal.wal.num_entries(), 4);
        assert_eq!(wal.index_offset().unwrap().wal_to_raft_offset, 1);
        assert_eq!(wal.first_entry().unwrap().unwrap().index, 1);
        assert_eq!(wal.last_entry().unwrap().unwrap().index, 4);

        let result_entries = wal.entries(1, 5, None).unwrap();
        assert_eq!(result_entries.len(), 4);
        assert_eq!(result_entries[0].data, vec![1, 1, 1]); // survived the truncation
        assert_eq!(result_entries[1].data, vec![2, 2, 2]);
        assert_eq!(result_entries[2].data, vec![2, 2, 2]);
        assert_eq!(result_entries[3].data, vec![2, 2, 2]);

        // drop wal to check persistence
        drop(wal);
        let wal = ConsensusOpWal::new(temp_dir.path().to_str().unwrap());
        assert_eq!(wal.wal.num_segments(), 1);
        assert_eq!(wal.wal.num_entries(), 4);
        assert_eq!(wal.index_offset().unwrap().wal_to_raft_offset, 1);
        assert_eq!(wal.first_entry().unwrap().unwrap().index, 1);
        assert_eq!(wal.last_entry().unwrap().unwrap().index, 4);
    }
    #[test]
    fn test_log_rewrite_last() {
        init_logger();
        let entries_orig = vec![
            Entry {
                entry_type: 0,
                term: 1,
                index: 1,
                data: vec![1, 1, 1],
                context: vec![],
                sync_log: false,
            },
            Entry {
                entry_type: 0,
                term: 1,
                index: 2,
                data: vec![1, 1, 1],
                context: vec![],
                sync_log: false,
            },
            Entry {
                entry_type: 0,
                term: 1,
                index: 3,
                data: vec![1, 1, 1],
                context: vec![],
                sync_log: false,
            },
        ];

        // change only the last entry
        let entries_new = vec![Entry {
            entry_type: 0,
            term: 1,
            index: 3,
            data: vec![2, 2, 2],
            context: vec![],
            sync_log: false,
        }];

        let temp_dir = tempfile::tempdir().unwrap();
        let mut wal = ConsensusOpWal::new(temp_dir.path().to_str().unwrap());

        // append original entries
        wal.append_entries(entries_orig).unwrap();
        assert_eq!(wal.wal.num_segments(), 1);
        assert_eq!(wal.wal.num_entries(), 3);
        assert_eq!(wal.index_offset().unwrap().wal_to_raft_offset, 1);
        assert_eq!(wal.first_entry().unwrap().unwrap().index, 1);
        assert_eq!(wal.last_entry().unwrap().unwrap().index, 3);

        let result_entries = wal.entries(1, 4, None).unwrap();
        assert_eq!(result_entries.len(), 3);
        assert_eq!(result_entries[0].data, vec![1, 1, 1]);
        assert_eq!(result_entries[1].data, vec![1, 1, 1]);
        assert_eq!(result_entries[2].data, vec![1, 1, 1]);

        // drop wal to check persistence
        drop(wal);
        let mut wal = ConsensusOpWal::new(temp_dir.path().to_str().unwrap());

        // append overlapping entries
        wal.append_entries(entries_new).unwrap();
        assert_eq!(wal.wal.num_segments(), 1);
        assert_eq!(wal.wal.num_entries(), 3);
        assert_eq!(wal.index_offset().unwrap().wal_to_raft_offset, 1);
        assert_eq!(wal.first_entry().unwrap().unwrap().index, 1);
        assert_eq!(wal.last_entry().unwrap().unwrap().index, 3);

        let result_entries = wal.entries(1, 4, None).unwrap();
        assert_eq!(result_entries.len(), 3);
        assert_eq!(result_entries[0].data, vec![1, 1, 1]);
        assert_eq!(result_entries[1].data, vec![1, 1, 1]);
        assert_eq!(result_entries[2].data, vec![2, 2, 2]); // value updated

        // drop wal to check persistence
        drop(wal);
        let wal = ConsensusOpWal::new(temp_dir.path().to_str().unwrap());
        assert_eq!(wal.wal.num_segments(), 1);
        assert_eq!(wal.wal.num_entries(), 3);
        assert_eq!(wal.index_offset().unwrap().wal_to_raft_offset, 1);
        assert_eq!(wal.first_entry().unwrap().unwrap().index, 1);
        assert_eq!(wal.last_entry().unwrap().unwrap().index, 3);
    }
}
