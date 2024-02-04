use std::path::PathBuf;

use super::file::SnapshotFile;
use super::SnapshotManager;
use crate::error::SnapshotManagerError;

impl SnapshotManager {
    pub(super) fn snapshots_path(&self) -> PathBuf {
        self.0.path.clone()
    }

    pub fn temp_path(&self) -> PathBuf {
        self.snapshots_path().join("tmp")
    }

    pub(super) async fn snapshot_exists(
        &self,
        snapshot: &SnapshotFile,
    ) -> Result<bool, SnapshotManagerError> {
        Ok(snapshot.get_path(self.snapshots_path()).exists())
    }

    /// Removes the specified snapshot and checksum files.
    pub(super) async fn remove_snapshot(
        &self,
        snapshot: &SnapshotFile,
    ) -> Result<(), SnapshotManagerError> {
        if let Some(collection) = &snapshot.collection() {
            log::info!(
                "Deleting collection {:?} snapshot {:?}",
                collection,
                snapshot.name()
            );
        } else {
            log::info!("Deleting full storage snapshot {:?}", snapshot.name());
        }

        let base = self.snapshots_path();
        let snapshot_path = snapshot.get_path(&base);
        let checksum_path = snapshot.get_checksum_path(base);

        let (delete_snapshot, delete_checksum) = tokio::join!(
            tokio::fs::remove_file(snapshot_path),
            tokio::fs::remove_file(checksum_path),
        );
        delete_snapshot?;

        // We might not have a checksum file for the snapshot, ignore deletion errors in that case
        if let Err(err) = delete_checksum {
            log::warn!("Failed to delete checksum file for snapshot, ignoring: {err}");
        }

        Ok(())
    }

    pub(super) async fn list_snapshots_in_directory(
        &self,
        directory: impl Into<PathBuf>,
    ) -> Result<Vec<String>, SnapshotManagerError> {
        let directory: PathBuf = directory.into();

        if !directory.exists() {
            return Ok(vec![]);
        }

        let mut entries = tokio::fs::read_dir(directory).await?;
        let mut snapshots = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if !path.is_dir() && path.extension().map_or(false, |ext| ext == "snapshot") {
                if let Some(file_name) = path.file_name() {
                    snapshots.push(file_name.to_string_lossy().to_string());
                }
            }
        }

        Ok(snapshots)
    }
}
