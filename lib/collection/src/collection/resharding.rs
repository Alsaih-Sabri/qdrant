use std::path::PathBuf;
use std::sync::Arc;

use futures::Future;
use parking_lot::Mutex;

use super::Collection;
use crate::config::ShardingMethod;
use crate::operations::types::CollectionResult;
use crate::shards::replica_set::ReplicaState;
use crate::shards::resharding::tasks_pool::{ReshardTaskItem, ReshardTaskProgress};
use crate::shards::resharding::{self, ReshardKey, ReshardState};
use crate::shards::transfer::ShardTransferConsensus;

impl Collection {
    pub async fn resharding_state(&self) -> Option<ReshardState> {
        self.shards_holder
            .read()
            .await
            .resharding_state
            .read()
            .clone()
    }

    /// Start a new resharding operation
    pub async fn start_resharding<T, F>(
        &self,
        resharding_key: ReshardKey,
        consensus: Box<dyn ShardTransferConsensus>,
        temp_dir: PathBuf,
        on_finish: T,
        on_error: F,
    ) -> CollectionResult<()>
    where
        T: Future<Output = ()> + Send + 'static,
        F: Future<Output = ()> + Send + 'static,
    {
        {
            let mut shard_holder = self.shards_holder.write().await;

            shard_holder.check_start_resharding(&resharding_key)?;

            let replica_set = self
                .create_replica_set(
                    resharding_key.shard_id,
                    &[resharding_key.peer_id],
                    Some(ReplicaState::Resharding),
                )
                .await?;

            shard_holder.start_resharding_unchecked(resharding_key.clone(), replica_set)?;
        }

        // Drive resharding
        self.drive_resharding(resharding_key, consensus, temp_dir, on_finish, on_error)
            .await?;

        Ok(())
    }

    /// Resume an existing resharding operation
    ///
    /// This method will check if a resharding operation is in progress according to our state, and
    /// it will start and resume the driving task accordingly.
    ///
    /// This does not check whether the task is already active.
    ///
    /// If no resharding is active, this returns early without error.
    pub async fn resume_resharding_unchecked<T, F>(
        &self,
        // resharding_key: ReshardKey,
        consensus: Box<dyn ShardTransferConsensus>,
        temp_dir: PathBuf,
        on_finish: T,
        on_error: F,
    ) -> CollectionResult<()>
    where
        T: Future<Output = ()> + Send + 'static,
        F: Future<Output = ()> + Send + 'static,
    {
        let Some(state) = self.resharding_state().await else {
            return Ok(());
        };

        self.drive_resharding(state.key(), consensus, temp_dir, on_finish, on_error)
            .await?;

        Ok(())
    }

    async fn drive_resharding<T, F>(
        &self,
        resharding_key: ReshardKey,
        consensus: Box<dyn ShardTransferConsensus>,
        temp_dir: PathBuf,
        on_finish: T,
        on_error: F,
    ) -> CollectionResult<()>
    where
        T: Future<Output = ()> + Send + 'static,
        F: Future<Output = ()> + Send + 'static,
    {
        // Skip if this peer is not responsible for driving the resharding
        if resharding_key.peer_id != self.this_peer_id {
            return Ok(());
        }

        // Stop any already active resharding task to allow starting a new one
        let mut active_reshard_tasks = self.reshard_tasks.lock().await;
        let task_result = active_reshard_tasks.stop_task(&resharding_key).await;
        debug_assert!(task_result.is_none(), "Reshard task already exists");

        let shard_holder = self.shards_holder.clone();
        let collection_id = self.id.clone();
        let collection_config = Arc::clone(&self.collection_config);
        let channel_service = self.channel_service.clone();
        let progress = Arc::new(Mutex::new(ReshardTaskProgress::new()));
        let spawned_task = resharding::spawn_resharding_task(
            shard_holder,
            progress.clone(),
            resharding_key.clone(),
            consensus,
            collection_id,
            self.path.clone(),
            collection_config,
            self.shared_storage_config.clone(),
            channel_service,
            temp_dir,
            on_finish,
            on_error,
        );

        active_reshard_tasks.add_task(
            resharding_key,
            ReshardTaskItem {
                task: spawned_task,
                started_at: chrono::Utc::now(),
                progress,
            },
        );

        Ok(())
    }

    pub async fn commit_read_hashring(&self, resharding_key: ReshardKey) -> CollectionResult<()> {
        self.shards_holder
            .write()
            .await
            .commit_read_hashring(resharding_key)
    }

    pub async fn commit_write_hashring(&self, resharding_key: ReshardKey) -> CollectionResult<()> {
        self.shards_holder
            .write()
            .await
            .commit_write_hashring(resharding_key)
    }

    pub async fn finish_resharding(&self, resharding_key: ReshardKey) -> CollectionResult<()> {
        let mut shard_holder = self.shards_holder.write().await;

        shard_holder.check_finish_resharding(&resharding_key)?;
        let _ = self.stop_resharding_task(&resharding_key).await;
        shard_holder.finish_resharding_unchecked(&resharding_key)?;

        // Update the stored shard count, ensures we load the new shard on restart
        {
            let mut config = self.collection_config.write().await;
            config.params.shard_number = config.params.shard_number.saturating_add(1);

            // If using automatic sharding, the reshard key shard ID must correspond with number
            if config.params.sharding_method.unwrap_or_default() == ShardingMethod::Auto {
                debug_assert_eq!(
                    config.params.shard_number.get() - 1,
                    resharding_key.shard_id,
                );
            }

            if let Err(err) = config.save(&self.path) {
                log::error!("Failed to update and save collection config during resharding: {err}");
            }
        }

        Ok(())
    }

    pub async fn abort_resharding(&self, resharding_key: ReshardKey) -> CollectionResult<()> {
        let _ = self.stop_resharding_task(&resharding_key).await;

        self.shards_holder
            .write()
            .await
            .abort_resharding(resharding_key)
            .await?;

        Ok(())
    }

    async fn stop_resharding_task(
        &self,
        resharding_key: &ReshardKey,
    ) -> Option<resharding::tasks_pool::TaskResult> {
        self.reshard_tasks
            .lock()
            .await
            .stop_task(resharding_key)
            .await
    }
}
