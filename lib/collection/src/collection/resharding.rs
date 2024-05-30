use std::path::PathBuf;
use std::sync::Arc;

use futures::Future;
use parking_lot::Mutex;

use super::Collection;
use crate::operations::types::CollectionResult;
use crate::shards::replica_set::ReplicaState;
use crate::shards::resharding::tasks_pool::{ReshardTaskItem, ReshardTaskProgress};
use crate::shards::resharding::{self, ReshardKey, ReshardState, ReshardTask};
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

    pub async fn start_resharding<T, F>(
        &self,
        reshard_task: ReshardTask,
        consensus: Box<dyn ShardTransferConsensus>,
        temp_dir: PathBuf,
        on_finish: T,
        on_error: F,
    ) -> CollectionResult<()>
    where
        T: Future<Output = ()> + Send + 'static,
        F: Future<Output = ()> + Send + 'static,
    {
        let mut shard_holder = self.shards_holder.write().await;

        let reshard_key = reshard_task.key();
        shard_holder.check_start_resharding(&reshard_key)?;

        let replica_set = self
            .create_replica_set(
                reshard_task.shard_id,
                &[reshard_task.peer_id],
                Some(ReplicaState::Resharding),
            )
            .await?;

        shard_holder.start_resharding_unchecked(reshard_key.clone(), replica_set)?;

        // TODO: ----------------------------------------------------------------------------------
        // TODO(resharding): start and drive functions seem similar, can we merge them?
        // TODO(resharding): clearly distinguish between resharding state and active task

        let mut active_reshard_tasks = self.reshard_tasks.lock().await;
        let task_result = active_reshard_tasks.stop_task(&reshard_key).await;
        debug_assert!(task_result.is_none(), "Reshard task already exists");

        let shard_holder = self.shards_holder.clone();
        let collection_id = self.id.clone();
        let channel_service = self.channel_service.clone();

        let progress = Arc::new(Mutex::new(ReshardTaskProgress::new()));

        let spawned_task = resharding::spawn_resharding_task(
            shard_holder,
            progress.clone(),
            reshard_task.clone(),
            consensus,
            collection_id,
            channel_service,
            self.name(),
            temp_dir,
            on_finish,
            on_error,
        );

        active_reshard_tasks.add_task(
            &reshard_task,
            ReshardTaskItem {
                task: spawned_task,
                started_at: chrono::Utc::now(),
                progress,
            },
        );

        Ok(())
    }

    pub async fn abort_resharding(&self, reshard_key: ReshardKey) -> CollectionResult<()> {
        self.shards_holder
            .write()
            .await
            .abort_resharding(reshard_key)
            .await
    }
}
