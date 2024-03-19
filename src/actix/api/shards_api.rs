use actix_web::{post, put, web, Responder};
use actix_web_validator::{Json, Path, Query};
use collection::operations::cluster_ops::{
    ClusterOperations, CreateShardingKey, CreateShardingKeyOperation, DropShardingKey,
    DropShardingKeyOperation,
};
use rbac::jwt::Claims;
use storage::dispatcher::Dispatcher;
use tokio::time::Instant;

use crate::actix::api::collections_api::WaitTimeout;
use crate::actix::api::CollectionPath;
use crate::actix::auth::Extension;
use crate::actix::helpers::process_response;
use crate::common::collections::do_update_collection_cluster;

// ToDo: introduce API for listing shard keys

#[put("/collections/{name}/shards")]
async fn create_shard_key(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: Json<CreateShardingKey>,
    Query(query): Query<WaitTimeout>,
    claims: Extension<Claims>,
) -> impl Responder {
    let timing = Instant::now();
    let wait_timeout = query.timeout();
    let dispatcher = dispatcher.into_inner();

    let request = request.into_inner();

    let operation = ClusterOperations::CreateShardingKey(CreateShardingKeyOperation {
        create_sharding_key: request,
    });

    let response = do_update_collection_cluster(
        &dispatcher,
        collection.name.clone(),
        operation,
        claims.into_inner(),
        wait_timeout,
    )
    .await;

    process_response(response, timing)
}

#[post("/collections/{name}/shards/delete")]
async fn delete_shard_key(
    dispatcher: web::Data<Dispatcher>,
    collection: Path<CollectionPath>,
    request: Json<DropShardingKey>,
    Query(query): Query<WaitTimeout>,
    claims: Extension<Claims>,
) -> impl Responder {
    let timing = Instant::now();
    let wait_timeout = query.timeout();

    let dispatcher = dispatcher.into_inner();
    let request = request.into_inner();

    let operation = ClusterOperations::DropShardingKey(DropShardingKeyOperation {
        drop_sharding_key: request,
    });

    let response = do_update_collection_cluster(
        &dispatcher,
        collection.name.clone(),
        operation,
        claims.into_inner(),
        wait_timeout,
    )
    .await;

    process_response(response, timing)
}

pub fn config_shards_api(cfg: &mut web::ServiceConfig) {
    cfg.service(create_shard_key).service(delete_shard_key);
}
