use crate::actix::helpers::process_response;
use crate::common::points::{do_update_points, do_delete_points, do_set_payload, do_delete_payload, do_upsert_points};
use actix_web::rt::time::Instant;
use actix_web::web::Query;
use actix_web::{post, web, Responder};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use collection::operations::CollectionUpdateOperations;
use collection::operations::payload_ops::{DeletePayload, SetPayload};
use collection::operations::point_ops::{PointInsertOperations};
use segment::types::PointIdType;
use storage::content_manager::toc::TableOfContent;

#[derive(Deserialize, Serialize, JsonSchema)]
pub struct UpdateParam {
    pub wait: Option<bool>,
}

// Deprecated
#[post("/collections/{name}")]
pub async fn update_points(
    toc: web::Data<Arc<TableOfContent>>,
    path: web::Path<String>,
    operation: web::Json<CollectionUpdateOperations>,
    params: Query<UpdateParam>,
) -> impl Responder {
    let collection_name = path.into_inner();
    let operation = operation.into_inner();
    let wait = params.wait.unwrap_or(false);
    let timing = Instant::now();

    let response = do_update_points(&toc.into_inner(), &collection_name, operation, wait).await;
    process_response(response, timing)
}

#[post("/collections/{name}/points")]
pub async fn upsert_points(
    toc: web::Data<Arc<TableOfContent>>,
    path: web::Path<String>,
    operation: web::Json<PointInsertOperations>,
    params: Query<UpdateParam>,
) -> impl Responder {
    let collection_name = path.into_inner();
    let operation = operation.into_inner();
    let wait = params.wait.unwrap_or(false);
    let timing = Instant::now();

    let response = do_upsert_points(&toc.into_inner(), &collection_name, operation, wait).await;
    process_response(response, timing)
}

#[post("/collections/{name}/points/delete")]
pub async fn delete_points(
    toc: web::Data<Arc<TableOfContent>>,
    path: web::Path<String>,
    operation: web::Json<Vec<PointIdType>>,
    params: Query<UpdateParam>,
) -> impl Responder {
    let collection_name = path.into_inner();
    let operation = operation.into_inner();
    let wait = params.wait.unwrap_or(false);
    let timing = Instant::now();

    let response = do_delete_points(&toc.into_inner(), &collection_name, operation, wait).await;
    process_response(response, timing)
}

#[post("/collections/{name}/points/payload")]
pub async fn set_payload(
    toc: web::Data<Arc<TableOfContent>>,
    path: web::Path<String>,
    operation: web::Json<SetPayload>,
    params: Query<UpdateParam>,
) -> impl Responder {
    let collection_name = path.into_inner();
    let operation = operation.into_inner();
    let wait = params.wait.unwrap_or(false);
    let timing = Instant::now();

    let response = do_set_payload(&toc.into_inner(), &collection_name, operation, wait).await;
    process_response(response, timing)
}

#[post("/collections/{name}/points/payload/delete")]
pub async fn delete_payload(
    toc: web::Data<Arc<TableOfContent>>,
    path: web::Path<String>,
    operation: web::Json<DeletePayload>,
    params: Query<UpdateParam>,
) -> impl Responder {
    let collection_name = path.into_inner();
    let operation = operation.into_inner();
    let wait = params.wait.unwrap_or(false);
    let timing = Instant::now();

    let response = do_delete_payload(&toc.into_inner(), &collection_name, operation, wait).await;
    process_response(response, timing)
}

#[post("/collections/{name}/points/payload/clear")]
pub async fn clear_payload(
    toc: web::Data<Arc<TableOfContent>>,
    path: web::Path<String>,
    operation: web::Json<Vec<PointIdType>>,
    params: Query<UpdateParam>,
) -> impl Responder {
    let collection_name = path.into_inner();
    let operation = operation.into_inner();
    let wait = params.wait.unwrap_or(false);
    let timing = Instant::now();

    let response = do_delete_points(&toc.into_inner(), &collection_name, operation, wait).await;
    process_response(response, timing)
}
