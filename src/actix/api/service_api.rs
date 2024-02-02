use std::any::Any;
use std::sync::Arc;

use actix_web::http::header::ContentType;
use actix_web::http::StatusCode;
use actix_web::rt::time::Instant;
use actix_web::web::Query;
use actix_web::{get, post, web, HttpResponse, Responder};
use actix_web_validator::Json;
use schemars::JsonSchema;
use segment::common::anonymize::Anonymize;
use serde::{Deserialize, Serialize};
use storage::content_manager::toc::TableOfContent;
use tokio::sync::Mutex;

use crate::actix::helpers::process_response;
use crate::common::health;
use crate::common::helpers::LocksOption;
use crate::common::metrics::MetricsData;
use crate::common::stacktrace::get_stack_trace;
use crate::common::telemetry::{TelemetryCollector, TelemetryData};
use crate::common::telemetry_ops::collections_telemetry::CollectionTelemetryEnum;

#[derive(Deserialize, Serialize, JsonSchema)]
pub struct TelemetryParam {
    pub anonymize: Option<bool>,
    pub details_level: Option<usize>,
}

#[get("/telemetry")]
async fn telemetry(
    telemetry_collector: web::Data<Mutex<TelemetryCollector>>,
    params: Query<TelemetryParam>,
) -> impl Responder {
    let timing = Instant::now();
    let anonymize = params.anonymize.unwrap_or(false);
    let details_level = params.details_level.unwrap_or(0);
    let telemetry_collector = telemetry_collector.lock().await;
    let telemetry_data = telemetry_collector.prepare_data(details_level).await;
    let telemetry_data = if anonymize {
        telemetry_data.anonymize()
    } else {
        telemetry_data
    };
    process_response(Ok(telemetry_data), timing)
}

#[derive(Deserialize, Serialize, JsonSchema)]
pub struct MetricsParam {
    pub anonymize: Option<bool>,
    pub collection_names: Option<Vec<String>>,
}

#[get("/metrics")]
async fn metrics(
    telemetry_collector: web::Data<Mutex<TelemetryCollector>>,
    params: actix_web_lab::extract::Query<MetricsParam>,
) -> impl Responder {
    let timing = Instant::now();
    let anonymize = params.anonymize.unwrap_or(false);
    let collection_names = params.collection_names.clone().unwrap_or_default();
    println!("names: {collection_names:?}");
    let telemetry_collector = telemetry_collector.lock().await;

    let resp = telemetry_collector
        .prepare_data_for(2, &collection_names)
        .await;
    if let Err(e) = resp {
        return process_response::<TelemetryData>(Err(e), timing);
    }
    // safe to unwrap here
    let telemetry_data = resp.unwrap();
    println!("{:#?}", telemetry_data.requests);
    // for v in telemetry_data.collections.collections.clone().unwrap() {
    //     match v {
    //         CollectionTelemetryEnum::Full(v) => println!("full {:#?}", v.id),
    //         CollectionTelemetryEnum::Aggregated(v) => println!("agg {:#?}", v.type_id()),
    //     };
    // }

    let telemetry_data = if anonymize {
        telemetry_data.anonymize()
    } else {
        telemetry_data
    };

    HttpResponse::Ok()
        .content_type(ContentType::plaintext())
        .body(MetricsData::from(telemetry_data).format_metrics())
}

#[post("/locks")]
async fn put_locks(
    toc: web::Data<TableOfContent>,
    locks_option: Json<LocksOption>,
) -> impl Responder {
    let timing = Instant::now();
    let result = LocksOption {
        write: toc.get_ref().is_write_locked(),
        error_message: toc.get_ref().get_lock_error_message(),
    };
    toc.get_ref()
        .set_locks(locks_option.write, locks_option.error_message.clone());
    process_response(Ok(result), timing)
}

#[get("/locks")]
async fn get_locks(toc: web::Data<TableOfContent>) -> impl Responder {
    let timing = Instant::now();
    let result = LocksOption {
        write: toc.get_ref().is_write_locked(),
        error_message: toc.get_ref().get_lock_error_message(),
    };
    process_response(Ok(result), timing)
}

#[get("/stacktrace")]
async fn get_stacktrace() -> impl Responder {
    let timing = Instant::now();
    let result = get_stack_trace();
    process_response(Ok(result), timing)
}

#[get("/healthz")]
async fn healthz() -> impl Responder {
    kubernetes_healthz().await
}

#[get("/livez")]
async fn livez() -> impl Responder {
    kubernetes_healthz().await
}

#[get("/readyz")]
async fn readyz(health_checker: web::Data<Option<Arc<health::HealthChecker>>>) -> impl Responder {
    let is_ready = match health_checker.as_ref() {
        Some(health_checker) => health_checker.check_ready().await,
        None => true,
    };

    let (status, body) = if is_ready {
        (StatusCode::OK, "all shards are ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "some shards are not ready")
    };

    HttpResponse::build(status)
        .content_type(ContentType::plaintext())
        .body(body)
}

/// Basic Kubernetes healthz endpoint
async fn kubernetes_healthz() -> impl Responder {
    HttpResponse::Ok()
        .content_type(ContentType::plaintext())
        .body("healthz check passed")
}

// Configure services
pub fn config_service_api(cfg: &mut web::ServiceConfig) {
    cfg.service(telemetry)
        .service(metrics)
        .service(put_locks)
        .service(get_locks)
        .service(get_stacktrace)
        .service(healthz)
        .service(livez)
        .service(readyz);
}
