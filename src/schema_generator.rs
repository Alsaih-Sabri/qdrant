use api::grpc::models::CollectionsResponse;
use collection::operations::cluster_ops::ClusterOperations;
use collection::operations::payload_ops::{DeletePayload, SetPayload};
use collection::operations::point_ops::{PointInsertOperations, PointsSelector};
use collection::operations::snapshot_ops::SnapshotDescription;
use collection::operations::types::{
    CollectionClusterInfo, CollectionInfo, CountRequest, CountResult, PointRequest,
    RecommendRequest, Record, ScrollRequest, ScrollResult, SearchRequest, SearchRequestBatch,
    UpdateResult,
};
use schemars::{schema_for, JsonSchema};
use segment::types::ScoredPoint;
use serde::{Deserialize, Serialize};
use storage::content_manager::collection_meta_ops::{
    ChangeAliasesOperation, CreateCollection, UpdateCollection,
};
use storage::types::ClusterStatus;

use crate::common::points::CreateFieldIndex;
use crate::common::telemetry::TelemetryData;

mod actix;
mod common;
mod settings;

#[derive(Deserialize, Serialize, JsonSchema)]
struct AllDefinitions {
    a1: CollectionsResponse,
    a2: CollectionInfo,
    // a3: CollectionMetaOperations,
    a4: PointRequest,
    a5: Record,
    a6: SearchRequest,
    a7: ScoredPoint,
    a8: UpdateResult,
    // a9: CollectionUpdateOperations,
    aa: RecommendRequest,
    ab: ScrollRequest,
    ac: ScrollResult,
    ad: CreateCollection,
    ae: UpdateCollection,
    af: ChangeAliasesOperation,
    ag: CreateFieldIndex,
    ah: PointsSelector,
    ai: PointInsertOperations,
    aj: SetPayload,
    ak: DeletePayload,
    al: ClusterStatus,
    am: SnapshotDescription,
    an: CountRequest,
    ao: CountResult,
    ap: CollectionClusterInfo,
    aq: TelemetryData,
    ar: ClusterOperations,
    at: SearchRequestBatch,
}

fn save_schema<T: JsonSchema>() {
    let schema = schema_for!(T);
    let schema_str = serde_json::to_string_pretty(&schema).unwrap();
    println!("{}", schema_str)
}

fn main() {
    save_schema::<AllDefinitions>();
}
