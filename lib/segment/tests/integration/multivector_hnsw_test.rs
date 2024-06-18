use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use atomic_refcell::AtomicRefCell;
use common::cpu::CpuPermit;
use rand::prelude::StdRng;
use rand::SeedableRng;
use segment::common::rocksdb_wrapper::{open_db, DB_VECTOR_CF};
use segment::data_types::vectors::{
    only_default_vector, MultiDenseVector, QueryVector, TypedMultiDenseVectorRef,
    VectorElementType, VectorRef, DEFAULT_VECTOR_NAME,
};
use segment::entry::entry_point::SegmentEntry;
use segment::fixtures::index_fixtures::random_vector;
use segment::fixtures::payload_fixtures::random_int_payload;
use segment::index::hnsw_index::graph_links::GraphLinksRam;
use segment::index::hnsw_index::hnsw::{HNSWIndex, HnswIndexOpenArgs};
use segment::index::VectorIndex;
use segment::json_path::path;
use segment::segment_constructor::build_segment;
use segment::spaces::metric::Metric;
use segment::spaces::simple::{CosineMetric, DotProductMetric, EuclidMetric, ManhattanMetric};
use segment::types::{
    Condition, Distance, FieldCondition, Filter, HnswConfig, Indexes, MultiVectorConfig, Payload,
    PayloadSchemaType, SegmentConfig, SeqNumberType, VectorDataConfig, VectorStorageType,
};
use segment::vector_storage::multi_dense::simple_multi_dense_vector_storage::open_simple_multi_dense_vector_storage;
use segment::vector_storage::VectorStorage;
use serde_json::json;
use tempfile::Builder;

#[test]
fn test_single_multi_and_dense_hnsw_equivalency() {
    let num_vectors: u64 = 1_000;
    let distance = Distance::Cosine;
    let num_payload_values = 2;
    let dim = 8;

    let mut rnd = StdRng::seed_from_u64(42);

    let dir = Builder::new().prefix("segment_dir").tempdir().unwrap();

    let config = SegmentConfig {
        vector_data: HashMap::from([(
            DEFAULT_VECTOR_NAME.to_owned(),
            VectorDataConfig {
                size: dim,
                distance,
                storage_type: VectorStorageType::Memory,
                index: Indexes::Plain {},
                quantization_config: None,
                multivec_config: None,
                datatype: None,
            },
        )]),
        sparse_vector_data: Default::default(),
        payload_storage_type: Default::default(),
    };

    let int_key = "int";

    let mut segment = build_segment(dir.path(), &config, true).unwrap();

    segment
        .create_field_index(0, &path(int_key), Some(&PayloadSchemaType::Integer.into()))
        .unwrap();

    let dir = Builder::new().prefix("storage_dir").tempdir().unwrap();
    let db = open_db(dir.path(), &[DB_VECTOR_CF]).unwrap();
    let mut multi_storage = open_simple_multi_dense_vector_storage(
        db,
        DB_VECTOR_CF,
        dim,
        distance,
        MultiVectorConfig::default(),
        &AtomicBool::new(false),
    )
    .unwrap();

    for n in 0..num_vectors {
        let idx = n.into();
        let vector = random_vector(&mut rnd, dim);
        let preprocessed_vector = match distance {
            Distance::Cosine => {
                <CosineMetric as Metric<VectorElementType>>::preprocess(vector.clone())
            }
            Distance::Euclid => {
                <EuclidMetric as Metric<VectorElementType>>::preprocess(vector.clone())
            }
            Distance::Dot => {
                <DotProductMetric as Metric<VectorElementType>>::preprocess(vector.clone())
            }
            Distance::Manhattan => {
                <ManhattanMetric as Metric<VectorElementType>>::preprocess(vector.clone())
            }
        };
        let vector_multi = MultiDenseVector::new(preprocessed_vector, vector.len());

        let int_payload = random_int_payload(&mut rnd, num_payload_values..=num_payload_values);
        let payload: Payload = json!({int_key:int_payload,}).into();

        segment
            .upsert_point(n as SeqNumberType, idx, only_default_vector(&vector))
            .unwrap();
        segment
            .set_full_payload(n as SeqNumberType, idx, &payload)
            .unwrap();

        let internal_id = segment.id_tracker.borrow().internal_id(idx).unwrap();
        multi_storage
            .insert_vector(
                internal_id,
                VectorRef::MultiDense(TypedMultiDenseVectorRef::from(&vector_multi)),
            )
            .unwrap();
    }

    let hnsw_dir = Builder::new().prefix("hnsw_dir").tempdir().unwrap();

    let stopped = AtomicBool::new(false);

    let m = 8;
    let ef_construct = 100;
    let full_scan_threshold = 10000;

    let hnsw_config = HnswConfig {
        m,
        ef_construct,
        full_scan_threshold,
        max_indexing_threads: 2,
        on_disk: Some(false),
        payload_m: None,
    };

    // single threaded mode to guarantee equivalency between single and multi hnsw
    let permit = Arc::new(CpuPermit::dummy(1));

    let vector_storage = &segment.vector_data[DEFAULT_VECTOR_NAME].vector_storage;
    let quantized_vectors = &segment.vector_data[DEFAULT_VECTOR_NAME].quantized_vectors;
    let hnsw_index_dense = HNSWIndex::<GraphLinksRam>::open(HnswIndexOpenArgs {
        path: hnsw_dir.path(),
        id_tracker: segment.id_tracker.clone(),
        vector_storage: vector_storage.clone(),
        quantized_vectors: quantized_vectors.clone(),
        payload_index: segment.payload_index.clone(),
        hnsw_config: hnsw_config.clone(),
        permit: Some(permit.clone()),
        stopped: &stopped,
    })
    .unwrap();

    let multi_storage = Arc::new(AtomicRefCell::new(multi_storage));

    let hnsw_index_multi = HNSWIndex::<GraphLinksRam>::open(HnswIndexOpenArgs {
        path: hnsw_dir.path(),
        id_tracker: segment.id_tracker.clone(),
        vector_storage: multi_storage,
        quantized_vectors: quantized_vectors.clone(),
        payload_index: segment.payload_index.clone(),
        hnsw_config,
        permit: Some(permit),
        stopped: &stopped,
    })
    .unwrap();

    for _ in 0..10 {
        let random_vector = random_vector(&mut rnd, dim);
        let query_vector = random_vector.clone().into();
        let query_vector_multi = QueryVector::Nearest(vec![random_vector].try_into().unwrap());

        let payload_value = random_int_payload(&mut rnd, 1..=1).pop().unwrap();

        let filter = Filter::new_must(Condition::Field(FieldCondition::new_match(
            path(int_key),
            payload_value.into(),
        )));

        let search_res_dense = hnsw_index_dense
            .search(
                &[&query_vector],
                Some(&filter),
                10,
                None,
                &Default::default(),
            )
            .unwrap();

        let search_res_multi = hnsw_index_multi
            .search(
                &[&query_vector_multi],
                Some(&filter),
                10,
                None,
                &Default::default(),
            )
            .unwrap();

        assert_eq!(search_res_dense, search_res_multi);
    }
}
