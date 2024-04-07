#[cfg(not(target_os = "windows"))]
mod prof;

use avx::avx_cosine::avx_cosine_similarity_bytes;
use avx::avx_dot::avx_dot_similarity_bytes;
use avx::avx_euclid::avx_euclid_similarity_bytes;
use avx::avx_manhattan::avx_manhattan_similarity_bytes;
use criterion::{criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use segment::data_types::vectors::VectorElementTypeByte;
use segment::spaces::metric::Metric;
use segment::spaces::simple::{CosineMetric, DotProductMetric, EuclidMetric, ManhattanMetric};
use sse::sse_cosine::sse_cosine_similarity_bytes;
use sse::sse_dot::sse_dot_similarity_bytes;
use sse::sse_euclid::sse_euclid_similarity_bytes;
use sse::sse_manhattan::sse_manhattan_similarity_bytes;

const DIM: usize = 1024;
const COUNT: usize = 100_000;

fn byte_metrics_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("byte-metrics-bench-group");

    let mut rng = StdRng::seed_from_u64(42);

    let random_vectors_1: Vec<Vec<u8>> = (0..COUNT)
        .map(|_| (0..DIM).map(|_| rng.gen_range(0..255)).collect())
        .collect();
    let random_vectors_2: Vec<Vec<u8>> = (0..COUNT)
        .map(|_| (0..DIM).map(|_| rng.gen_range(0..255)).collect())
        .collect();

    group.bench_function("byte-dot-no-simd", |b| {
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) % COUNT;
            <DotProductMetric as Metric<VectorElementTypeByte>>::similarity(
                &random_vectors_1[i],
                &random_vectors_2[i],
            )
        });
    });

    group.bench_function("byte-dot-avx", |b| {
        let mut i = 0;
        b.iter(|| unsafe {
            i = (i + 1) % COUNT;
            avx_dot_similarity_bytes(&random_vectors_1[i], &random_vectors_2[i])
        });
    });

    group.bench_function("byte-dot-sse", |b| {
        let mut i = 0;
        b.iter(|| unsafe {
            i = (i + 1) % COUNT;
            sse_dot_similarity_bytes(&random_vectors_1[i], &random_vectors_2[i])
        });
    });

    group.bench_function("byte-cosine-no-simd", |b| {
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) % COUNT;
            <CosineMetric as Metric<VectorElementTypeByte>>::similarity(
                &random_vectors_1[i],
                &random_vectors_2[i],
            )
        });
    });

    group.bench_function("byte-cosine-avx", |b| {
        let mut i = 0;
        b.iter(|| unsafe {
            i = (i + 1) % COUNT;
            avx_cosine_similarity_bytes(&random_vectors_1[i], &random_vectors_2[i])
        });
    });

    group.bench_function("byte-cosine-sse", |b| {
        let mut i = 0;
        b.iter(|| unsafe {
            i = (i + 1) % COUNT;
            sse_cosine_similarity_bytes(&random_vectors_1[i], &random_vectors_2[i])
        });
    });

    group.bench_function("byte-euclid-no-simd", |b| {
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) % COUNT;
            <EuclidMetric as Metric<VectorElementTypeByte>>::similarity(
                &random_vectors_1[i],
                &random_vectors_2[i],
            )
        });
    });

    group.bench_function("byte-euclid-avx", |b| {
        let mut i = 0;
        b.iter(|| unsafe {
            i = (i + 1) % COUNT;
            avx_euclid_similarity_bytes(&random_vectors_1[i], &random_vectors_2[i])
        });
    });

    group.bench_function("byte-euclid-sse", |b| {
        let mut i = 0;
        b.iter(|| unsafe {
            i = (i + 1) % COUNT;
            sse_euclid_similarity_bytes(&random_vectors_1[i], &random_vectors_2[i])
        });
    });

    group.bench_function("byte-manhattan-no-simd", |b| {
        let mut i = 0;
        b.iter(|| {
            i = (i + 1) % COUNT;
            <ManhattanMetric as Metric<VectorElementTypeByte>>::similarity(
                &random_vectors_1[i],
                &random_vectors_2[i],
            )
        });
    });

    group.bench_function("byte-manhattan-avx", |b| {
        let mut i = 0;
        b.iter(|| unsafe {
            i = (i + 1) % COUNT;
            avx_manhattan_similarity_bytes(&random_vectors_1[i], &random_vectors_2[i])
        });
    });

    group.bench_function("byte-manhattan-sse", |b| {
        let mut i = 0;
        b.iter(|| unsafe {
            i = (i + 1) % COUNT;
            sse_manhattan_similarity_bytes(&random_vectors_1[i], &random_vectors_2[i])
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = byte_metrics_bench
}

criterion_main!(benches);
