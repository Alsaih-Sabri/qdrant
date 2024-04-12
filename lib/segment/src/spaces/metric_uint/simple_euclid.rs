use common::types::ScoreType;

use crate::data_types::vectors::{
    DenseVector, FromVectorElementSlice, TypedDenseVector, VectorElementTypeByte,
};
use crate::spaces::metric::Metric;
use crate::spaces::simple::EuclidMetric;
use crate::types::Distance;

impl Metric<VectorElementTypeByte> for EuclidMetric {
    fn distance() -> Distance {
        Distance::Euclid
    }

    fn similarity(v1: &[VectorElementTypeByte], v2: &[VectorElementTypeByte]) -> ScoreType {
        euclid_similarity_bytes(v1, v2)
    }

    fn preprocess(vector: DenseVector) -> TypedDenseVector<VectorElementTypeByte> {
        Vec::from_vector_element_slice(&vector)
    }
}

fn euclid_similarity_bytes(
    v1: &[VectorElementTypeByte],
    v2: &[VectorElementTypeByte],
) -> ScoreType {
    -v1.iter()
        .zip(v2)
        .map(|(a, b)| {
            let diff = *a as i32 - *b as i32;
            diff * diff
        })
        .sum::<i32>() as ScoreType
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_types::vectors::IntoDenseVector;

    #[test]
    fn test_conversion_to_bytes() {
        let dense_vector = [-10.0, 1.0, 2.0, 3.0, 255., 300.].into_dense_vector();
        let typed_dense_vector: TypedDenseVector<VectorElementTypeByte> =
            EuclidMetric::preprocess(dense_vector);
        let expected: TypedDenseVector<VectorElementTypeByte> = vec![0, 1, 2, 3, 255, 255];
        assert_eq!(typed_dense_vector, expected);
    }
}
