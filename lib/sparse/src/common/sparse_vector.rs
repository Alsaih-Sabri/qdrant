use std::borrow::Cow;

use itertools::Itertools;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError, ValidationErrors};

use crate::common::types::{DimId, DimWeight};

/// Sparse vector structure
#[derive(Debug, PartialEq, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct SparseVector {
    /// indices must be unique
    pub indices: Vec<DimId>,
    /// values and indices must be the same length
    pub values: Vec<DimWeight>,
}

impl SparseVector {
    pub fn new(indices: Vec<DimId>, values: Vec<DimWeight>) -> Result<Self, ValidationErrors> {
        let vector = SparseVector { indices, values };
        vector.validate()?;
        Ok(vector)
    }

    /// Sort this vector by indices.
    ///
    /// Sorting is required for scoring and overlap checks.
    pub fn sort_by_indices(&mut self) {
        // do not sort if already sorted
        if self.is_sorted() {
            return;
        }

        let mut indexed_values: Vec<(u32, f32)> = self
            .indices
            .iter()
            .zip(self.values.iter())
            .map(|(&i, &v)| (i, v))
            .collect();

        // Sort the vector of tuples by indices
        indexed_values.sort_by_key(|&(i, _)| i);

        self.indices = indexed_values.iter().map(|&(i, _)| i).collect();
        self.values = indexed_values.iter().map(|&(_, v)| v).collect();
    }

    /// Check if this vector is sorted by indices.
    pub fn is_sorted(&self) -> bool {
        self.indices.windows(2).all(|w| w[0] < w[1])
    }

    /// Check if this vector is empty.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() && self.values.is_empty()
    }

    /// Score this vector against another vector using dot product.
    /// Warning: Expects both vectors to be sorted by indices.
    ///
    /// Return None if the vectors do not overlap.
    pub fn score(&self, other: &SparseVector) -> Option<f32> {
        debug_assert!(self.is_sorted());
        debug_assert!(other.is_sorted());
        let mut score = 0.0;
        // track whether there is any overlap
        let mut overlap = false;
        let mut i = 0;
        let mut j = 0;
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    overlap = true;
                    score += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
            }
        }
        if overlap {
            Some(score)
        } else {
            None
        }
    }

    /// Update sparse vector with the result of performing all indices-wise operations with `other`
    pub fn combine_aggregate(
        &mut self,
        other: &SparseVector,
        op: impl Fn(DimWeight, DimWeight) -> DimWeight,
    ) {
        // Sort vectors if not already sorted
        if !self.is_sorted() {
            self.sort_by_indices();
        }

        // Copy and sort `other` vector if not already sorted
        let cow_other: Cow<SparseVector> = if !other.is_sorted() {
            let mut other = other.clone();
            other.sort_by_indices();
            Cow::Owned(other)
        } else {
            Cow::Borrowed(other)
        };
        let other = &cow_other;
        assert!(other.is_sorted());

        // record initial length of `self` vector
        let initial_self_len = self.indices.len();

        // update overlapping indices
        let mut i = 0;
        let mut j = 0;
        // stop when either vector is exhausted
        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Less => {
                    // update existing value with missing one in `other` - combine with 0.0
                    self.values[i] = op(self.values[i], 0.0);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    // value missing in `self` - add index and combine with 0.0
                    self.indices.push(other.indices[j]);
                    self.values.push(op(0.0, other.values[j]));
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    // combine values
                    self.values[i] = op(self.values[i], other.values[j]);
                    i += 1;
                    j += 1;
                }
            }
        }

        // add remaining values from `other` if any
        while i < self.indices.len() {
            self.values[i] = op(self.values[i], 0.0);
            i += 1;
        }

        // add remaining values from `self` if any
        while j < other.indices.len() {
            self.indices.push(other.indices[j]);
            self.values.push(op(0.0, other.values[j]));
            j += 1;
        }

        // if new indices were added, sort the vector
        if initial_self_len != self.indices.len() {
            self.sort_by_indices();
            assert!(self.validate().is_ok());
        }
    }
}

impl TryFrom<Vec<(u32, f32)>> for SparseVector {
    type Error = ValidationErrors;

    fn try_from(tuples: Vec<(u32, f32)>) -> Result<Self, Self::Error> {
        let mut indices = Vec::with_capacity(tuples.len());
        let mut values = Vec::with_capacity(tuples.len());
        for (i, w) in tuples {
            indices.push(i);
            values.push(w);
        }
        SparseVector::new(indices, values)
    }
}

#[cfg(test)]
impl<const N: usize> From<[(u32, f32); N]> for SparseVector {
    fn from(value: [(u32, f32); N]) -> Self {
        value.to_vec().try_into().unwrap()
    }
}

impl Validate for SparseVector {
    fn validate(&self) -> Result<(), ValidationErrors> {
        validate_sparse_vector_impl(&self.indices, &self.values)
    }
}

pub fn validate_sparse_vector_impl(
    indices: &[DimId],
    values: &[DimWeight],
) -> Result<(), ValidationErrors> {
    let mut errors = ValidationErrors::default();

    if indices.len() != values.len() {
        errors.add(
            "values",
            ValidationError::new("must be the same length as indices"),
        );
    }
    if indices.iter().unique().count() != indices.len() {
        errors.add("indices", ValidationError::new("must be unique"));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_aligned_same_size() {
        let v1 = SparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
        let v2 = SparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(v1.score(&v2), Some(14.0));
    }

    #[test]
    fn test_score_not_aligned_same_size() {
        let v1 = SparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
        let v2 = SparseVector::new(vec![2, 3, 4], vec![2.0, 3.0, 4.0]).unwrap();
        assert_eq!(v1.score(&v2), Some(13.0));
    }

    #[test]
    fn test_score_aligned_different_size() {
        let v1 = SparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
        let v2 = SparseVector::new(vec![1, 2, 3, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(v1.score(&v2), Some(14.0));
    }

    #[test]
    fn test_score_not_aligned_different_size() {
        let v1 = SparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
        let v2 = SparseVector::new(vec![2, 3, 4, 5], vec![2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(v1.score(&v2), Some(13.0));
    }

    #[test]
    fn test_score_no_overlap() {
        let v1 = SparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
        let v2 = SparseVector::new(vec![4, 5, 6], vec![2.0, 3.0, 4.0]).unwrap();
        assert!(v1.score(&v2).is_none());
    }

    #[test]
    fn validation_test() {
        let fully_empty = SparseVector::new(vec![], vec![]);
        assert!(fully_empty.is_ok());
        assert!(fully_empty.unwrap().is_empty());

        let different_length = SparseVector::new(vec![1, 2, 3], vec![1.0, 2.0]);
        assert!(different_length.is_err());

        let not_sorted = SparseVector::new(vec![1, 3, 2], vec![1.0, 2.0, 3.0]);
        assert!(not_sorted.is_ok());

        let not_unique = SparseVector::new(vec![1, 2, 3, 2], vec![1.0, 2.0, 3.0, 4.0]);
        assert!(not_unique.is_err());
    }

    #[test]
    fn sorting_test() {
        let mut not_sorted = SparseVector::new(vec![1, 3, 2], vec![1.0, 2.0, 3.0]).unwrap();
        assert!(!not_sorted.is_sorted());
        not_sorted.sort_by_indices();
        assert!(not_sorted.is_sorted());
    }

    #[test]
    fn combine_aggregate_test() {
        let mut a = SparseVector::new(vec![1, 2, 3], vec![0.1, 0.2, 0.3]).unwrap();
        let b = SparseVector::new(vec![2, 3, 4], vec![2.0, 3.0, 4.0]).unwrap();
        a.combine_aggregate(&b, |x, y| x + 2.0 * y);
        assert_eq!(a.indices, vec![1, 2, 3, 4]);
        assert_eq!(a.values, vec![0.1, 4.2, 6.3, 8.0]);

        let a = SparseVector::new(vec![1, 2, 3], vec![0.1, 0.2, 0.3]).unwrap();
        let mut b = SparseVector::new(vec![2, 3, 4], vec![2.0, 3.0, 4.0]).unwrap();
        b.combine_aggregate(&a, |x, y| x + 2.0 * y);
        assert_eq!(b.indices, vec![1, 2, 3, 4]);
        assert_eq!(b.values, vec![0.2, 2.4, 3.6, 4.0]);

        let mut a = SparseVector::new(vec![1, 2, 3], vec![0.1, 0.2, 0.3]).unwrap();
        let b = SparseVector::new(vec![4, 2, 3], vec![4.0, 2.0, 3.0]).unwrap();
        a.combine_aggregate(&b, |x, y| x + 2.0 * y);
        assert_eq!(a.indices, vec![1, 2, 3, 4]);
        assert_eq!(a.values, vec![0.1, 4.2, 6.3, 8.0]);

        let a = SparseVector::new(vec![1, 3, 2], vec![0.1, 0.3, 0.2]).unwrap();
        let mut b = SparseVector::new(vec![2, 3, 4], vec![2.0, 3.0, 4.0]).unwrap();
        b.combine_aggregate(&a, |x, y| x + 2.0 * y);
        assert_eq!(b.indices, vec![1, 2, 3, 4]);
        assert_eq!(b.values, vec![0.2, 2.4, 3.6, 4.0]);
    }
}
