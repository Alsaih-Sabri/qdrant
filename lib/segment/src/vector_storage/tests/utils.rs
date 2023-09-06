use std::{error, result};

use rand::seq::IteratorRandom;

use crate::id_tracker::IdTracker;
use crate::types::PointOffsetType;
use crate::vector_storage::{RawScorer, ScoredPointOffset, VectorStorage};

pub type Result<T, E = Error> = result::Result<T, E>;
pub type Error = Box<dyn error::Error>;

pub fn sampler(rng: impl rand::Rng) -> impl Iterator<Item = f32> {
    rng.sample_iter(rand::distributions::Standard)
}

pub fn insert_random_vectors(
    rng: &mut impl rand::Rng,
    storage: &mut impl VectorStorage,
    vectors: usize,
) -> Result<()> {
    let start = storage.total_vector_count() as u32;
    let end = start + vectors as u32;

    let mut vector = vec![0.; storage.vector_dim()];
    let mut sampler = sampler(rng);

    for offset in start..end {
        for (item, value) in vector.iter_mut().zip(&mut sampler) {
            *item = value;
        }

        storage.insert_vector(offset, &vector)?;
    }

    Ok(())
}

pub fn delete_random_vectors(
    rng: &mut impl rand::Rng,
    storage: &mut impl VectorStorage,
    id_tracker: &mut impl IdTracker,
    vectors: usize,
) -> Result<()> {
    let offsets = (0..storage.total_vector_count() as _).choose_multiple(rng, vectors);

    for offset in offsets {
        storage.delete_vector(offset)?;
        id_tracker.drop(crate::types::ExtendedPointId::NumId(offset.into()))?;
    }

    Ok(())
}

pub fn score(scorer: &dyn RawScorer, points: &[PointOffsetType]) -> Vec<ScoredPointOffset> {
    let mut scores = vec![Default::default(); points.len()];
    let scored = scorer.score_points(points, &mut scores);
    scores.resize_with(scored, Default::default);
    scores
}
