use std::collections::HashMap;
use std::ops::Deref;

use collection::operations::point_ops::VectorPersisted;
use storage::content_manager::errors::StorageError;

use super::batch_processing::BatchAccum;
use super::service::{InferenceData, InferenceInput, InferenceService, InferenceType};

pub struct BatchAccumInferred {
    objects: HashMap<InferenceData, VectorPersisted>,
}

impl BatchAccumInferred {
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
        }
    }

    pub async fn from_batch_accum(
        batch: BatchAccum,
        inference_type: InferenceType,
    ) -> Result<Self, StorageError> {
        let BatchAccum { objects } = batch;

        if objects.is_empty() {
            return Ok(Self::new());
        }

        let guard = InferenceService::get_global().await;

        let service = match guard.deref() {
            None => return Err(StorageError::service_error(
                "InferenceService is not initialized. Please check if it was properly configured and initialized during startup."
            )),
            Some(service) => service
        };

        service.validate()?;

        let objects_serialized: Vec<_> = objects.into_iter().collect();
        let inference_inputs: Vec<_> = objects_serialized
            .iter()
            .cloned()
            .map(InferenceInput::from)
            .collect();

        let vectors = service
            .infer(inference_inputs, inference_type)
            .await
            .map_err(|e| StorageError::service_error(
                format!("Inference request failed. Check if inference service is running and properly configured: {e}")
            ))?;

        if vectors.is_empty() {
            return Err(StorageError::service_error(
                "Inference service returned no vectors. Check if models are properly loaded.",
            ));
        }

        let objects = objects_serialized.into_iter().zip(vectors).collect();

        Ok(Self { objects })
    }

    pub fn get_vector(&self, data: &InferenceData) -> Option<&VectorPersisted> {
        self.objects.get(data)
    }
}
