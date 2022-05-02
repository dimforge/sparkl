use super::CudaParticleModel;
use crate::core::dynamics::ParticleData;
use crate::cuda::{CudaConstitutiveModel, CudaFailureModel, CudaPlasticModel, SharedCudaVec};
use crate::dynamics::ParticleModel;
use crate::kernels::GpuParticleModel;
use cust::{error::CudaResult, memory::DeviceBuffer};

pub struct CudaParticleModelSet {
    pub models: Vec<CudaParticleModel>,
    pub gpu_models: Vec<GpuParticleModel>,
    pub buffer: DeviceBuffer<GpuParticleModel>,
}

impl Default for CudaParticleModelSet {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl CudaParticleModelSet {
    pub fn new() -> CudaResult<Self> {
        Self::from_models(vec![])
    }

    // TODO: make this async?
    pub fn from_models(models: Vec<CudaParticleModel>) -> CudaResult<Self> {
        let gpu_models: Vec<_> = models.iter().map(|model| model.to_gpu_model()).collect();
        let buffer = DeviceBuffer::from_slice(&gpu_models)?;

        Ok(Self {
            models,
            gpu_models,
            buffer,
        })
    }

    pub fn from_model_set<'a>(
        models: impl Iterator<Item = &'a ParticleModel>,
        attribute: SharedCudaVec<ParticleData>,
    ) -> CudaResult<Self> {
        let cuda_models: Vec<_> = models
            .map(|model| CudaParticleModel {
                constitutive_model: CudaConstitutiveModel::from_core_model(
                    model.constitutive_model.to_core_model().unwrap(),
                    attribute.clone(),
                ),
                plastic_model: model.plastic_model.as_ref().map(|m| {
                    CudaPlasticModel::from_core_model(m.to_core_model().unwrap(), attribute.clone())
                }),
                failure_model: model
                    .failure_model
                    .as_ref()
                    .map(|m| CudaFailureModel::from_core_model(m.to_core_model().unwrap())),
            })
            .collect();

        Self::from_models(cuda_models)
    }

    // TODO: make this async?
    pub fn ensure_models_are_uploaded(&mut self) -> CudaResult<()> {
        // NOTE: the models may have been modified by the user.
        //       Make sure we re-upload them if needed.
        let mut outdated = false;
        for (model, uploaded) in self.models.iter().zip(self.gpu_models.iter()) {
            if model.to_gpu_model() != *uploaded {
                outdated = true;
                break;
            }
        }

        if outdated {
            let gpu_models: Vec<_> = self
                .models
                .iter()
                .map(|model| model.to_gpu_model())
                .collect();
            let buffer = DeviceBuffer::from_slice(&gpu_models)?;
            self.buffer = buffer;
            self.gpu_models = gpu_models;
        }

        Ok(())
    }
}
