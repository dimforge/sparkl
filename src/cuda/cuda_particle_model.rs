use cust::error::CudaResult;
use cust::memory::DeviceCopy;
use kernels::{GpuConstitutiveModel, GpuFailureModel, GpuParticleModel, GpuPlasticModel};

use crate::core::dynamics::models::{
    CoreConstitutiveModel, CoreFailureModel, CorePlasticModel, CorotatedLinearElasticity,
    DruckerPragerPlasticity, MaximumStressFailure, MonaghanSphEos, NaccPlasticity,
    NeoHookeanElasticity, RankinePlasticity, SnowPlasticity,
};
use crate::core::dynamics::ParticleData;
use crate::cuda::CudaVec;
use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub struct SharedCudaVec<T: DeviceCopy> {
    pub data: Arc<RwLock<CudaVec<T>>>,
}

impl<T: DeviceCopy> SharedCudaVec<T> {
    pub fn new(data: CudaVec<T>) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
        }
    }

    pub fn from_slice(data: &[T]) -> CudaResult<Self> {
        CudaVec::from_slice(data).map(Self::new)
    }
}

#[derive(Clone)]
pub enum CudaConstitutiveModel {
    CorotatedLinearElasticity(CorotatedLinearElasticity, SharedCudaVec<ParticleData>),
    NeoHookeanElasticity(NeoHookeanElasticity, SharedCudaVec<ParticleData>),
    EosMonaghanSph(MonaghanSphEos),
    Custom(u32),
}

impl CudaConstitutiveModel {
    pub fn from_core_model(
        model: CoreConstitutiveModel,
        data: SharedCudaVec<ParticleData>,
    ) -> Self {
        match model {
            CoreConstitutiveModel::CorotatedLinearElasticity(m) => {
                Self::CorotatedLinearElasticity(m, data)
            }
            CoreConstitutiveModel::NeoHookeanElasticity(m) => Self::NeoHookeanElasticity(m, data),
            CoreConstitutiveModel::EosMonaghanSph(m) => Self::EosMonaghanSph(m),
            CoreConstitutiveModel::Custom(id) => Self::Custom(id),
        }
    }

    pub fn to_gpu_model(&self) -> GpuConstitutiveModel {
        match self {
            Self::CorotatedLinearElasticity(m, buf) => {
                GpuConstitutiveModel::CorotatedLinearElasticity(
                    *m,
                    buf.data.read().unwrap().as_device_ptr(),
                )
            }
            Self::NeoHookeanElasticity(m, buf) => GpuConstitutiveModel::NeoHookeanElasticity(
                *m,
                buf.data.read().unwrap().as_device_ptr(),
            ),
            Self::EosMonaghanSph(m) => GpuConstitutiveModel::EosMonaghanSph(*m),
            Self::Custom(id) => GpuConstitutiveModel::Custom(*id),
        }
    }
}

#[derive(Clone)]
pub enum CudaPlasticModel {
    DruckerPrager(DruckerPragerPlasticity, SharedCudaVec<ParticleData>),
    Nacc(NaccPlasticity, SharedCudaVec<ParticleData>),
    Rankine(RankinePlasticity, SharedCudaVec<ParticleData>),
    Snow(SnowPlasticity, SharedCudaVec<ParticleData>),
    Custom(u32),
}

impl CudaPlasticModel {
    pub fn from_core_model(model: CorePlasticModel, data: SharedCudaVec<ParticleData>) -> Self {
        match model {
            CorePlasticModel::DruckerPrager(m) => Self::DruckerPrager(m, data),
            CorePlasticModel::Nacc(m) => Self::Nacc(m, data),
            CorePlasticModel::Rankine(m) => Self::Rankine(m, data),
            CorePlasticModel::Snow(m) => Self::Snow(m, data),
            CorePlasticModel::Custom(id) => Self::Custom(id),
        }
    }

    pub fn to_gpu_model(&self) -> GpuPlasticModel {
        match self {
            Self::DruckerPrager(m, buf) => {
                GpuPlasticModel::DruckerPrager(*m, buf.data.read().unwrap().as_device_ptr())
            }
            Self::Nacc(m, buf) => {
                GpuPlasticModel::Nacc(*m, buf.data.read().unwrap().as_device_ptr())
            }
            Self::Rankine(m, buf) => {
                GpuPlasticModel::Rankine(*m, buf.data.read().unwrap().as_device_ptr())
            }
            Self::Snow(m, buf) => {
                GpuPlasticModel::Snow(*m, buf.data.read().unwrap().as_device_ptr())
            }
            Self::Custom(id) => GpuPlasticModel::Custom(*id),
        }
    }
}

#[derive(Clone)]
pub enum CudaFailureModel {
    MaximumStress(MaximumStressFailure),
    Custom(u32),
}

impl CudaFailureModel {
    pub fn from_core_model(model: CoreFailureModel) -> Self {
        match model {
            CoreFailureModel::MaximumStress(m) => Self::MaximumStress(m),
            CoreFailureModel::Custom(id) => Self::Custom(id),
        }
    }

    pub fn to_gpu_model(&self) -> GpuFailureModel {
        match self {
            Self::MaximumStress(m) => GpuFailureModel::MaximumStress(*m),
            Self::Custom(id) => GpuFailureModel::Custom(*id),
        }
    }
}

pub struct CudaParticleModel {
    pub constitutive_model: CudaConstitutiveModel,
    pub plastic_model: Option<CudaPlasticModel>,
    pub failure_model: Option<CudaFailureModel>,
}

impl CudaParticleModel {
    pub fn to_gpu_model(&self) -> GpuParticleModel {
        GpuParticleModel {
            constitutive_model: self.constitutive_model.to_gpu_model(),
            plastic_model: self.plastic_model.as_ref().map(|m| m.to_gpu_model()),
            failure_model: self.failure_model.as_ref().map(|m| m.to_gpu_model()),
        }
    }
}
