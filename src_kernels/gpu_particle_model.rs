use crate::DevicePointer;
use crate::{GpuConstitutiveModel, GpuPlasticModel};
use sparkl_core::dynamics::models::CorotatedLinearElasticity;
use sparkl_core::prelude::ParticleData;

pub type GpuFailureModel = sparkl_core::dynamics::models::CoreFailureModel;

#[derive(cust_core::DeviceCopy, Copy, Clone, PartialEq)]
#[repr(C)]
pub struct GpuParticleModel {
    pub constitutive_model: GpuConstitutiveModel,
    pub plastic_model: Option<GpuPlasticModel>,
    pub failure_model: Option<GpuFailureModel>,
}

impl GpuParticleModel {
    pub fn get_particle_data(&self) -> Option<DevicePointer<ParticleData>> {
        self.constitutive_model.get_particle_data().or_else(|| {
            if let Some(plastic_model) = &self.plastic_model {
                plastic_model.get_particle_data()
            } else {
                None
            }
        })
    }
}

impl Default for GpuParticleModel {
    fn default() -> Self {
        Self {
            constitutive_model: GpuConstitutiveModel::CorotatedLinearElasticity(
                CorotatedLinearElasticity::new(1.0e6, 0.2),
                DevicePointer::null(),
            ),
            plastic_model: None,
            failure_model: None,
        }
    }
}
