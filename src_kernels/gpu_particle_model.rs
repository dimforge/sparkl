use crate::DevicePointer;
use crate::{GpuConstitutiveModel, GpuPlasticModel};
use sparkl_core::dynamics::models::CorotatedLinearElasticity;

pub type GpuFailureModel = sparkl_core::dynamics::models::CoreFailureModel;

#[derive(cust_core::DeviceCopy, Copy, Clone, PartialEq)]
#[repr(C)]
pub struct GpuParticleModel {
    pub constitutive_model: GpuConstitutiveModel,
    pub plastic_model: Option<GpuPlasticModel>,
    pub failure_model: Option<GpuFailureModel>,
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
