use crate::dynamics::models::MaximumStressFailure;
use crate::math::{Matrix, Real};

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub enum CoreFailureModel {
    MaximumStress(MaximumStressFailure),
    Custom(u32),
}

impl CoreFailureModel {
    pub fn particle_failed(&self, particle_stress: &Matrix<Real>) -> bool {
        match self {
            Self::MaximumStress(m) => m.particle_failed(particle_stress),
            Self::Custom(_) => false,
        }
    }
}
