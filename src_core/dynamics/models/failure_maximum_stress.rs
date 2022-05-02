use crate::math::{Matrix, Real};

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct MaximumStressFailure {
    pub max_principal_stress: Real,
    pub max_shear_stress: Real,
}

impl MaximumStressFailure {
    pub fn new(max_principal_stress: Real, max_shear_stress: Real) -> Self {
        Self {
            max_principal_stress,
            max_shear_stress,
        }
    }

    pub fn particle_failed(&self, particle_stress: &Matrix<Real>) -> bool {
        if let Some(eig) = particle_stress.try_symmetric_eigen(1.0e-6, 100) {
            let min = eig.eigenvalues.min();
            let max = eig.eigenvalues.max();
            max > self.max_principal_stress || (max - min) / 2.0 > self.max_shear_stress
        } else {
            false
        }
    }
}
