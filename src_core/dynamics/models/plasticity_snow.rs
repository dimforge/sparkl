use crate::math::{Matrix, Real, Vector};

#[cfg(target_os = "cuda")]
use na::ComplexField;

/// The Snow plasticity model.
#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct SnowPlasticity {
    pub min_epsilon: Real,
    pub max_epsilon: Real,
    pub hardening_coeff: Real, // Typically between 3 and 10
}

impl SnowPlasticity {
    pub fn new() -> Self {
        SnowPlasticity {
            min_epsilon: 2.5e-2,
            max_epsilon: 4.5e-3,
            hardening_coeff: 10.0,
        }
    }

    pub fn project_deformation_gradient(&self, singular_values: Vector<Real>) -> Vector<Real> {
        singular_values.map(|e| e.max(1.0 - self.min_epsilon).min(1.0 + self.max_epsilon))
    }

    pub fn update_particle(
        &self,
        particle_deformation_gradient: &mut Matrix<Real>,
        particle_elastic_hardening: &mut Real,
        particle_plastic_deformation_gradient_det: &mut Real,
    ) {
        let mut svd = particle_deformation_gradient.svd_unordered(true, true);

        let new_sig_vals = self.project_deformation_gradient(svd.singular_values);

        *particle_plastic_deformation_gradient_det *=
            svd.singular_values.product() / new_sig_vals.product();

        *particle_elastic_hardening =
            (self.hardening_coeff * (1.0 - *particle_plastic_deformation_gradient_det)).exp();

        svd.singular_values = new_sig_vals;
        *particle_deformation_gradient = svd.recompose().unwrap();
    }
}
