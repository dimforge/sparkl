use crate::math::{Real, Vector};

#[cfg(target_os = "cuda")]
use na::ComplexField;

#[derive(Copy, Clone, Debug)]
pub struct ElasticitySoundSpeedTimestepBound {
    pub alpha: Real,
    pub bulk_modulus: Real,
    pub shear_modulus: Real,
}

impl ElasticitySoundSpeedTimestepBound {
    pub fn new(alpha: Real, young_modulus: Real, poisson_ratio: Real) -> Self {
        Self {
            alpha,
            bulk_modulus: crate::utils::bulk_modulus(young_modulus, poisson_ratio),
            shear_modulus: crate::utils::shear_modulus(young_modulus, poisson_ratio),
        }
    }

    pub fn timestep_bound(
        &self,
        density0: Real,
        velocity: &Vector<Real>,
        cell_width: Real,
    ) -> Real {
        let c_dir = ((self.bulk_modulus + 4.0 / 3.0 * self.shear_modulus)
            * 1.0 // particle.deformation_gradient.determinant()
            / density0)
            .sqrt();
        let max_denom = velocity.norm().max(c_dir);
        self.alpha * cell_width / max_denom
    }
}
