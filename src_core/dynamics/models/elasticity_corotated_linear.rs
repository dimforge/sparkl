use crate::dynamics::models::ActiveTimestepBounds;
use crate::dynamics::timestep::ElasticitySoundSpeedTimestepBound;
use crate::math::{Matrix, Real, Vector};

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct CorotatedLinearElasticity {
    pub split_stress_on_failure: bool,
    pub cfl_coeff: Real,
    pub lambda: Real,
    pub mu: Real,
}

impl CorotatedLinearElasticity {
    pub fn new(young_modulus: Real, poisson_ratio: Real) -> Self {
        let (lambda, mu) = crate::utils::lame_lambda_mu(young_modulus, poisson_ratio);

        Self {
            split_stress_on_failure: true,
            cfl_coeff: 0.9,
            lambda,
            mu,
        }
    }

    pub fn kirchhoff_stress(
        &self,
        phase: Real,
        hardening: Real,
        deformation_gradient: &Matrix<Real>,
    ) -> Matrix<Real> {
        let j = deformation_gradient.determinant();
        let mut svd = deformation_gradient.svd_unordered(true, true);
        svd.singular_values.apply(|e| *e = *e - 1.0);

        if phase == 1.0 {
            (2.0 * self.mu * hardening)
                * svd.recompose().unwrap()
                * deformation_gradient.transpose()
                + (self.lambda * hardening * (j - 1.0) * j) * Matrix::identity()
        } else {
            let mut pos_def = svd;
            pos_def.singular_values.apply(|e| *e = e.max(0.0));
            let mut neg_def = svd;
            neg_def.singular_values.apply(|e| *e = e.min(0.0));

            let pos_dev_stress = 2.0
                * self.mu
                * hardening
                * pos_def.recompose().unwrap()
                * deformation_gradient.transpose();
            let neg_dev_stress = 2.0
                * self.mu
                * hardening
                * neg_def.recompose().unwrap()
                * deformation_gradient.transpose();

            let spherical_stress = (self.lambda * hardening * (j - 1.0) * j) * Matrix::identity();

            let (pos_part, neg_part) = if j < 1.0 {
                (pos_dev_stress, neg_dev_stress + spherical_stress)
            } else {
                (pos_dev_stress + spherical_stress, neg_dev_stress)
            };

            let phase_coeff = if self.split_stress_on_failure && phase == 0.0 {
                0.0
            } else {
                1.0
            };
            pos_part * phase_coeff + neg_part
        }
    }

    // General elastic density function: https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf#subsection.6.3 (49)
    pub fn elastic_energy_density(&self, deformation_gradient: Matrix<Real>) -> Real {
        let singular_values = deformation_gradient
            .svd_unordered(false, false)
            .singular_values;
        let determinant: Real = singular_values.iter().product();

        self.mu
            * singular_values
                .iter()
                .map(|sigma| (sigma - 1.).powi(2))
                .sum::<Real>()
            + self.lambda / 2. * (determinant - 1.).powi(2)
    }

    pub fn pos_energy(&self, deformation_gradient: Matrix<Real>, elastic_hardening: Real) -> Real {
        let j = deformation_gradient.determinant();
        let mut pos_def = deformation_gradient.svd_unordered(true, true); // TODO: why compute U and V?
        pos_def.singular_values.apply(|e| *e = (*e - 1.0).max(0.0));

        let pos_dev_part = self.mu * elastic_hardening * pos_def.singular_values.norm_squared();
        let spherical_part = self.lambda * elastic_hardening / 2.0 * (j - 1.0) * (j - 1.0);

        if j < 1.0 {
            pos_dev_part
        } else {
            pos_dev_part + spherical_part
        }
    }

    pub fn active_timestep_bounds(&self) -> ActiveTimestepBounds {
        ActiveTimestepBounds::CONSTITUTIVE_MODEL_BOUND
            | ActiveTimestepBounds::PARTICLE_VELOCITY_BOUND
            | ActiveTimestepBounds::DEFORMATION_GRADIENT_CHANGE_BOUND
    }

    pub fn timestep_bound(
        &self,
        particle_density0: Real,
        particle_velocity: &Vector<Real>,
        elastic_hardening: Real,
        cell_width: Real,
    ) -> Real {
        let bulk_modulus = crate::utils::bulk_modulus_from_lame(self.lambda, self.mu);
        let shear_modulus = crate::utils::shear_modulus_from_lame(self.lambda, self.mu);

        let bound = ElasticitySoundSpeedTimestepBound {
            alpha: self.cfl_coeff,
            bulk_modulus: bulk_modulus * elastic_hardening,
            shear_modulus: shear_modulus * elastic_hardening,
        };
        bound.timestep_bound(particle_density0, particle_velocity, cell_width)
    }

    pub fn is_fluid(&self) -> bool {
        false
    }
}
