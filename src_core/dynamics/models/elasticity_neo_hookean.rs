use crate::dynamics::models::ActiveTimestepBounds;
use crate::dynamics::timestep::ElasticitySoundSpeedTimestepBound;
use crate::math::{Matrix, Real, Vector, DIM};
#[cfg(not(feature = "std"))]
use na::ComplexField;

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct NeoHookeanElasticity {
    pub cfl_coeff: Real,
    pub lambda: Real,
    pub mu: Real,
}

impl NeoHookeanElasticity {
    pub fn new(young_modulus: Real, poisson_ratio: Real) -> Self {
        let (lambda, mu) = crate::utils::lame_lambda_mu(young_modulus, poisson_ratio);

        Self {
            cfl_coeff: 0.5,
            lambda,
            mu,
        }
    }

    pub fn fpk_stress(&self, deformation_gradient: &Matrix<Real>, hardening: Real) -> Matrix<Real> {
        let j = deformation_gradient.determinant();
        let k = 2.0 / 3.0 * self.mu * hardening + self.lambda * hardening;
        let inv_f_tr = deformation_gradient
            .transpose()
            .try_inverse()
            .unwrap_or_else(Matrix::identity);
        let cauchy_green_strain = deformation_gradient * deformation_gradient.transpose();
        let deviatoric_stress = (self.mu * hardening * j.powf(-2.0 / (DIM as Real)))
            * crate::utils::deviatoric_part(&cauchy_green_strain)
            * inv_f_tr;
        let volumetric_stress = j * k / 2.0 * (j - 1.0 / j) * inv_f_tr;
        deviatoric_stress + volumetric_stress
    }

    pub fn kirchhoff_stress(
        &self,
        particle_phase: Real,
        particle_elastic_hardening: Real,
        particle_deformation_gradient: &Matrix<Real>,
    ) -> Matrix<Real> {
        let particle_phase_coeff = Self::phase_coeff(particle_phase);
        let j = particle_deformation_gradient.determinant();
        let k = 2.0 / 3.0 * self.mu * particle_elastic_hardening
            + self.lambda * particle_elastic_hardening;
        let cauchy_green_strain =
            particle_deformation_gradient * particle_deformation_gradient.transpose();
        let deviatoric_stress = (self.mu * particle_elastic_hardening)
            * j.powf(-2.0 / (DIM as Real))
            * crate::utils::deviatoric_part(&cauchy_green_strain);
        let volumetric_stress = k / 2.0 * (j * j - 1.0) * Matrix::identity();

        let (pos_part, neg_part) = if j >= 1.0 {
            (deviatoric_stress + volumetric_stress, Matrix::zeros())
        } else {
            (deviatoric_stress, volumetric_stress)
        };

        pos_part * particle_phase_coeff + neg_part
    }

    // https://dl.acm.org/doi/pdf/10.1145/3306346.3322949#subsection.3.2
    pub fn phase_coeff(phase: Real) -> Real {
        const R: Real = 0.001;
        (1.0 - R) * phase * phase + R
    }

    pub fn is_fluid(&self) -> bool {
        false
    }

    // General elastic density function: https://dl.acm.org/doi/pdf/10.1145/3306346.3322949#section.5 (8) and (9)
    // With degradation: https://dl.acm.org/doi/pdf/10.1145/3306346.3322949#subsection.3.2 (3)
    // With hardening: https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf#subsection.6.5 (87)
    pub fn elastic_energy_density(
        &self,
        particle_phase: Real,                // aka. c
        elastic_hardening: Real,             // aka. e^(xi (1 - J_P))
        deformation_gradient: &Matrix<Real>, // aka. F
    ) -> Real {
        let hardened_mu = self.mu * elastic_hardening;
        let hardened_lambda = self.lambda * elastic_hardening;

        // aka. bulk modulus
        let kappa = 2.0 / 3.0 * hardened_mu + hardened_lambda;

        let j = deformation_gradient.determinant();
        let alpha = -1. / DIM as Real;

        // aka. Psi_mu
        let deviatoric_part = hardened_mu / 2.
            * (((j.powf(alpha) * deformation_gradient).transpose()
                * (j.powf(alpha) * deformation_gradient))
                .trace()
                - DIM as Real);

        // aka. Psi_kappa
        let volumetric_part = kappa / 2. * ((j.powi(2) - 1.) / 2. - j.ln());

        // aka. (Psi_plus, Psi_minus)
        let (tensile_part, compressive_part) = if j >= 1. {
            (volumetric_part + deviatoric_part, 0.)
        } else {
            (volumetric_part, deviatoric_part)
        };

        Self::phase_coeff(particle_phase) * tensile_part + compressive_part
    }

    // Basically the same as [elastic_energy_density] but only the positive part
    pub fn pos_energy(
        &self,
        particle_phase: Real,
        particle_elastic_hardening: Real,
        particle_deformation_gradient: &Matrix<Real>,
    ) -> Real {
        let phase_coeff = Self::phase_coeff(particle_phase);
        let hardening = particle_elastic_hardening;

        let j = particle_deformation_gradient.determinant();
        let k = 2.0 / 3.0 * self.mu * hardening + self.lambda * hardening;
        let deviatoric_part = hardening * self.mu / 2.0
            * ((particle_deformation_gradient * particle_deformation_gradient.transpose()).trace()
                * j.powf(-2.0 / (DIM as Real))
                - DIM as Real);
        let volumetric_part = k / 2.0 * ((j * j - 1.0) / 2.0 - j.ln());

        // TODO: shouldn't this be free of any phase dependency?
        if j < 1.0 {
            deviatoric_part * phase_coeff
        } else {
            (deviatoric_part + volumetric_part) * particle_phase
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
        particle_elastic_hardening: Real,
        cell_width: Real,
    ) -> Real {
        let bulk_modulus = crate::utils::bulk_modulus_from_lame(self.lambda, self.mu);
        let shear_modulus = crate::utils::shear_modulus_from_lame(self.lambda, self.mu);

        let bound = ElasticitySoundSpeedTimestepBound {
            alpha: self.cfl_coeff,
            bulk_modulus: bulk_modulus * particle_elastic_hardening,
            shear_modulus: shear_modulus * particle_elastic_hardening,
        };
        bound.timestep_bound(particle_density0, particle_velocity, cell_width)
    }
}
