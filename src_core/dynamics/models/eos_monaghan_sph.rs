use crate::dynamics::models::ActiveTimestepBounds;
use crate::math::{DecomposedTensor, Matrix, Real, Vector};

#[cfg(target_os = "cuda")]
use na::ComplexField;

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct MonaghanSphEos {
    pub pressure0: Real,
    pub gamma: u32,
    pub viscosity: Real,
    pub max_neg_pressure: Real,
}

impl MonaghanSphEos {
    pub fn new(pressure0: Real, gamma: u32, viscosity: Real) -> Self {
        Self {
            pressure0,
            gamma,
            viscosity,
            max_neg_pressure: 1.0,
        }
    }

    pub fn pressure(
        &self,
        particle_mass: Real,
        particle_volume0: Real,
        particle_density_fluid: Real,
    ) -> Real {
        let density0 = particle_mass / particle_volume0;
        (self.pressure0 * ((particle_density_fluid / density0).powi(self.gamma as i32) - 1.0))
            .max(-self.max_neg_pressure)
    }

    pub fn is_fluid(&self) -> bool {
        true
    }

    pub fn kirchhoff_stress(
        &self,
        particle_mass: Real,
        particle_volume0: Real,
        particle_density_fluid: Real,
        particle_fluid_deformation_gradient_det: Real,
        particle_velocity_gradient: &Matrix<Real>,
    ) -> Matrix<Real> {
        let mut stress = (-self.pressure(particle_mass, particle_volume0, particle_density_fluid)
            * particle_fluid_deformation_gradient_det)
            * Matrix::identity();

        if self.viscosity != 0.0 {
            let strain_rate = crate::utils::strain_rate(particle_velocity_gradient);
            let strain_rate = DecomposedTensor::decompose(&strain_rate);
            stress += (2.0 * self.viscosity * particle_fluid_deformation_gradient_det)
                * strain_rate.deviatoric_part;
        }

        stress
    }

    pub fn active_timestep_bounds(&self) -> ActiveTimestepBounds {
        ActiveTimestepBounds::CONSTITUTIVE_MODEL_BOUND
            | ActiveTimestepBounds::PARTICLE_VELOCITY_BOUND
            | ActiveTimestepBounds::PARTICLE_DISPLACEMENT_BOUND
            | ActiveTimestepBounds::SINGLE_PARTICLE_STABILITY_BOUND
    }

    pub fn timestep_bound(
        &self,
        particle_fluid_deformation_gradient_det: Real,
        particle_mass: Real,
        particle_volume0: Real,
        particle_density_fluid: Real,
        particle_velocity: &Vector<Real>,
        cell_width: Real,
    ) -> Real {
        // Single-particle criteria.
        let j = particle_fluid_deformation_gradient_det;
        let density0 = particle_mass / particle_volume0;
        let k = 6.0; // For quadratic splines.
        let d = crate::math::DIM as Real;
        let pressure = -self.pressure(particle_mass, particle_volume0, particle_density_fluid);

        let single_particle_dt =
            (cell_width / j) * (density0 * (j - 1.0) / (k * pressure * d)).sqrt();

        // CFL
        let density_fluctuation = 0.1;
        let c_square = particle_velocity.norm_squared().max(1.0) / density_fluctuation;
        let cfl_dt = cell_width / c_square.sqrt();

        // Result
        single_particle_dt.min(cfl_dt)
        // cfl_dt
    }
}
