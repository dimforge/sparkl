use crate::dynamics::models::{
    ActiveTimestepBounds, ConstitutiveModel, CoreConstitutiveModel, MonaghanSphEos,
};
use crate::dynamics::Particle;
use crate::math::{Matrix, Real};

impl ConstitutiveModel for MonaghanSphEos {
    fn is_fluid(&self) -> bool {
        true
    }

    fn update_particle_stress(&self, particle: &Particle) -> Matrix<Real> {
        self.kirchhoff_stress(
            particle.mass,
            particle.volume0,
            particle.density_fluid(),
            particle.plastic_deformation_gradient_det,
            &particle.velocity_gradient,
        )
    }

    fn active_timestep_bounds(&self) -> ActiveTimestepBounds {
        self.active_timestep_bounds()
    }

    fn timestep_bound(&self, particle: &Particle, cell_width: Real) -> Real {
        self.timestep_bound(
            particle.fluid_deformation_gradient_det(),
            particle.mass,
            particle.volume0,
            particle.density_fluid(),
            &particle.velocity,
            cell_width,
        )
    }

    fn to_core_model(&self) -> Option<CoreConstitutiveModel> {
        Some(CoreConstitutiveModel::EosMonaghanSph(*self))
    }
}
