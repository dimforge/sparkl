use crate::dynamics::models::{
    ActiveTimestepBounds, ConstitutiveModel, CoreConstitutiveModel, NeoHookeanElasticity,
};
use crate::dynamics::Particle;
use crate::math::{Matrix, Real};

impl ConstitutiveModel for NeoHookeanElasticity {
    fn is_fluid(&self) -> bool {
        false
    }

    fn update_particle_stress(&self, particle: &Particle) -> Matrix<Real> {
        self.kirchhoff_stress(
            particle.phase,
            particle.elastic_hardening,
            &particle.deformation_gradient,
        )
    }

    fn pos_energy(&self, particle: &Particle) -> Real {
        self.pos_energy(
            particle.phase,
            particle.elastic_hardening,
            &particle.deformation_gradient,
        )
    }

    fn active_timestep_bounds(&self) -> ActiveTimestepBounds {
        self.active_timestep_bounds()
    }

    fn timestep_bound(&self, particle: &Particle, cell_width: Real) -> Real {
        self.timestep_bound(
            particle.density0(),
            &particle.velocity,
            particle.elastic_hardening,
            cell_width,
        )
    }

    fn to_core_model(&self) -> Option<CoreConstitutiveModel> {
        Some(CoreConstitutiveModel::NeoHookeanElasticity(*self))
    }
}
