use crate::dynamics::models::{
    ActiveTimestepBounds, CoreConstitutiveModel, CorotatedLinearElasticity,
};
use crate::dynamics::{models::ConstitutiveModel, Particle};
use crate::math::{Matrix, Real};

impl ConstitutiveModel for CorotatedLinearElasticity {
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

    fn elastic_energy_density(&self, particle: &Particle) -> Real {
        self.elastic_energy_density(particle.deformation_gradient)
    }

    fn pos_energy(&self, particle: &Particle) -> Real {
        self.pos_energy(particle.deformation_gradient, particle.elastic_hardening)
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
        Some(CoreConstitutiveModel::CorotatedLinearElasticity(*self))
    }
}
