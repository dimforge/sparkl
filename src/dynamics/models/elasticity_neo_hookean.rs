use crate::dynamics::models::{
    ActiveTimestepBounds, ConstitutiveModel, CoreConstitutiveModel, NeoHookeanElasticity,
};
use crate::dynamics::Particle;
use crate::math::{Matrix, Real, DIM};

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

    // https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf#subsection.6.2
    fn elastic_energy_density(&self, deformation_gradient: Matrix<Real>) -> Real {
        let determinant_log = deformation_gradient.determinant().ln();
        self.mu / 2.
            * ((deformation_gradient.transpose() * deformation_gradient).trace() - DIM as Real)
            - self.mu * determinant_log
            + self.lambda / 2. * determinant_log.powi(2)
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
