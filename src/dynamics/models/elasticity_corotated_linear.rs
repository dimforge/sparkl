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

    // https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf#subsection.6.3
    fn elastic_energy_density(&self, deformation_gradient: Matrix<Real>) -> Real {
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
