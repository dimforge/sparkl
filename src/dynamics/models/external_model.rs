use crate::core::dynamics::models::{ActiveTimestepBounds, CoreConstitutiveModel};
use crate::dynamics::models::ConstitutiveModel;
use crate::dynamics::Particle;
use crate::math::{Matrix, Real};

/// A constitutive model that does nothing.
pub struct ExternalModel(pub u32);

impl ConstitutiveModel for ExternalModel {
    fn is_fluid(&self) -> bool {
        false
    }

    fn update_particle_stress(&self, _particle: &Particle) -> Matrix<Real> {
        Matrix::zeros()
    }

    fn update_internal_energy_and_pressure(
        &self,
        _particle: &mut Particle,
        _dt: Real,
        _cell_width: Real,
    ) {
    }

    fn active_timestep_bounds(&self) -> ActiveTimestepBounds {
        ActiveTimestepBounds::NONE
    }

    fn timestep_bound(&self, _particle: &Particle, _cell_width: Real) -> Real {
        Real::MAX
    }

    fn to_core_model(&self) -> Option<CoreConstitutiveModel> {
        Some(CoreConstitutiveModel::Custom(self.0))
    }
}
