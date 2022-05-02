use crate::dynamics::models::{CoreFailureModel, FailureModel, MaximumStressFailure};
use crate::dynamics::{Particle, ParticleModel};

impl FailureModel for MaximumStressFailure {
    fn particle_failed(&self, particle: &Particle, model: &ParticleModel) -> bool {
        let stress = model.constitutive_model.update_particle_stress(particle);
        self.particle_failed(&stress)
    }

    fn to_core_model(&self) -> Option<CoreFailureModel> {
        Some(CoreFailureModel::MaximumStress(*self))
    }
}
