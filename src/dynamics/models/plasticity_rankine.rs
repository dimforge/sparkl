use crate::dynamics::models::{CorePlasticModel, PlasticModel, RankinePlasticity};
use crate::dynamics::Particle;

impl PlasticModel for RankinePlasticity {
    fn update_particle(&self, particle: &mut Particle) {
        self.update_particle(
            &mut particle.deformation_gradient,
            &mut particle.plastic_hardening,
        )
    }

    fn to_core_model(&self) -> Option<CorePlasticModel> {
        Some(CorePlasticModel::Rankine(*self))
    }
}
