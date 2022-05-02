use crate::dynamics::models::{CorePlasticModel, PlasticModel, SnowPlasticity};
use crate::dynamics::Particle;

impl PlasticModel for SnowPlasticity {
    fn update_particle(&self, particle: &mut Particle) {
        self.update_particle(
            &mut particle.deformation_gradient,
            &mut particle.elastic_hardening,
            &mut particle.plastic_deformation_gradient_det,
        )
    }

    fn to_core_model(&self) -> Option<CorePlasticModel> {
        Some(CorePlasticModel::Snow(*self))
    }
}
