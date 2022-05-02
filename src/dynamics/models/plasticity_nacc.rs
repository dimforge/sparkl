use crate::dynamics::models::{CorePlasticModel, NaccPlasticity, PlasticModel};
use crate::dynamics::Particle;

impl PlasticModel for NaccPlasticity {
    fn update_particle(&self, particle: &mut Particle) {
        self.update_particle(&mut particle.deformation_gradient, &mut particle.nacc_alpha)
    }

    fn to_core_model(&self) -> Option<CorePlasticModel> {
        Some(CorePlasticModel::Nacc(*self))
    }
}
