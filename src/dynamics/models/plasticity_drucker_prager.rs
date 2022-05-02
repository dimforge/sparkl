use crate::dynamics::models::{CorePlasticModel, DruckerPragerPlasticity, PlasticModel};
use crate::dynamics::Particle;

impl PlasticModel for DruckerPragerPlasticity {
    fn update_particle(&self, particle: &mut Particle) {
        self.update_particle(
            particle.phase,
            &mut particle.deformation_gradient,
            &mut particle.plastic_deformation_gradient_det,
            &mut particle.plastic_hardening,
            &mut particle.log_vol_gain,
        )
    }

    fn to_core_model(&self) -> Option<CorePlasticModel> {
        Some(CorePlasticModel::DruckerPrager(*self))
    }
}
