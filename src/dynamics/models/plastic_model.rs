use crate::core::prelude::CorePlasticModel;
use crate::dynamics::Particle;

pub trait PlasticModel: Send + Sync {
    // TODO: this should just take the deformation gradient as input and ouptut
    //       the projected deformation gradient.
    fn update_particle(&self, particle: &mut Particle);
    fn to_core_model(&self) -> Option<CorePlasticModel>;
}
