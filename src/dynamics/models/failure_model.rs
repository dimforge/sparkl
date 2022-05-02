use crate::dynamics::models::CoreFailureModel;
use crate::dynamics::{Particle, ParticleModel};

pub trait FailureModel: Send + Sync {
    fn particle_failed(&self, particle: &Particle, model: &ParticleModel) -> bool;
    fn to_core_model(&self) -> Option<CoreFailureModel>;
}
