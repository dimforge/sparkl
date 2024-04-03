use std::any::Any;
use std::sync::Arc;

use crate::core::prelude::CorePlasticModel;
use crate::dynamics::Particle;

pub trait AsAnyArc {
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any>;
}

impl<T: 'static> AsAnyArc for T {
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any> {
        self
    }
}

pub trait PlasticModel: Send + Sync + AsAnyArc {
    // TODO: this should just take the deformation gradient as input and ouptut
    //       the projected deformation gradient.
    fn update_particle(&self, particle: &mut Particle);
    fn to_core_model(&self) -> Option<CorePlasticModel>;
}
