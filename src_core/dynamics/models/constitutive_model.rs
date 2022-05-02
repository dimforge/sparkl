use crate::prelude::{CorotatedLinearElasticity, MonaghanSphEos, NeoHookeanElasticity};

bitflags::bitflags! {
    pub struct ActiveTimestepBounds: u8 {
        const NONE = 0;
        const CONSTITUTIVE_MODEL_BOUND = 1 << 0;
        const PARTICLE_VELOCITY_BOUND = 1 << 1;
        const PARTICLE_DISPLACEMENT_BOUND = 1 << 2;
        const DEFORMATION_GRADIENT_CHANGE_BOUND = 1 << 3;
        const SINGLE_PARTICLE_STABILITY_BOUND = 1 << 4;
    }
}

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub enum CoreConstitutiveModel {
    CorotatedLinearElasticity(CorotatedLinearElasticity),
    NeoHookeanElasticity(NeoHookeanElasticity),
    EosMonaghanSph(MonaghanSphEos),
    Custom(u32),
}
