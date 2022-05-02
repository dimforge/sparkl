use crate::DevicePointer;
use sparkl_core::dynamics::models::{
    DruckerPragerPlasticity, NaccPlasticity, RankinePlasticity, SnowPlasticity,
};
use sparkl_core::dynamics::ParticleData;
use sparkl_core::math::Real;
use sparkl_core::prelude::ParticleVolume;

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(cust_core::DeviceCopy, Copy, Clone, PartialEq)]
#[repr(C)]
pub enum GpuPlasticModel {
    DruckerPrager(DruckerPragerPlasticity, DevicePointer<ParticleData>),
    Nacc(NaccPlasticity, DevicePointer<ParticleData>),
    Rankine(RankinePlasticity, DevicePointer<ParticleData>),
    Snow(SnowPlasticity, DevicePointer<ParticleData>),
    Custom(u32),
}

impl GpuPlasticModel {
    pub unsafe fn update_particle(
        &self,
        particle_id: u32,
        particle_volume: &mut ParticleVolume,
        particle_phase: Real,
    ) {
        match self {
            Self::DruckerPrager(dp, d) => {
                let plastic_hardening =
                    &mut (*d.as_mut_ptr().add(particle_id as usize)).plastic_hardening;
                let log_vol_gain = &mut (*d.as_mut_ptr().add(particle_id as usize)).log_vol_gain;

                dp.update_particle(
                    particle_phase,
                    &mut particle_volume.deformation_gradient,
                    &mut particle_volume.plastic_deformation_gradient_det,
                    plastic_hardening,
                    log_vol_gain,
                );
            }
            Self::Nacc(nacc, d) => {
                let nacc_alpha = &mut (*d.as_mut_ptr().add(particle_id as usize)).nacc_alpha;
                nacc.update_particle(&mut particle_volume.deformation_gradient, nacc_alpha);
            }
            Self::Rankine(r, d) => {
                let plastic_hardening =
                    &mut (*d.as_mut_ptr().add(particle_id as usize)).plastic_hardening;
                r.update_particle(&mut particle_volume.deformation_gradient, plastic_hardening);
            }
            Self::Snow(s, d) => {
                let elastic_hardening =
                    &mut (*d.as_mut_ptr().add(particle_id as usize)).elastic_hardening;
                s.update_particle(
                    &mut particle_volume.deformation_gradient,
                    elastic_hardening,
                    &mut particle_volume.plastic_deformation_gradient_det,
                );
            }
            Self::Custom(_) => {}
        }
    }
}
