use crate::DevicePointer;
use sparkl_core::dynamics::{ParticleData, ParticleVolume};
use sparkl_core::math::{Matrix, Real};
use sparkl_core::prelude::{
    ActiveTimestepBounds, CorotatedLinearElasticity, MonaghanSphEos, NeoHookeanElasticity,
    ParticleVelocity,
};

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(cust_core::DeviceCopy, Copy, Clone, PartialEq)]
#[repr(C)]
pub enum GpuConstitutiveModel {
    CorotatedLinearElasticity(CorotatedLinearElasticity, DevicePointer<ParticleData>),
    NeoHookeanElasticity(NeoHookeanElasticity, DevicePointer<ParticleData>),
    EosMonaghanSph(MonaghanSphEos),
    Custom(u32),
}

impl GpuConstitutiveModel {
    pub fn get_particle_data(&self) -> Option<DevicePointer<ParticleData>> {
        match self {
            GpuConstitutiveModel::CorotatedLinearElasticity(_, pointer) => Some(*pointer),
            GpuConstitutiveModel::NeoHookeanElasticity(_, pointer) => Some(*pointer),
            GpuConstitutiveModel::EosMonaghanSph(_) => None,
            GpuConstitutiveModel::Custom(_) => None,
        }
    }

    pub fn is_fluid(&self) -> bool {
        match self {
            Self::CorotatedLinearElasticity(m, _) => m.is_fluid(),
            Self::NeoHookeanElasticity(m, _) => m.is_fluid(),
            Self::EosMonaghanSph(m) => m.is_fluid(),
            Self::Custom(_) => false,
        }
    }
    pub unsafe fn pos_energy(
        &self,
        particle_id: u32,
        particle_volume: &ParticleVolume,
        particle_phase: Real,
    ) -> Real {
        match self {
            Self::CorotatedLinearElasticity(m, d) => {
                let hardening = (*d.as_ptr().add(particle_id as usize)).elastic_hardening;
                m.pos_energy(particle_volume.deformation_gradient, hardening)
            }
            Self::NeoHookeanElasticity(m, d) => {
                let hardening = (*d.as_ptr().add(particle_id as usize)).elastic_hardening;
                m.pos_energy(
                    particle_phase,
                    hardening,
                    &particle_volume.deformation_gradient,
                )
            }
            Self::EosMonaghanSph(..) => 0.0,
            Self::Custom(_) => 0.0,
        }
    }

    pub unsafe fn kirchhoff_stress(
        &self,
        particle_id: u32,
        particle_volume: &ParticleVolume,
        particle_phase: Real,
        velocity_gradient: &Matrix<Real>,
    ) -> Matrix<Real> {
        match self {
            Self::CorotatedLinearElasticity(m, d) => {
                let hardening = (*d.as_ptr().add(particle_id as usize)).elastic_hardening;
                m.kirchhoff_stress(
                    particle_phase,
                    hardening,
                    &particle_volume.deformation_gradient,
                )
            }
            Self::NeoHookeanElasticity(m, d) => {
                let hardening = (*d.as_ptr().add(particle_id as usize)).elastic_hardening;
                m.kirchhoff_stress(
                    particle_phase,
                    hardening,
                    &particle_volume.deformation_gradient,
                )
            }
            Self::EosMonaghanSph(m) => m.kirchhoff_stress(
                particle_volume.mass,
                particle_volume.volume0,
                particle_volume.density_fluid(),
                particle_volume.plastic_deformation_gradient_det,
                &velocity_gradient,
            ),
            Self::Custom(_) => Matrix::zeros(),
        }
    }

    pub unsafe fn update_internal_energy_and_pressure(
        &self,
        _particle_id: u32,
        _particle_volume: &ParticleVolume,
        _dt: Real,
        _cell_width: Real,
        _velocity_gradient: &Matrix<Real>,
    ) {
        match self {
            Self::CorotatedLinearElasticity(..)
            | Self::NeoHookeanElasticity(..)
            | Self::EosMonaghanSph(..)
            | Self::Custom(_) => {}
        }
    }

    pub fn active_timestep_bounds(&self) -> ActiveTimestepBounds {
        match self {
            Self::CorotatedLinearElasticity(m, _) => m.active_timestep_bounds(),
            Self::NeoHookeanElasticity(m, _) => m.active_timestep_bounds(),
            Self::EosMonaghanSph(m) => m.active_timestep_bounds(),
            Self::Custom(_) => ActiveTimestepBounds::NONE,
        }
    }

    pub unsafe fn timestep_bound(
        &self,
        particle_id: u32,
        particle_volume: &ParticleVolume,
        particle_vel: &ParticleVelocity,
        cell_width: Real,
    ) -> Real {
        match self {
            Self::CorotatedLinearElasticity(m, d) => {
                let hardening = (*d.as_ptr().add(particle_id as usize)).elastic_hardening;
                m.timestep_bound(
                    particle_volume.density0(),
                    &particle_vel.vector,
                    hardening,
                    cell_width,
                )
            }
            Self::NeoHookeanElasticity(m, d) => {
                let hardening = (*d.as_ptr().add(particle_id as usize)).elastic_hardening;
                m.timestep_bound(
                    particle_volume.density0(),
                    &particle_vel.vector,
                    cell_width,
                    hardening,
                )
            }
            Self::EosMonaghanSph(m) => m.timestep_bound(
                particle_volume.fluid_deformation_gradient_det(),
                particle_volume.mass,
                particle_volume.volume0,
                particle_volume.density_fluid(),
                &particle_vel.vector,
                cell_width,
            ),
            Self::Custom(_) => 0.0,
        }
    }
}
