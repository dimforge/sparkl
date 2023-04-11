use super::CudaVec;
use crate::core::prelude::{
    ParticleCdf, ParticleData, ParticlePhase, ParticlePosition, ParticleStatus, ParticleVelocity,
    ParticleVolume,
};
use crate::dynamics::Particle;
use cust::{
    error::CudaResult,
    memory::{CopyDestination, DeviceBuffer},
};
use std::ops::Range;

pub struct CudaParticleSet {
    pub particle_status: CudaVec<ParticleStatus>,
    pub particle_pos: CudaVec<ParticlePosition>,
    pub particle_vel: CudaVec<ParticleVelocity>,
    pub particle_volume: CudaVec<ParticleVolume>,
    pub particle_phase: CudaVec<ParticlePhase>,
    pub particle_cdf: CudaVec<ParticleCdf>,
    pub sorted_particle_ids: DeviceBuffer<u32>,
}

impl Default for CudaParticleSet {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl CudaParticleSet {
    pub fn new() -> CudaResult<Self> {
        Self::from_particles(&[], &[], &[], &[], &[], &[])
    }

    // TODO: make this async?
    pub fn from_particles(
        particle_status: &[ParticleStatus],
        particle_pos: &[ParticlePosition],
        particle_vel: &[ParticleVelocity],
        particle_volume: &[ParticleVolume],
        particle_phase: &[ParticlePhase],
        particle_cdf: &[ParticleCdf],
    ) -> CudaResult<Self> {
        assert_eq!(
            particle_status.len(),
            particle_pos.len(),
            "All attribute buffer must have the same length."
        );
        assert_eq!(
            particle_status.len(),
            particle_vel.len(),
            "All attribute buffer must have the same length."
        );
        assert_eq!(
            particle_status.len(),
            particle_volume.len(),
            "All attribute buffer must have the same length."
        );
        assert_eq!(
            particle_status.len(),
            particle_phase.len(),
            "All attribute buffer must have the same length."
        );
        assert_eq!(
            particle_status.len(),
            particle_cdf.len(),
            "All attribute buffer must have the same length."
        );

        let sorted_particle_ids = DeviceBuffer::zeroed(particle_status.len())?;

        Ok(Self {
            particle_status: CudaVec::from_slice(particle_status)?,
            particle_pos: CudaVec::from_slice(particle_pos)?,
            particle_vel: CudaVec::from_slice(particle_vel)?,
            particle_volume: CudaVec::from_slice(particle_volume)?,
            particle_phase: CudaVec::from_slice(particle_phase)?,
            particle_cdf: CudaVec::from_slice(particle_cdf)?,
            sorted_particle_ids,
        })
    }

    pub fn len(&self) -> usize {
        self.particle_status.len()
    }

    pub fn capacity(&self) -> usize {
        self.particle_status.capacity()
    }

    pub fn truncate(&mut self, new_len: usize) {
        self.particle_status.truncate(new_len);
        self.particle_pos.truncate(new_len);
        self.particle_vel.truncate(new_len);
        self.particle_volume.truncate(new_len);
        self.particle_phase.truncate(new_len);
        self.particle_cdf.truncate(new_len);
    }

    pub fn is_empty(&self) -> bool {
        self.particle_status.is_empty()
    }

    pub fn read_positions(&self) -> CudaResult<Vec<ParticlePosition>> {
        self.particle_pos.to_vec()
    }

    pub fn read_velocities(&self) -> CudaResult<Vec<ParticleVelocity>> {
        self.particle_vel.to_vec()
    }

    pub fn read_statuses(&self) -> CudaResult<Vec<ParticleStatus>> {
        self.particle_status.to_vec()
    }

    pub fn read_volumes(&self) -> CudaResult<Vec<ParticleVolume>> {
        self.particle_volume.to_vec()
    }

    pub fn read_phase(&self) -> CudaResult<Vec<ParticlePhase>> {
        self.particle_phase.to_vec()
    }

    pub fn read_cdf(&self) -> CudaResult<Vec<ParticleCdf>> {
        self.particle_cdf.to_vec()
    }

    pub fn read_sorted_particle_ids(&self) -> CudaResult<Vec<u32>> {
        let mut out = vec![0; self.len()];
        self.sorted_particle_ids
            .index(..self.len())
            .copy_to(&mut out)?;
        Ok(out)
    }

    pub fn remove_range(&mut self, range: Range<usize>) -> CudaResult<()> {
        self.particle_status.remove_range(range.clone())?;
        self.particle_pos.remove_range(range.clone())?;
        self.particle_vel.remove_range(range.clone())?;
        self.particle_volume.remove_range(range.clone())?;
        self.particle_phase.remove_range(range.clone())?;
        self.particle_cdf.remove_range(range.clone())?;

        Ok(())
    }

    /// Inserts a set of particles, as well as their attributes into this set.
    ///
    /// The `particles.len()` must be equal to all the `particle_attributes[j].len()`.
    /// In other words, all the particles must share the same set of attributes.
    pub fn append(
        &mut self,
        particle_status: &[ParticleStatus],
        particle_pos: &[ParticlePosition],
        particle_vel: &[ParticleVelocity],
        particle_volume: &[ParticleVolume],
        particle_phase: &[ParticlePhase],
        particle_cdf: &[ParticleCdf],
    ) -> CudaResult<()> {
        assert_eq!(
            particle_status.len(),
            particle_pos.len(),
            "All attribute buffer must have the same length."
        );
        assert_eq!(
            particle_status.len(),
            particle_vel.len(),
            "All attribute buffer must have the same length."
        );
        assert_eq!(
            particle_status.len(),
            particle_volume.len(),
            "All attribute buffer must have the same length."
        );
        assert_eq!(
            particle_status.len(),
            particle_phase.len(),
            "All attribute buffer must have the same length."
        );
        assert_eq!(
            particle_status.len(),
            particle_cdf.len(),
            "All attribute buffer must have the same length."
        );

        let prev_capacity = self.capacity();

        self.particle_status.append(&particle_status)?;
        self.particle_pos.append(&particle_pos)?;
        self.particle_vel.append(&particle_vel)?;
        self.particle_volume.append(&particle_volume)?;
        self.particle_phase.append(&particle_phase)?;
        self.particle_cdf.append(&particle_cdf)?;

        if prev_capacity != self.capacity() {
            self.sorted_particle_ids = DeviceBuffer::zeroed(self.capacity())?;
        }

        Ok(())
    }
}

pub struct ParticleComponents {
    pub position: Vec<ParticlePosition>,
    pub velocity: Vec<ParticleVelocity>,
    pub volume: Vec<ParticleVolume>,
    pub status: Vec<ParticleStatus>,
    pub phase: Vec<ParticlePhase>,
    pub cdf: Vec<ParticleCdf>,
    pub data: Vec<ParticleData>,
}

pub fn extract_particles_components(particles: &[Particle]) -> ParticleComponents {
    let position: Vec<_> = particles
        .iter()
        .map(|p| ParticlePosition { point: p.position })
        .collect();

    let velocity: Vec<_> = particles
        .iter()
        .map(|p| ParticleVelocity { vector: p.velocity })
        .collect();

    let volume: Vec<_> = particles
        .iter()
        .map(|p| ParticleVolume {
            mass: p.mass,
            volume0: p.volume0,
            radius0: p.radius0,
            deformation_gradient: p.deformation_gradient,
            plastic_deformation_gradient_det: p.plastic_deformation_gradient_det,
        })
        .collect();

    let status: Vec<_> = particles
        .iter()
        .map(|p| ParticleStatus {
            failed: p.failed,
            is_static: p.is_static,
            kinematic_vel_enabled: p.kinematic_vel.is_some(),
            kinematic_vel: p.kinematic_vel.unwrap_or(na::zero()),
            model_index: p.model.into_raw_parts().0 as usize,
        })
        .collect();

    let phase: Vec<_> = particles
        .iter()
        .map(|p| ParticlePhase {
            phase: p.phase,
            psi_pos: p.psi_pos,
        })
        .collect();

    let cdf: Vec<_> = particles
        .iter()
        .map(|p| ParticleCdf {
            color: p.color,
            distance: p.distance,
            normal: p.normal,
        })
        .collect();

    let data: Vec<_> = particles
        .iter()
        .map(|p| ParticleData {
            nacc_alpha: p.nacc_alpha,
            plastic_hardening: p.plastic_hardening,
            elastic_hardening: p.elastic_hardening,
            log_vol_gain: p.log_vol_gain,
        })
        .collect();

    ParticleComponents {
        position,
        velocity,
        volume,
        status,
        phase,
        cdf,
        data,
    }
}
