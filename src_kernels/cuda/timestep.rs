use crate::cuda::{AtomicInt, DefaultParticleUpdater};
use crate::gpu_timestep::GpuTimestepLength;
use crate::GpuParticleModel;
use sparkl_core::dynamics::{ParticleStatus, ParticleVelocity, ParticleVolume};
use sparkl_core::math::Real;

#[cfg(target_os = "cuda")]
use na::ComplexField;
use sparkl_core::prelude::ActiveTimestepBounds;

use super::ParticleUpdater;

#[cuda_std::kernel]
pub unsafe fn estimate_timestep_length(
    min_dt: Real,
    max_dt: Real,
    particle_status: *const ParticleStatus,
    particle_volume: *const ParticleVolume,
    particle_vel: *const ParticleVelocity,
    num_particles: usize,
    models: *mut GpuParticleModel,
    cell_width: Real,
    timestep_length: *mut GpuTimestepLength,
) {
    estimate_timestep_length_generic(
        min_dt,
        max_dt,
        particle_status,
        particle_volume,
        particle_vel,
        num_particles,
        cell_width,
        timestep_length,
        DefaultParticleUpdater { models },
    )
}

pub unsafe fn estimate_timestep_length_generic(
    min_dt: Real,
    max_dt: Real,
    particle_status: *const ParticleStatus,
    particle_volume: *const ParticleVolume,
    particle_vel: *const ParticleVelocity,
    num_particles: usize,
    cell_width: Real,
    timestep_length: *mut GpuTimestepLength,
    updater: impl ParticleUpdater,
) {
    let i = cuda_std::thread::index();

    if i < num_particles as u32 {
        let particle_status = &*particle_status.add(i as usize);
        let particle_volume = &*particle_volume.add(i as usize);
        let particle_vel = &*particle_vel.add(i as usize);

        timestep_length_for_particle(
            min_dt,
            max_dt,
            i,
            particle_status,
            particle_volume,
            particle_vel,
            cell_width,
            &mut (*timestep_length),
            updater,
        );
    }
}

unsafe fn timestep_length_for_particle(
    min_dt: Real,
    max_dt: Real,
    particle_id: u32,
    particle_status: &ParticleStatus,
    particle_volume: &ParticleVolume,
    particle_vel: &ParticleVelocity,
    cell_width: Real,
    result: &mut GpuTimestepLength,
    updater: impl ParticleUpdater,
) {
    if particle_status.failed {
        return;
    }

    // let d = (cell_width * cell_width) / 4.0;

    let mut dt: Real = max_dt;

    let (active_timestep_bounds, candidate_dt) = updater.estimate_particle_timestep_length(
        cell_width,
        particle_id,
        particle_status,
        particle_volume,
        particle_vel,
    );
    dt = dt.min(candidate_dt);

    if active_timestep_bounds.contains(ActiveTimestepBounds::PARTICLE_VELOCITY_BOUND) {
        // Velocity-based restriction.
        // let norm_b = d * velocity_gradient.norm();
        // let apic_v = norm_b * 6.0 * (DIM as Real).sqrt() / cell_width;
        let v = particle_vel.vector.norm(); //  + apic_v;

        dt = dt.min(cell_width / v);
    }

    if dt < min_dt && max_dt > min_dt {
        dt = min_dt;
    }

    let candidate = GpuTimestepLength::from_sec(dt);
    result.0.global_red_min(candidate.0);
}
