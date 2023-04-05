use crate::dynamics::solver::SolverParameters;
use crate::math::Real;
use cust::{error::CudaResult, prelude::*};

use super::SingleGpuMpmContext;

pub trait CudaParticleKernelsLauncher {
    unsafe fn timestep_started(&mut self) -> CudaResult<()>;
    unsafe fn launch_estimate_particle_timestep_length(
        &self,
        context: &SingleGpuMpmContext,
        blocks: (u32, u32, u32),
        threads: (u32, u32, u32),
        min_dt: Real,
        remaining_time: Real,
    ) -> CudaResult<()>;

    unsafe fn launch_g2p2g(
        &self,
        params: &SolverParameters,
        context: &mut SingleGpuMpmContext,
        blocks: u32,
        threads: u32,
        timestep_length: Real,
        halo: bool,
        enable_cdf: bool,
    ) -> CudaResult<()>;
}

#[derive(Copy, Clone, Debug)]
pub struct DefaultCudaParticleKernelsLauncher;

impl CudaParticleKernelsLauncher for DefaultCudaParticleKernelsLauncher {
    unsafe fn timestep_started(&mut self) -> CudaResult<()> {
        // Nothing to do here.
        Ok(())
    }

    unsafe fn launch_estimate_particle_timestep_length(
        &self,
        context: &SingleGpuMpmContext,
        blocks: (u32, u32, u32),
        threads: (u32, u32, u32),
        min_dt: Real,
        remaining_time: Real,
    ) -> CudaResult<()> {
        let stream = &context.stream;
        let module = &context.module;

        launch!(
            module.estimate_timestep_length<<<blocks, threads, 0, stream>>>(
                min_dt,
                remaining_time,
                context.particles.particle_status.as_device_ptr(),
                context.particles.particle_volume.as_device_ptr(),
                context.particles.particle_vel.as_device_ptr(),
                context.particles.len(),
                context.models.buffer.as_device_ptr(),
                context.grid.cell_width(),
                context.timestep_length.as_device_ptr()
            )
        )
    }

    unsafe fn launch_g2p2g(
        &self,
        params: &SolverParameters,
        context: &mut SingleGpuMpmContext,
        blocks: u32,
        threads: u32,
        timestep_length: Real,
        halo: bool,
        enable_cdf: bool,
    ) -> CudaResult<()> {
        let stream = if halo {
            &context.halo_stream
        } else {
            &context.stream
        };
        let module = &context.module;

        launch!(
            module.g2p2g<<<blocks, threads, 0, stream>>>(
                timestep_length,
                context.rigid_world.device_elements(),
                context.particles.particle_status.as_device_ptr(),
                context.particles.particle_pos.as_device_ptr(),
                context.particles.particle_vel.as_device_ptr(),
                context.particles.particle_volume.as_device_ptr(),
                context.particles.particle_phase.as_device_ptr(),
                context.particles.particle_cdf.as_device_ptr(),
                context.particles.sorted_particle_ids.as_device_ptr(),
                context.models.buffer.as_device_ptr(),
                context.grid.curr_device_elements(),
                context.grid.next_device_elements(),
                params.damage_model,
                halo,
                enable_cdf
            )
        )
    }
}
