use crate::cuda::{
    CudaColliderSet, CudaParticleKernelsLauncher, CudaParticleModelSet, CudaParticleSet,
    CudaSparseGrid, DefaultCudaParticleKernelsLauncher,
};
use crate::dynamics::solver::SolverParameters;
use crate::kernels::cuda::G2P2G_THREADS;
use crate::kernels::{GpuTimestepLength, NUM_CELL_PER_BLOCK};
use crate::math::{Real, Vector};
use cust::context::CurrentContext;
use cust::{
    error::CudaResult,
    event::{Event, EventFlags},
    memory::{CopyDestination, DeviceBox},
    module::ModuleJitOption,
    prelude::*,
};
use instant::{Duration, Instant};

/// Chosen from analyzing NSight Compute metrics on a 3070 and a 3080 Ti, should be optimal for non-ampere
/// GPUs also, but this has not been tested.
const MAX_REGISTERS: u32 = 130;

#[cfg(feature = "dim2")]
pub(crate) static PTX: &str = include_str!("../../resources/sparkl2d-kernels.ptx");
#[cfg(feature = "dim3")]
pub(crate) static PTX: &str = include_str!("../../resources/sparkl3d-kernels.ptx");

/// Timing data associated with a single timestep of the CUDA pipeline.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct CudaTimestepTimings {
    /// How long the timestep was.
    pub dt: f32,
    /// A list of every substep and its timing events.
    pub substeps: Vec<CudaSubstepTimings>,
    /// The total time for the entire step.
    pub total: Duration,
}

/// Timing data associated with a single substep of the CUDA pipeline.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct CudaSubstepTimings {
    /// How long the substep was.
    pub dt: f32,
    /// Time to allocate enough blocks for the sparse grid in the simulation. Happens
    /// at the start of the substep and is usually minor unless there is a lot of movement
    /// in the world.
    pub alloc_sparse_grid: Duration,
    /// Per-gpu timing events, this vector's length will be `1` unless multiple GPUs are being used.
    pub context_timings: Vec<CudaDeviceTimings>,
    /// The total time for this entire substep.
    pub total: Duration,
}

/// Timing data associated with a single device (context) used by the CUDA pipeline and its kernels.
///
/// Fields are in order of execution in the timestep.
#[derive(Debug, Clone, PartialEq)]
pub struct CudaDeviceTimings {
    /// The ordinal of this device.
    pub device: Device,
    /// Time to resize the sparse grid and sort particles.
    pub grid_resize_and_sort: Duration,
    /// Time to reset the sparse grid's blocks to zeroed values.
    pub reset_grid: Duration,
    /// Time to estimate the minimum required timestep for the substep.
    /// Note that all devices synchronize after this so the total time is
    /// around the max of all devices' timings.
    pub estimate_timestep: Duration,
    /// Builds a colored distance field (CDF) used to cheaply lookup the distance to colliders.
    pub updated_cdf: Duration,
    /// The main kernel in the pipeline, it spreads influence from particles to the grid,
    /// evaluates the constitutional model, and spreads influence from the grid back to the particles.
    /// Each device computes a chunk of the simulation when multigpu is being used.
    pub g2p2g: Duration,
    /// The main kernel in the pipeline, it spreads influence from particles to the grid,
    /// evaluates the constitutional model, and spreads influence from the grid back to the particles.
    /// Each device computes a chunk of the simulation when multigpu is being used.
    pub halo_g2p2g: Option<Duration>,
    /// Copying the halo (glue blocks) blocks to the staging buffer. Only available when multigpu is being used.
    pub copy_halo_to_staging: Option<Duration>,
    /// Copying other devices' halos to the device's halo from the staging buffer.
    pub copy_staging_to_remote_halo: Option<Duration>,
    /// Merging the halo blocks into the device's grid.
    pub merge_halo: Option<Duration>,
    /// Grid updates AFTER the halo merging/updating is finished, primarily includes collision resolution.
    pub grid_update: Duration,
}

struct EventTimer {
    start: Event,
    stop: Option<Event>,
    enabled: bool,
}

impl EventTimer {
    pub fn new(enabled: bool) -> CudaResult<EventTimer> {
        let flags = if enabled {
            EventFlags::empty()
        } else {
            EventFlags::DISABLE_TIMING
        };

        let start = Event::new(flags)?;

        Ok(EventTimer {
            start,
            stop: None,
            enabled,
        })
    }

    pub fn start(&self, stream: &Stream) -> CudaResult<()> {
        if self.enabled {
            self.start.record(stream)?;
        }
        Ok(())
    }

    pub fn stop(&mut self, stream: &Stream) -> CudaResult<()> {
        if self.enabled {
            let stop = Event::new(EventFlags::empty())?;
            stop.record(stream)?;
            self.stop = Some(stop);
        }
        Ok(())
    }

    #[track_caller]
    pub fn end(self) -> CudaResult<Duration> {
        if self.enabled {
            self.stop
                .expect("stop() was not called!")
                .elapsed(&self.start)
        } else {
            Ok(Duration::default())
        }
    }
}

/// CUDA-specific pipeline parameters
pub struct CudaMpmPipelineParameters<'t> {
    /// An optional handle to a timer to override timings for in the next timestep
    pub timings: Option<&'t mut CudaTimestepTimings>,
}

impl<'t> CudaMpmPipelineParameters<'t> {
    /// make sure there are enough allocated elements for the next substep
    fn reset_timings(&mut self, contexts: &[SingleGpuMpmContext], step: usize) -> CudaResult<()> {
        if let Some(timings) = &mut self.timings {
            timings.substeps.resize(step, Default::default());
            let substep = &mut timings.substeps[step - 1];
            substep.context_timings.clear();

            for context in contexts {
                context.make_current()?;
                let device = CurrentContext::get_device()?;
                substep.context_timings.push(CudaDeviceTimings {
                    device,
                    grid_resize_and_sort: Duration::default(),
                    reset_grid: Duration::default(),
                    estimate_timestep: Duration::default(),
                    updated_cdf: Duration::default(),
                    g2p2g: Duration::default(),
                    halo_g2p2g: None,
                    copy_halo_to_staging: None,
                    copy_staging_to_remote_halo: None,
                    merge_halo: None,
                    grid_update: Duration::default(),
                });
            }
        }
        Ok(())
    }
}

pub struct CudaMpmPipeline {
    step_id: u64,
}

pub struct SingleGpuMpmContext<'a> {
    pub context: Context,
    pub stream: &'a mut Stream,
    pub halo_stream: &'a mut Stream,
    pub module: &'a mut Module,
    pub colliders: &'a mut CudaColliderSet,
    pub particles: &'a mut CudaParticleSet,
    pub models: &'a mut CudaParticleModelSet,
    pub grid: &'a mut CudaSparseGrid,
    pub timestep_length: &'a mut DeviceBox<GpuTimestepLength>,
    pub num_active_blocks: u32,
    pub num_dispatch_blocks: u32,
    pub num_dispatch_halo_blocks: u32,
    pub sparse_grid_has_the_correct_size: bool,
    pub num_halo_blocks: u32,
    pub num_remote_halo_blocks: u32,
    pub remote_halo_index: u32,
}

struct ContextEvents {
    grid_resize_and_sort: EventTimer,
    reset_grid: EventTimer,
    estimate_timestep: EventTimer,
    update_cdf: EventTimer,
    g2p2g: EventTimer,
    halo_g2p2g: EventTimer,
    copy_halo_to_staging: Option<EventTimer>,
    copy_staging_to_remote_halo: Option<EventTimer>,
    merge_halo: Option<EventTimer>,
    grid_update: EventTimer,
}

impl ContextEvents {
    pub fn new(contexts: &mut [SingleGpuMpmContext], enabled: bool) -> CudaResult<Vec<Self>> {
        let mut events = Vec::with_capacity(contexts.len());
        for context in contexts {
            context.make_current()?;
            events.push(ContextEvents {
                grid_resize_and_sort: EventTimer::new(enabled)?,
                reset_grid: EventTimer::new(enabled)?,
                estimate_timestep: EventTimer::new(enabled)?,
                update_cdf: EventTimer::new(enabled)?,
                g2p2g: EventTimer::new(enabled)?,
                halo_g2p2g: EventTimer::new(enabled)?,
                copy_halo_to_staging: Some(EventTimer::new(enabled)?),
                copy_staging_to_remote_halo: Some(EventTimer::new(enabled)?),
                merge_halo: Some(EventTimer::new(enabled)?),
                grid_update: EventTimer::new(enabled)?,
            });
        }
        Ok(events)
    }
}

pub struct StepDetails {
    pub simulated_time: Real,
    pub num_substeps: u32,
}

impl<'a> SingleGpuMpmContext<'a> {
    pub fn make_current(&self) -> CudaResult<()> {
        cust::context::CurrentContext::set_current(&self.context)
    }
}

impl Default for CudaMpmPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaMpmPipeline {
    pub fn load_module() -> CudaResult<Module> {
        // NOTE(RDambrosio016): This might seem odd but we force the CUDA JIT compiler to spill any registers over
        // MAX_REGISTERS to local memory for optimization. The reasoning behind this is that the g2p2g kernel spends
        // a lot of its time on loads and stores to global memory, which is fairly slow. To counter this, NVIDIA GPU warp schedulers
        // will choose a different warp to schedule if their previous one(s) is now stuck on a slow stall op (such as global memory loads/stores).
        // Which means that many warps can effectively hide memory latency with this scheduling. However, this can only happen
        // if the GPU can schedule many warps, which it cannot do if the kernel uses many registers (because each SM only has 64k registers).
        // So we force the compiler to spill extra regs to local memory to gain more active warps (called occupancy).
        Module::from_ptx(PTX, &[ModuleJitOption::MaxRegisters(MAX_REGISTERS)])
    }

    pub fn new() -> Self {
        Self { step_id: 0 }
    }

    pub fn step(
        &mut self,
        cuda_params: CudaMpmPipelineParameters,
        params: &SolverParameters,
        gravity: &Vector<Real>,
        contexts: &mut [SingleGpuMpmContext],
    ) -> CudaResult<f32> {
        let mut launchers = vec![DefaultCudaParticleKernelsLauncher; contexts.len()];
        self.step_generic(cuda_params, params, gravity, contexts, &mut launchers)
    }

    pub fn step_generic(
        &mut self,
        mut cuda_params: CudaMpmPipelineParameters,
        params: &SolverParameters,
        gravity: &Vector<Real>,
        contexts: &mut [SingleGpuMpmContext],
        custom_launchers: &mut [impl CudaParticleKernelsLauncher],
    ) -> CudaResult<f32> {
        info!("Stepping the CUDA pipeline.");

        let timing_enabled = cuda_params.timings.is_some();
        let total_step_time = Instant::now();
        let multigpu = contexts.len() > 1;

        // NOTE: if we use 1024 threads, we run out of registers (so the launch fails).
        // See https://stackoverflow.com/a/40093367
        // This started happening after I added the CorotatedLinearElasticity stress computation.
        let threads = (512, 1, 1);

        // Reset the timestep length and ensure the most up-to-date models are available on
        // the GPU.
        for (i, context) in contexts.iter_mut().enumerate() {
            context.make_current()?;
            context
                .timestep_length
                .copy_from(&GpuTimestepLength::from_sec(params.dt))?;
            context.models.ensure_models_are_uploaded()?;
            unsafe {
                custom_launchers[i].timestep_started()?;
            }
        }

        let total_num_particles: usize = contexts.iter().map(|c| c.particles.len()).sum();
        if params.dt == 0.0 || total_num_particles == 0 {
            // Nothing to simulate.
            return Ok(params.dt);
        }

        let mut niter = 0;

        let min_dt = params.dt / (params.max_num_substeps as Real);
        let mut remaining_time = params.dt;
        #[cfg(feature = "dim2")]
        let blk_threads = (4u32, 4);
        #[cfg(feature = "dim3")]
        let blk_threads = (4u32, 4, 4);

        let t0 = instant::now();

        while remaining_time > 0.0 {
            let mut timestep_length = Real::MAX;
            self.step_id += 1;
            niter += 1;

            cuda_params.reset_timings(contexts, niter)?;

            for context in &mut *contexts {
                context.num_active_blocks = 0;
                context.num_dispatch_blocks = 0;
                context.sparse_grid_has_the_correct_size = false;
                context.num_halo_blocks = 0;
                context.num_remote_halo_blocks = 0;
                context.remote_halo_index = 0;
            }

            let total_substep_time;

            unsafe {
                let mut events = ContextEvents::new(contexts, timing_enabled)?;

                total_substep_time = Instant::now();

                let alloc_sparse_grid_timing = Instant::now();

                for (context, events) in contexts.iter_mut().zip(events.iter()) {
                    context.make_current()?;
                    let stream = &context.stream;
                    events.grid_resize_and_sort.start(stream)?;
                }

                CudaSparseGrid::launch_sort(contexts, G2P2G_THREADS as u32)?;

                for (context, events) in contexts.iter_mut().zip(events.iter_mut()) {
                    context.make_current()?;
                    let stream = &context.stream;
                    events.grid_resize_and_sort.stop(stream)?;
                }

                let alloc_sparse_grid = alloc_sparse_grid_timing.elapsed();

                for (i, context) in contexts.iter_mut().enumerate() {
                    context.make_current()?;
                    let module = &context.module;
                    let stream = &context.stream;
                    let num_active_blocks = context.num_active_blocks;

                    events[i].reset_grid.start(stream)?;

                    launch!(
                        module.reset_grid<<<num_active_blocks, blk_threads, 0, stream>>>(context.grid.next_device_elements())
                    )?;
                    launch!(
                        module.copy_grid_projection_data<<<num_active_blocks, blk_threads, 0, stream>>>(context.grid.curr_device_elements(), context.grid.next_device_elements())
                    )?;

                    events[i].reset_grid.stop(stream)?;
                    events[i].estimate_timestep.start(stream)?;

                    let blocks = (context.particles.len() as u32 / threads.0 + 1, 1, 1);
                    custom_launchers[i].launch_estimate_particle_timestep_length(
                        context,
                        blocks,
                        threads,
                        min_dt,
                        remaining_time.min(params.max_substep_dt),
                    )?;

                    events[i].estimate_timestep.stop(stream)?;
                }

                for context in &mut *contexts {
                    context.make_current()?;
                    context.stream.synchronize()?;
                    let candidate_dt = context.timestep_length.as_host_value()?.into_sec();
                    timestep_length = timestep_length.min(candidate_dt);
                }

                // Todo: add cdf update here
                for (i, context) in contexts.iter_mut().enumerate() {
                    context.make_current()?;
                    let module = &context.module;
                    let stream = &context.stream;

                    events[i].update_cdf.start(stream)?;

                    let particle_count = context.colliders.rigid_particles.len() as u32;

                    launch!(
                        module.update_cdf<<<particle_count, (1, 1, 1), 0, stream>>>(
                            context.grid.curr_device_elements(),
                            context.colliders.as_device(),

                        )
                    )?;

                    events[i].update_cdf.stop(stream)?;
                }

                // if remaining_time > min_dt {
                //     timestep_length = timestep_length.max(min_dt);
                // } else {
                //     timestep_length = remaining_time;
                // }

                // Halo G2P2G execution
                if multigpu {
                    use cust::memory::AsyncCopyDestination;

                    for i in 0..contexts.len() {
                        let context = &mut contexts[i];
                        context.make_current()?;
                        let num_dispatch_blocks = context.num_dispatch_halo_blocks;

                        events[i].halo_g2p2g.start(&context.halo_stream)?;

                        // NOTE: launching with 0 would cause an invalid argument error.
                        if num_dispatch_blocks > 0 {
                            custom_launchers[i].launch_g2p2g(
                                params,
                                context,
                                num_dispatch_blocks,
                                G2P2G_THREADS as u32,
                                timestep_length,
                                true,
                            )?;
                        }

                        events[i].halo_g2p2g.stop(&context.halo_stream)?;
                    }

                    for i in 0..contexts.len() {
                        let context = &mut contexts[i];
                        context.make_current()?;

                        let module = &context.module;
                        let halo_stream = &context.halo_stream;
                        // Copy the halo blocks to the staging buffer.
                        let nthreads = 32;
                        let ngroups = context.num_active_blocks / nthreads + 1;
                        let event = events[i].copy_halo_to_staging.as_mut().unwrap();

                        event.start(halo_stream)?;

                        launch!(
                            module.copy_halo_to_staging<<<ngroups, nthreads, 0, halo_stream>>>(
                                context.grid.next_device_elements(),
                                context.grid.halo_blocks_staging.as_device_ptr(),
                                context.grid.num_halo_blocks.as_device_ptr(),
                            )
                        )?;
                        event.stop(halo_stream)?;

                        // Copy the staging buffer to the other devices remote halo buffers.
                        let event = events[i].copy_staging_to_remote_halo.as_mut().unwrap();
                        event.start(contexts[i].halo_stream)?;

                        for j in 0..contexts.len() {
                            if i != j {
                                let halo_len_i = contexts[i].num_halo_blocks as usize;
                                let staging_i =
                                    contexts[i].grid.halo_blocks_staging.index(..halo_len_i);
                                let mut remote_halo_j = contexts[j].grid.remote_halo_blocks.index(
                                    contexts[j].remote_halo_index as usize
                                        ..contexts[j].remote_halo_index as usize + halo_len_i,
                                );
                                remote_halo_j
                                    .async_copy_from(&staging_i, contexts[i].halo_stream)
                                    .unwrap();
                                contexts[j].remote_halo_index += halo_len_i as u32;
                            }
                        }

                        event.stop(contexts[i].halo_stream)?;
                    }
                }

                // G2P2G execution.
                for (i, context) in contexts.iter_mut().enumerate() {
                    context.make_current()?;
                    let num_dispatch_blocks = context.num_dispatch_blocks;

                    events[i].g2p2g.start(&context.stream)?;

                    // NOTE: launching with 0 would cause an invalid argument error.
                    if num_dispatch_blocks > 0 {
                        custom_launchers[i].launch_g2p2g(
                            params,
                            context,
                            num_dispatch_blocks,
                            G2P2G_THREADS as u32,
                            timestep_length,
                            false,
                        )?;
                    }

                    events[i].g2p2g.stop(&context.stream)?;
                }

                if multigpu {
                    for context in &*contexts {
                        context.make_current()?;
                        // Wait for all the staging buffers to be populated
                        // before reading from them.
                        context.halo_stream.synchronize()?;
                    }

                    // NOTE: at this point, we synchronized all the halo_streams.
                    // This means that we cane safely read from the grid.remote_halo_blocks
                    // because they have been fully populated.

                    // Merge the halo blocks.
                    for i in 0..contexts.len() {
                        contexts[i].make_current()?;

                        let merge_halo = events[i].merge_halo.as_mut().unwrap();
                        let context = &mut contexts[i];
                        let module = &context.module;
                        let stream = &context.stream;

                        merge_halo.start(stream)?;

                        let num_remote_halo_blocks = context.num_remote_halo_blocks as u32;
                        launch!(
                            module.merge_halo_blocks<<<num_remote_halo_blocks, NUM_CELL_PER_BLOCK as u32, 0, stream>>>(
                                context.grid.next_device_elements(),
                                context.grid.remote_halo_blocks.as_device_ptr(),
                            )
                        )?;

                        merge_halo.stop(stream)?;
                    }
                }

                // Grid update.
                for (i, context) in contexts.iter_mut().enumerate() {
                    context.make_current()?;
                    let module = &context.module;
                    let stream = &context.stream;
                    let num_active_blocks = context.num_active_blocks;

                    events[i].grid_update.start(stream)?;

                    let (coll_ptr, coll_len) = context.colliders.device_elements();
                    launch!(
                        module.grid_update<<<num_active_blocks, blk_threads, 0, stream>>>(
                            timestep_length,
                            context.grid.next_device_elements(),
                            coll_ptr,
                            coll_len,
                            *gravity,
                        )
                    )?;

                    events[i].grid_update.stop(stream)?;

                    context.grid.swap_buffers();
                }

                if let Some(timings) = &mut cuda_params.timings {
                    let substep = &mut timings.substeps[niter - 1];

                    for context in &mut *contexts {
                        context.make_current()?;
                        context.stream.synchronize()?;
                    }

                    // Do this now so that it accounts for synchronizing
                    substep.total = total_substep_time.elapsed();
                    substep.dt = timestep_length;
                    substep.alloc_sparse_grid = alloc_sparse_grid;
                    info!("Substep time: {}", substep.total.as_secs_f32() * 1000.0);

                    for (i, (timings, device)) in substep
                        .context_timings
                        .iter_mut()
                        .zip(events.into_iter())
                        .enumerate()
                    {
                        contexts[i].make_current()?;

                        *timings = CudaDeviceTimings {
                            grid_resize_and_sort: device.grid_resize_and_sort.end()?,
                            reset_grid: device.reset_grid.end()?,
                            estimate_timestep: device.estimate_timestep.end()?,
                            updated_cdf: device.update_cdf.end()?,
                            g2p2g: device.g2p2g.end()?,
                            halo_g2p2g: if multigpu {
                                Some(device.halo_g2p2g.end()?)
                            } else {
                                None
                            },
                            copy_halo_to_staging: device
                                .copy_halo_to_staging
                                .filter(|_| multigpu)
                                .map(|x| x.end())
                                .transpose()?,
                            copy_staging_to_remote_halo: device
                                .copy_staging_to_remote_halo
                                .filter(|_| multigpu)
                                .map(|x| x.end())
                                .transpose()?,
                            merge_halo: device
                                .merge_halo
                                .filter(|_| multigpu)
                                .map(|x| x.end())
                                .transpose()?,
                            grid_update: device.grid_update.end()?,
                            ..*timings
                        };

                        info!("Substep device timings: {:?}", *timings);
                    }
                }
            }

            // info!(
            //     ">> Total substep ({}s {}Hz) computation time: {}, remaining step time: {}",
            //     timestep_length,
            //     1.0 / timestep_length,
            //     instant::now() - tt0,
            //     remaining_time - timestep_length
            // );

            info!(
                "Used effective timestep: {} = {}Hz",
                timestep_length,
                1.0 / timestep_length
            );

            remaining_time -= timestep_length;

            // NOTE: the remaining time can be very small instead of being equal to zero, because
            //       of the rounding done by the GpuTimestepLength. So we must break is it is
            //       too small due to this rounding (otherwise we will loop indefinitely with a
            //       substep length rounded to 0).
            if remaining_time <= 10.0 / GpuTimestepLength::FACTOR || params.stop_after_one_substep {
                break;
            }
        }

        // We are done simulating this step, synchronize.
        for context in &*contexts {
            context.make_current()?;
            context.stream.synchronize()?;
        }

        if let Some(timings) = &mut cuda_params.timings {
            timings.total = total_step_time.elapsed();
            timings.dt = params.dt;
        }

        info!(
            ">>>> Total step computation time ({} iterations): {}",
            niter,
            instant::now() - t0
        );

        Ok(params.dt - remaining_time)
    }
}
