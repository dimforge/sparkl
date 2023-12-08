use cust::device::Device;
use instant::Duration;

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
