use crate::math::{Real, Vector};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContext};
use na::{Point3, Vector3};
use rapier::geometry::ColliderSet;
use rapier_testbed::{harness::Harness, GraphicsManager, PhysicsState, TestbedPlugin};

use super::point_cloud_render::ParticleInstanceData;
use crate::core::dynamics::ParticleData;
use crate::dynamics::{GridNode, GridNodeCgPhase, ParticleModelSet, ParticleSet};
use crate::geometry::SpGrid;
use crate::pipelines::MpmPipeline;
use crate::prelude::{MpmHooks, RigidWorld, SolverParameters};
use instant::Duration;
use rapier::data::Coarena;

#[cfg(feature = "dim3")]
use super::point_cloud_render::ParticleInstanceMaterialData;
#[cfg(feature = "cuda")]
use {
    crate::{
        cuda::{
            CudaColliderSet, CudaConstitutiveModel, CudaFailureModel, CudaMpmPipeline,
            CudaMpmPipelineParameters, CudaParticleModelSet, CudaParticleSet, CudaPlasticModel,
            CudaSparseGrid, CudaTimestepTimings, SharedCudaVec, SingleGpuMpmContext,
        },
        kernels::GpuTimestepLength,
    },
    cust::{
        error::CudaResult,
        memory::{CopyDestination, LockedBuffer},
    },
};

/// How the fluids should be rendered by the testbed.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ParticlesRenderingMode {
    /// Use a plain color.
    StaticColor,
    /// Use a red taint the closer to `max` the velocity is.
    VelocityColor {
        /// Fluids with a velocity smaller than this will not have any red taint.
        min: f32,
        /// Fluids with a velocity greater than this will be completely red.
        max: f32,
    },
    DensityRatio {
        max: f32,
    },
    Position {
        mins: Vector<f32>,
        maxs: Vector<f32>,
    },
    Blocks {
        block_len: usize,
    },
}

/// A user-defined callback executed at each frame.
pub type UserCallback = Box<dyn FnMut(&mut Harness, &mut ParticleSet, f64)>;

#[cfg(feature = "cuda")]
pub type UserCudaCallback =
    Box<dyn FnMut(&mut Harness, &mut ParticleSet, &mut CudaParticleSet, f64)>;

struct ParticleGfx {
    entity: Entity,
}

#[cfg(feature = "cuda")]
struct CudaData {
    stream: cust::prelude::Stream,
    halo_stream: cust::prelude::Stream,
    module: cust::prelude::Module,
    colliders: Option<CudaColliderSet>,
    particles: CudaParticleSet,
    models: CudaParticleModelSet,
    grid: CudaSparseGrid,
    timestep_length: cust::memory::DeviceBox<GpuTimestepLength>,
    context: cust::prelude::Context,
    particle_attribute_data: SharedCudaVec<ParticleData>,
}

#[cfg(feature = "cuda")]
impl CudaData {
    pub fn make_current(&self) -> CudaResult<()> {
        cust::context::CurrentContext::set_current(&self.context)
    }
}

/// A plugin for rendering fluids with the Rapier testbed.
pub struct MpmTestbedPlugin {
    pub render_boundary_particles: bool,
    pub particles_rendering_mode: ParticlesRenderingMode,
    step_id: usize,
    pub callbacks: Vec<UserCallback>,
    step_time: f64,
    simulated_time: f64,
    real_time: f64,
    mpm_pipeline: MpmPipeline,
    #[cfg(feature = "cuda")]
    pub cuda_callbacks: Vec<UserCudaCallback>,
    #[cfg(feature = "cuda")]
    cuda_pipeline: CudaMpmPipeline,
    #[cfg(feature = "cuda")]
    cuda_data: Vec<CudaData>, // One per device.
    // We use a pinned buffer instead of a vec because it makes memory transfers faster. With
    // pageable memory, the CUDA driver must allocate a staging buffer before dispatching a DMA
    // transfer because the OS may page out the memory out of RAM at any time. With pinned memory
    // this will not happen so the driver can initiate the DMA transfer instantly, making it 2x+ faster
    // than normal copies. However abusing this can cause the whole system to slow down because the OS
    // cannot swap out memory to disk.
    #[cfg(feature = "cuda")]
    cuda_pos_writeback: LockedBuffer<crate::core::dynamics::ParticlePosition>,
    #[cfg(feature = "cuda")]
    cuda_vel_writeback: LockedBuffer<crate::core::dynamics::ParticleVelocity>,
    #[cfg(feature = "cuda")]
    cuda_phase_writeback: LockedBuffer<crate::core::dynamics::ParticlePhase>,
    // wgpu_pipeline: WgpuMpmPipeline,
    // f2sn: HashMap<FluidHandle, Vec<EntityWithGraphics>>,
    // boundary2sn: HashMap<BoundaryHandle, Vec<EntityWithGraphics>>,
    // f2color: HashMap<FluidHandle, Point3<f32>>,
    #[cfg(feature = "dim2")]
    particle_gfx: Option<Vec<ParticleGfx>>,
    #[cfg(feature = "dim3")]
    particle_gfx: Option<ParticleGfx>,
    grid_gfx: Option<ParticleGfx>,
    models: ParticleModelSet,
    pub particles: ParticleSet,
    // pub wgpu_particles: WParticleDataSet,
    boundaries: Option<ColliderSet>,
    model_colors: Coarena<Point3<f32>>,
    pub hooks: Box<dyn MpmHooks>,
    sp_grid: SpGrid<GridNode>,
    sp_grid_phase: SpGrid<GridNodeCgPhase>,
    pub solver_params: SolverParameters,
    blocks_color: Vec<[f32; 3]>,
    run_on_gpu: bool,
    #[cfg(feature = "cuda")]
    pub last_timing: Option<CudaTimestepTimings>,
}

impl MpmTestbedPlugin {
    pub fn new(models: ParticleModelSet, particles: ParticleSet, cell_width: Real) -> Self {
        Self::with_boundaries(models, particles, cell_width, None, None)
    }

    pub fn init(app: &mut App) {
        super::point_cloud_render::init_renderer(app)
    }

    pub fn with_max_num_devices(
        models: ParticleModelSet,
        particles: ParticleSet,
        cell_width: Real,
        max_num_devices: usize,
    ) -> Self {
        Self::with_boundaries(models, particles, cell_width, None, Some(max_num_devices))
    }

    /// Initializes the plugin.
    pub fn with_boundaries(
        models: ParticleModelSet,
        particles: ParticleSet,
        cell_width: Real,
        boundaries: Option<ColliderSet>,
        max_num_devices: Option<usize>,
    ) -> Self {
        let sp_grid = SpGrid::new(cell_width).unwrap();
        let sp_grid_phase = SpGrid::new(cell_width).unwrap();

        #[cfg(feature = "cuda")]
        let cuda_data: Vec<_> = {
            use cust::memory::DeviceBox;
            use cust::prelude::*;
            let err = "Failed to initialize Cuda";
            cust::init(cust::CudaFlags::empty()).expect(err);
            let mut num_devices =
                max_num_devices.unwrap_or_else(|| Device::devices().expect(err).count());

            if true
                || particles.iter().all(|p| p.model.into_raw_parts().0 == 0)
                || particles.iter().all(|p| p.model.into_raw_parts().0 == 1)
            {
                num_devices = 1;
            }

            let num_devices = 1;

            Device::devices()
                .expect(err)
                .enumerate()
                .take(num_devices)
                .filter_map(|(i, device)| {
                    let device = device.ok()?;
                    let context = Context::new(device).ok()?;
                    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).ok()?;
                    let halo_stream = Stream::new(StreamFlags::NON_BLOCKING, None).ok()?;
                    let mut cpu_particles = particles.clone();
                    cpu_particles.retain(|p| {
                        p.model.into_raw_parts().0 as usize == i
                            || p.model.into_raw_parts().0 as usize >= num_devices
                    });
                    println!("This device has {} particles", cpu_particles.len());
                    let comps = crate::cuda::extract_particles_components(&cpu_particles.particles);
                    let particles = CudaParticleSet::from_particles(
                        &comps.status,
                        &comps.position,
                        &comps.velocity,
                        &comps.volume,
                        &comps.phase,
                    )
                    .ok()?;
                    let particle_attribute_data = SharedCudaVec::from_slice(&comps.data).ok()?;

                    let models = CudaParticleModelSet::from_model_set(
                        models.iter().map(|(_, m)| m),
                        particle_attribute_data.clone(),
                    )
                    .ok()?;
                    let grid = CudaSparseGrid::new(cell_width).ok()?;
                    let colliders = boundaries
                        .as_ref()
                        .and_then(|b| CudaColliderSet::from_collider_set(b, vec![]).ok());
                    let timestep_length = DeviceBox::new(&GpuTimestepLength::default()).ok()?;
                    let module = CudaMpmPipeline::load_module().unwrap();

                    Some(CudaData {
                        context,
                        stream,
                        halo_stream,
                        module,
                        particles,
                        models,
                        grid,
                        colliders,
                        timestep_length,
                        particle_attribute_data,
                    })
                })
                .collect()
        };

        #[cfg(feature = "cuda")]
        if cuda_data.is_empty() {
            panic!("Could not initialize any suitable CUDA-compatible device.");
        }

        #[cfg(feature = "cuda")]
        println!("Found {} CUDA devices.", cuda_data.len());

        #[cfg(feature = "cuda")]
        let cuda_pipeline = CudaMpmPipeline::new();

        Self {
            step_id: 0,
            render_boundary_particles: false,
            particles_rendering_mode: ParticlesRenderingMode::VelocityColor {
                min: 0.0,
                max: 10.0,
            },
            step_time: 0.0,
            simulated_time: 0.0,
            real_time: 0.0,
            callbacks: Vec::new(),
            mpm_pipeline: MpmPipeline::new(),
            #[cfg(feature = "cuda")]
            cuda_pipeline,
            #[cfg(feature = "cuda")]
            cuda_data,
            #[cfg(feature = "cuda")]
            cuda_callbacks: vec![],
            #[cfg(feature = "cuda")]
            cuda_pos_writeback: cust::memory::LockedBuffer::new(
                &crate::core::dynamics::ParticlePosition::default(),
                0,
            )
            .expect("Failed to allocate locked staging buffer"),
            #[cfg(feature = "cuda")]
            cuda_vel_writeback: cust::memory::LockedBuffer::new(
                &crate::core::dynamics::ParticleVelocity::default(),
                0,
            )
            .expect("Failed to allocate locked staging buffer"),
            #[cfg(feature = "cuda")]
            cuda_phase_writeback: cust::memory::LockedBuffer::new(
                &crate::core::dynamics::ParticlePhase::default(),
                0,
            )
            .expect("Failed to allocate locked staging buffer"),
            // f2sn: HashMap::new(),
            // boundary2sn: HashMap::new(),
            // f2color: HashMap::new(),
            model_colors: Coarena::new(),
            particle_gfx: None,
            grid_gfx: None,
            models,
            particles,
            boundaries,
            hooks: Box::new(()),
            sp_grid,
            sp_grid_phase,
            solver_params: SolverParameters::default(),
            blocks_color: vec![],
            run_on_gpu: true,
            #[cfg(feature = "cuda")]
            last_timing: None,
        }
    }

    /// Adds a callback to be executed at each frame.
    pub fn add_callback(&mut self, f: impl FnMut(&mut Harness, &mut ParticleSet, f64) + 'static) {
        self.callbacks.push(Box::new(f))
    }

    /// Adds a callback to be executed at each frame.
    #[cfg(feature = "cuda")]
    pub fn add_cuda_callback(
        &mut self,
        f: impl FnMut(&mut Harness, &mut ParticleSet, &mut CudaParticleSet, f64) + 'static,
    ) {
        self.cuda_callbacks.push(Box::new(f))
    }

    fn lerp_velocity(
        velocity: f32,
        start: Vector3<f32>,
        end: Vector3<f32>,
        min: f32,
        max: f32,
    ) -> Vector3<f32> {
        let t = (velocity - min) / (max - min);
        start.lerp(&end, na::clamp(t, 0.0, 1.0))
    }
}

impl TestbedPlugin for MpmTestbedPlugin {
    fn init_plugin(&mut self) {
        // let pipeline = PipelineDescriptor::default_config(ShaderStages {
        //     vertex: shaders.add(Shader::from_glsl(
        //         ShaderStage::Vertex,
        //         super::point_cloud_render::PARTICLES_VERTEX_SHADER,
        //     )),
        //     fragment: Some(shaders.add(Shader::from_glsl(
        //         ShaderStage::Fragment,
        //         super::point_cloud_render::PARTICLES_FRAGMENT_SHADER,
        //     ))),
        // });
        // pipelines.set_untracked(
        //     super::point_cloud_render::PARTICLES_PIPELINE_HANDLE,
        //     pipeline,
        // );
    }

    fn init_graphics(
        &mut self,
        graphics: &mut GraphicsManager,
        _commands: &mut Commands,
        _meshes: &mut Assets<Mesh>,
        #[cfg(feature = "dim2")] _materials: &mut Assets<ColorMaterial>,
        #[cfg(feature = "dim3")] _materials: &mut Assets<StandardMaterial>,
        _components: &mut Query<(&mut Transform,)>,
        _harness: &mut Harness,
    ) {
        self.simulated_time = 0.0;
        self.real_time = 0.0;
        self.step_id = 0;
        graphics.next_color(); // Skip the first color because it’s boring.

        /*
         * Initialize the colors.
         */
        for (index, _) in self.models.iter() {
            if self.model_colors.get(index).is_none() {
                self.model_colors.insert(index, graphics.next_color());
            }
        }
        #[cfg(feature = "dim2")]
        {
            self.particle_gfx = Some(vec![]);
        }

        #[cfg(feature = "dim3")]
        {
            let entity = _commands
                .spawn((
                    _meshes.add(Mesh::from(shape::Icosphere {
                        radius: 1.0,
                        subdivisions: 5,
                    })),
                    Transform::from_xyz(0.0, 0.0, 0.0),
                    GlobalTransform::default(),
                    ParticleInstanceMaterialData(vec![]),
                    Visibility::default(),
                    ComputedVisibility::default(),
                ))
                .id();

            self.particle_gfx = Some(ParticleGfx { entity });
        }
    }

    fn clear_graphics(&mut self, _graphics: &mut GraphicsManager, commands: &mut Commands) {
        self.model_colors = Coarena::new();
        self.blocks_color.clear();

        #[cfg(feature = "dim2")]
        if let Some(particle_gfx) = self.particle_gfx.take() {
            for gfx in particle_gfx {
                commands.entity(gfx.entity).despawn();
            }
        }
        #[cfg(feature = "dim3")]
        if let Some(particle_gfx) = self.particle_gfx.take() {
            commands.entity(particle_gfx.entity).despawn();
        }
        if let Some(grid_gfx) = self.grid_gfx.take() {
            commands.entity(grid_gfx.entity).despawn();
        }

        // TODO: move this somewhere else; or perhaps rename this method
        //       to `clear_scene` instead of `clear_graphics`?
        #[cfg(feature = "cuda")]
        {
            for cuda_data in &mut self.cuda_data {
                cuda_data.colliders = None;
            }
        }
    }

    fn run_callbacks(&mut self, harness: &mut Harness) {
        for f in &mut self.callbacks {
            f(harness, &mut self.particles, self.simulated_time)
        }

        #[cfg(feature = "cuda")]
        for f in &mut self.cuda_callbacks {
            f(
                harness,
                &mut self.particles,
                &mut self.cuda_data[0].particles,
                self.simulated_time,
            )
        }
    }

    fn step(&mut self, physics: &mut PhysicsState) {
        // let step_time = instant::now();
        let rigid_world = RigidWorld {
            colliders: self.boundaries.as_ref().unwrap_or(&physics.colliders),
        };

        #[cfg(feature = "cuda")]
        for cuda_data in &mut self.cuda_data {
            if cuda_data.colliders.is_none() {
                log::info!("Initializing {} cuda colliders.", physics.colliders.len());
                cuda_data.make_current().unwrap();
                cuda_data.colliders = Some(
                    CudaColliderSet::from_collider_set(&physics.colliders, vec![])
                        .expect("Failed to initialize the CUDA colliders."),
                );
            }
        }

        let t0 = instant::now();
        let mut simulated_time = 0.0;

        if !self.run_on_gpu {
            self.solver_params.dt = physics.integration_parameters.dt;
            simulated_time = self.mpm_pipeline.step(
                &self.solver_params,
                &physics.gravity,
                &rigid_world,
                &mut self.sp_grid,
                &mut self.sp_grid_phase,
                &mut self.particles,
                &self.models,
                &mut *self.hooks,
            );
        } else {
            #[cfg(feature = "cuda")]
            {
                self.solver_params.dt = physics.integration_parameters.dt;

                let mut contexts: Vec<_> = self
                    .cuda_data
                    .iter_mut()
                    .map(|data| SingleGpuMpmContext {
                        context: data.context.clone(),
                        stream: &mut data.stream,
                        halo_stream: &mut data.halo_stream,
                        module: &mut data.module,
                        colliders: data.colliders.as_mut().unwrap(),
                        particles: &mut data.particles,
                        models: &mut data.models,
                        grid: &mut data.grid,
                        timestep_length: &mut data.timestep_length,
                        num_active_blocks: 0,
                        num_dispatch_blocks: 0,
                        num_dispatch_halo_blocks: 0,
                        sparse_grid_has_the_correct_size: false,
                        num_halo_blocks: 0,
                        num_remote_halo_blocks: 0,
                        remote_halo_index: 0,
                    })
                    .collect();

                let last_timing = self.last_timing.get_or_insert(Default::default());
                let cuda_params = CudaMpmPipelineParameters {
                    timings: Some(last_timing),
                };

                simulated_time = self
                    .cuda_pipeline
                    .step(
                        cuda_params,
                        &self.solver_params,
                        &physics.gravity,
                        &mut contexts,
                    )
                    .expect("CUDA stepping failed");

                /*
                 *
                 * Read back the particle data.
                 *
                 */
                let total_num_particles: usize = contexts.iter().map(|c| c.particles.len()).sum();
                if self.cuda_pos_writeback.len() < total_num_particles {
                    // SAFETY: this buffer is not read until it is written to.
                    self.cuda_pos_writeback =
                        unsafe { LockedBuffer::uninitialized(total_num_particles).unwrap() };
                    self.cuda_vel_writeback =
                        unsafe { LockedBuffer::uninitialized(total_num_particles).unwrap() };
                    self.cuda_phase_writeback =
                        unsafe { LockedBuffer::uninitialized(total_num_particles).unwrap() };
                }
                let transfer_time = std::time::Instant::now();

                let mut curr_shift = 0;
                for context in &mut contexts {
                    context.make_current().unwrap();
                    let num_particles = context.particles.len();
                    context
                        .particles
                        .particle_pos
                        .buffer()
                        .index(..num_particles)
                        .copy_to(
                            &mut self.cuda_pos_writeback[curr_shift..curr_shift + num_particles],
                        )
                        .expect("Could not retrieve particle data from the GPU.");
                    context
                        .particles
                        .particle_vel
                        .buffer()
                        .index(..num_particles)
                        .copy_to(
                            &mut self.cuda_vel_writeback[curr_shift..curr_shift + num_particles],
                        )
                        .expect("Could not retrieve particle data from the GPU.");
                    context
                        .particles
                        .particle_phase
                        .buffer()
                        .index(..num_particles)
                        .copy_to(
                            &mut self.cuda_phase_writeback[curr_shift..curr_shift + num_particles],
                        )
                        .expect("Could not retrieve particle data from the GPU.");
                    curr_shift += num_particles;
                }

                for (out_p, pos, vel, phase) in itertools::multizip((
                    self.particles.particles.iter_mut(),
                    self.cuda_pos_writeback.iter(),
                    self.cuda_vel_writeback.iter(),
                    self.cuda_phase_writeback.iter(),
                )) {
                    out_p.position = pos.point;
                    out_p.velocity = vel.vector;
                    out_p.phase = phase.phase;
                }

                let transfer_time = transfer_time.elapsed();
                println!("Transfer time: {}ms", transfer_time.as_millis())
            }
        }

        self.real_time += instant::now() - t0;
        self.simulated_time += simulated_time as f64;
        log::info!(
            "$$$$$$$$$$$ Simulated time: {}ms (real run time: {}ms)",
            self.simulated_time * 1000.0,
            self.real_time
        );
    }

    fn draw(
        &mut self,
        graphics: &mut GraphicsManager,
        commands: &mut Commands,
        _meshes: &mut Assets<Mesh>,
        #[cfg(feature = "dim2")] _materials: &mut Assets<ColorMaterial>,
        #[cfg(feature = "dim3")] _materials: &mut Assets<StandardMaterial>,
        _positions: &mut Query<(&mut Transform,)>,
        _harness: &mut Harness,
    ) {
        self.step_id += 1;

        // Update particles
        if let Some(gfx) = &mut self.particle_gfx {
            let mut instance_data = vec![];

            for (i, particle) in self.particles.particles.iter().enumerate() {
                #[cfg(feature = "dim2")]
                let pos_z = 0.0;
                #[cfg(feature = "dim3")]
                let pos_z = particle.position.z;
                let pos = [
                    particle.position.x as f32,
                    particle.position.y as f32,
                    pos_z,
                ];

                let color = match self.particles_rendering_mode {
                    ParticlesRenderingMode::Blocks { block_len } => {
                        while self.blocks_color.len() <= self.particles.len() / block_len {
                            self.blocks_color.push((graphics.next_color() / 2.0).into());
                        }

                        self.blocks_color[i / block_len]
                    }
                    ParticlesRenderingMode::Position { mins, maxs } => {
                        let color = (particle.position - mins)
                            .coords
                            .component_div(&(maxs - mins))
                            .map(|e| (e * 30.0).floor() / 60.0);
                        #[cfg(feature = "dim2")]
                        let color = color.push(0.0);
                        color.into()
                    }
                    ParticlesRenderingMode::DensityRatio { max } => {
                        let base_color = *self.model_colors.get(particle.model).unwrap();
                        let red = Vector3::x();
                        let blue = Vector3::z();
                        let ratio = particle.density_def_grad() / particle.density0();

                        if ratio > 1.0 {
                            let coeff = (ratio - 1.0) / max;
                            base_color.coords.lerp(&red, coeff).into()
                        } else {
                            blue.lerp(&base_color.coords, ratio).into()
                        }
                    }
                    ParticlesRenderingMode::VelocityColor { min, max } => {
                        let base_color = *self.model_colors.get(particle.model).unwrap();
                        let color = if particle.failed {
                            [1.0, 1.0, 0.0]
                        } else if false && particle.phase == 0.0 {
                            [0.0, 0.0, 1.0]
                        } else if false && particle.phase != 1.0 {
                            let red = Vector3::x();
                            let blue = Vector3::z();
                            let velocity_color = Self::lerp_velocity(
                                particle.velocity.norm(),
                                base_color.coords,
                                red,
                                min,
                                max,
                            );
                            let color = velocity_color.lerp(&blue, 1.0 - particle.phase);
                            [color.x, color.y, color.z]
                        } else {
                            let red = Vector3::x();
                            let lerp = Self::lerp_velocity(
                                particle.velocity.norm(),
                                base_color.coords,
                                red,
                                min,
                                max,
                            );
                            [lerp.x, lerp.y, lerp.z]
                        };
                        color
                    }
                    ParticlesRenderingMode::StaticColor => self
                        .model_colors
                        .get(particle.model)
                        .copied()
                        .unwrap()
                        .into(),
                };

                instance_data.push(ParticleInstanceData {
                    position: pos.into(),
                    #[cfg(feature = "dim2")]
                    scale: (particle.volume0 / std::f32::consts::PI).sqrt() / 2.0,
                    #[cfg(feature = "dim3")]
                    scale: (particle.volume0 * 3.0 / (4.0 * std::f32::consts::PI)).cbrt() / 2.0,
                    color: [color[0], color[1], color[2], 1.0],
                });
            }

            #[cfg(feature = "dim2")]
            {
                for (data, gfx) in instance_data.iter().zip(gfx.iter()) {
                    commands
                        .entity(gfx.entity)
                        .insert(Sprite {
                            color: Color::rgb(data.color[0], data.color[1], data.color[2]),
                            ..Default::default()
                        })
                        .insert(
                            Transform::from_xyz(data.position.x, data.position.y, data.position.z)
                                .with_scale(Vec3::splat(data.scale * 2.0)),
                        );
                }

                if instance_data.len() > gfx.len() {
                    for data in &instance_data[gfx.len()..] {
                        // We are missing some sprites.
                        let entity = commands
                            .spawn_bundle(SpriteBundle {
                                sprite: Sprite {
                                    color: Color::rgb(data.color[0], data.color[1], data.color[2]),
                                    ..Default::default()
                                },
                                transform: Transform::from_xyz(
                                    data.position.x,
                                    data.position.y,
                                    data.position.z,
                                )
                                .with_scale(Vec3::splat(data.scale * 2.0)),
                                ..Default::default()
                            })
                            .id();
                        gfx.push(ParticleGfx { entity });
                    }
                } else {
                    // Remove spurious entities.
                    for gfx in &gfx[instance_data.len()..] {
                        commands.entity(gfx.entity).despawn();
                    }

                    gfx.truncate(instance_data.len());
                }
            }
            #[cfg(feature = "dim3")]
            {
                commands
                    .entity(gfx.entity)
                    .insert(ParticleInstanceMaterialData(instance_data));
            }
        }
    }

    fn update_ui(
        &mut self,
        ui_context: &EguiContext,
        _harness: &mut Harness,
        _graphics: &mut GraphicsManager,
        _commands: &mut Commands,
        _meshes: &mut Assets<Mesh>,
        #[cfg(feature = "dim2")] _materials: &mut Assets<ColorMaterial>,
        #[cfg(feature = "dim3")] _materials: &mut Assets<StandardMaterial>,
        _components: &mut Query<(&mut Transform,)>,
    ) {
        #[cfg(feature = "cuda")]
        {
            egui::Window::new("MPM Params").show(ui_context.ctx(), |ui| {
                ui.checkbox(&mut self.run_on_gpu, "Run on GPU");
                if let Some(timing) = &self.last_timing {
                    ui.collapsing("Pipeline Timings", |ui| {
                        ui.label(format!("dt: {:.2}ms", timing.dt * 1000.0));
                        ui.label(format!(
                            "Total time: {:.2}ms",
                            timing.total.as_secs_f32() * 1000.0
                        ));
                        ui.label(format!("Substeps: {}", timing.substeps.len()));

                        ui.separator();

                        let mut longest_substep = (0, Duration::default());
                        // pick the slowest substep to display
                        for (idx, substep) in timing.substeps.iter().enumerate() {
                            if substep.total > longest_substep.1 {
                                longest_substep.0 = idx;
                            }
                        }

                        ui.heading(format!("Slowest Substep ({})", longest_substep.0));

                        let substep = &timing.substeps[longest_substep.0];
                        let total_millis = substep.total.as_secs_f32() * 1000.0;
                        ui.label(format!("Substep time: {:.2}ms", total_millis));
                        let device = &substep.context_timings[0];

                        let format_vals = |raw: Duration| -> (f32, u32, egui::Color32) {
                            let millis = raw.as_secs_f32() * 1000.0;
                            let fract_norm = millis / total_millis;
                            let percent = (fract_norm * 100.0) as u32;
                            let color = gradient(fract_norm);
                            (millis, percent, color)
                        };

                        let grid_resize_and_sort = format_vals(device.grid_resize_and_sort);
                        ui.colored_label(
                            grid_resize_and_sort.2,
                            format!(
                                "Grid Resize and Sort: {:.2}ms ({}%)",
                                grid_resize_and_sort.0, grid_resize_and_sort.1
                            ),
                        );

                        let reset_grid = format_vals(device.reset_grid);
                        ui.colored_label(
                            reset_grid.2,
                            format!("Reset Grid: {:.2}ms ({}%)", reset_grid.0, reset_grid.1),
                        );

                        let estimate_timestep = format_vals(device.estimate_timestep);
                        ui.colored_label(
                            estimate_timestep.2,
                            format!(
                                "Estimate Timestep: {:.2}ms ({}%)",
                                estimate_timestep.0, estimate_timestep.1
                            ),
                        );

                        let g2p2g = format_vals(device.g2p2g);
                        ui.colored_label(
                            g2p2g.2,
                            format!("g2p2g: {:.2}ms ({}%)", g2p2g.0, g2p2g.1),
                        );

                        if let Some(halo_g2p2g) = device.halo_g2p2g.map(format_vals) {
                            ui.colored_label(
                                halo_g2p2g.2,
                                format!("Halo g2p2g: {:.2}ms ({}%)", halo_g2p2g.0, halo_g2p2g.1),
                            );
                        }

                        let grid_update = format_vals(device.grid_update);
                        ui.colored_label(
                            grid_update.2,
                            format!("Grid Update: {:.2}ms ({}%)", grid_update.0, grid_update.1),
                        );
                    });
                }
            });
        }
    }

    fn profiling_string(&self) -> String {
        format!("MPM: {:.2}ms", self.step_time)
    }
}

fn gradient(frac: f32) -> egui::Color32 {
    let frac = frac.clamp(0.0, 1.0);
    let c1 = [0.0, 1.0, 0.0, 1.0];
    let c2 = [1.0, 0.0, 0.0, 1.0];
    let pow = 1.0 / 2.2;
    let new_color = [
        (c1[0] + frac * (c2[0] - c1[0])).powf(pow),
        (c1[1] + frac * (c2[1] - c1[1])).powf(pow),
        (c1[2] + frac * (c2[2] - c1[2])).powf(pow),
        1.0,
    ];
    egui::Color32::from_rgb(
        (new_color[0] * 255.0) as u8,
        (new_color[1] * 255.0) as u8,
        (new_color[2] * 255.0) as u8,
    )
}
