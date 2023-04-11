use crate::dynamics::solver::{DamageModel, MpmHooks, SolverParameters};
use crate::dynamics::{GridNode, GridNodeCgPhase, ParticleModelSet, ParticleSet};
use crate::geometry::SpGrid;
use crate::math::{Real, Vector};
use rapier::geometry::ColliderSet;

pub struct RigidWorld<'a> {
    pub colliders: &'a ColliderSet,
}

/// An MPM solver based on MPM-MLS with APIC transfer.
pub struct MlsSolver;

impl MlsSolver {
    pub fn step(
        params: &SolverParameters,
        gravity: &Vector<Real>,
        rigid_world: &RigidWorld,
        grid: &mut SpGrid<GridNode>,
        grid_phase: &mut SpGrid<GridNodeCgPhase>,
        particles: &mut ParticleSet,
        models: &ParticleModelSet,
        hooks: &mut dyn MpmHooks,
    ) -> Real {
        if particles.particles.is_empty() || params.dt == 0.0 {
            // Nothing to simulate.
            return params.dt;
        }

        let t0 = instant::now();
        let mut niter = 0;

        let min_dt = params.dt / (params.max_num_substeps as Real);
        let mut remaining_time = params.dt;

        while remaining_time > 0.0 {
            niter += 1;

            let tt0 = instant::now();

            let t0 = instant::now();
            particles.sort(params.damage_model == DamageModel::Eigenerosion, grid);
            // particles.sort(grid);
            info!("Sort time: {}ms", instant::now() - t0);

            if params.force_fluids_volume_recomputation {
                Self::recompute_fluids_volumes(grid, particles, models);
            }

            let t0 = instant::now();
            let mut dt = Self::adaptive_timestep_length(
                remaining_time.min(params.max_substep_dt),
                grid,
                particles,
                models,
            );
            if dt < min_dt && remaining_time > min_dt {
                dt = min_dt;
            }
            info!("Adaptive timestep: {}ms", instant::now() - t0);

            let t0 = instant::now();
            // Self::update_particles_pos_energy(particles, models);
            // Self::compute_particle_normals(rigid_world, particles);
            info!("Collision detection: {}ms", instant::now() - t0);

            let t0 = instant::now();
            // TODO: find a way to make this part of the constitutive model?
            match params.damage_model {
                DamageModel::None => {}
                DamageModel::CdMpm => {
                    crate::dynamics::phase_field::update_phase_field(
                        dt, grid, grid_phase, particles,
                    );
                }
                DamageModel::Eigenerosion => {
                    Self::evolve_eigenerosion(grid, particles);
                }
                DamageModel::ModifiedEigenerosion => {
                    /* Nothing to do, this will be handled in the P2G/G2P. */
                }
            }

            // Self::local_failure_update(particles, models);
            info!("Cracks propagation: {}ms", instant::now() - t0);

            let t0 = instant::now();
            // Self::update_particles_stress(dt, particles, models, grid.cell_width());
            info!("Stress update: {}ms", instant::now() - t0);

            let t0 = instant::now();
            Self::particle_to_grid_scatter(dt, gravity, grid, particles, models);
            Self::grid_update(
                dt,
                rigid_world,
                grid,
                particles,
                params.boundary_condition,
                params.simulation_dofs,
            );
            hooks.post_grid_update_hook(grid);
            info!("Grid update: {}ms", instant::now() - t0);

            let t0 = instant::now();
            Self::grid_to_particle(
                params.damage_model,
                params.enable_boundary_particle_projection,
                dt,
                rigid_world,
                grid,
                particles,
                models,
            );
            info!("Particle update: {}ms", instant::now() - t0);

            info!(
                ">> Total substep ({}s {}Hz) computation time: {}",
                dt,
                1.0 / dt,
                instant::now() - tt0
            );

            // info!("Used effective timestep: {} = {}Hz", dt, 1.0 / dt);

            remaining_time -= dt;

            if params.stop_after_one_substep {
                return dt;
            }
        }

        info!(
            ">>>> Total step computation time ({} iterations): {}",
            niter,
            instant::now() - t0
        );

        params.dt
    }
}
