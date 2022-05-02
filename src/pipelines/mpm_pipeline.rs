use crate::dynamics::solver::{MlsSolver, MpmHooks, RigidWorld, SolverParameters};
use crate::dynamics::{GridNode, GridNodeCgPhase, ParticleModelSet, ParticleSet};
use crate::geometry::SpGrid;
use crate::math::{Real, Vector};

pub struct MpmPipeline {
    first_step: bool,
}

impl MpmPipeline {
    pub fn new() -> Self {
        MpmPipeline { first_step: true }
    }

    pub fn step(
        &mut self,
        params: &SolverParameters,
        gravity: &Vector<Real>,
        rigid_world: &RigidWorld,
        grid: &mut SpGrid<GridNode>,
        grid_phase: &mut SpGrid<GridNodeCgPhase>,
        particles: &mut ParticleSet,
        models: &ParticleModelSet,
        hooks: &mut dyn MpmHooks,
    ) -> Real {
        if self.first_step {
            // MlsSolver::init(grid, particles);
            self.first_step = false;
        }

        MlsSolver::step(
            params,
            gravity,
            rigid_world,
            grid,
            grid_phase,
            particles,
            models,
            hooks,
        )
    }
}

impl Default for MpmPipeline {
    fn default() -> Self {
        Self::new()
    }
}
