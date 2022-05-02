use super::MlsSolver;
use crate::dynamics::models::ActiveTimestepBounds;
use crate::dynamics::{GridNode, ParticleModelSet, ParticleSet};
use crate::geometry::SpGrid;
use crate::math::{Real, DIM};
use ordered_float::NotNan;
use rayon::prelude::*;

impl MlsSolver {
    pub(crate) fn adaptive_timestep_length(
        max_dt: Real,
        grid: &SpGrid<GridNode>,
        particles: &ParticleSet,
        models: &ParticleModelSet,
    ) -> Real {
        let cell_width = grid.cell_width();
        let d = (cell_width * cell_width) / 4.0;

        particles
            .order
            .par_iter()
            .map(|i| {
                let particle = &particles.particles[*i];
                let model = &models[particle.model];
                let active_timestep_bounds = model.constitutive_model.active_timestep_bounds();

                let mut dt = max_dt;
                if active_timestep_bounds.contains(ActiveTimestepBounds::PARTICLE_VELOCITY_BOUND) {
                    // Velocity-based restriction.
                    let norm_b = d * particle.velocity_gradient.norm();
                    let apic_v = norm_b * 6.0 * (DIM as Real).sqrt() / cell_width;
                    let v = particle.velocity.norm() + apic_v;
                    dt = dt.min(cell_width / v);
                }

                if active_timestep_bounds.contains(ActiveTimestepBounds::CONSTITUTIVE_MODEL_BOUND)
                    && !particle.failed
                {
                    let candidate_dt = model
                        .constitutive_model
                        .timestep_bound(&particle, cell_width);

                    dt = dt.min(candidate_dt);
                }

                NotNan::new(dt).unwrap()
            })
            .min()
            .unwrap()
            .into_inner()
    }
}
