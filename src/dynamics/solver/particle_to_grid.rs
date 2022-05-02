use super::MlsSolver;
use crate::dynamics::{GridNode, ParticleModelSet, ParticleSet};
use crate::geometry::SpGrid;
use crate::math::{Kernel, Matrix, Real, Vector, DIM};
use crate::utils;
use rayon::prelude::*;
use std::sync::atomic::Ordering;

impl MlsSolver {
    pub(crate) fn particle_to_grid_scatter(
        dt: Real,
        gravity: &Vector<Real>,
        grid: &mut SpGrid<GridNode>,
        particles: &mut ParticleSet,
        models: &ParticleModelSet,
    ) {
        let cell_width = grid.cell_width();
        let inv_d = Kernel::inv_d(cell_width);

        // Transfer (APIC).
        let grid = &std::sync::atomic::AtomicPtr::new(grid as *mut _);

        for region_color in 0u64..1 << DIM {
            particles
                .regions
                .par_iter()
                .for_each(|(region_id, particle_range)| {
                    if !SpGrid::<GridNode>::is_in_region_with_color(*region_id, region_color) {
                        return;
                    }

                    let grid: &mut SpGrid<GridNode> =
                        unsafe { std::mem::transmute(grid.load(Ordering::Relaxed)) };

                    for particle_id in &particles.order[particle_range.clone()] {
                        let particle = &particles.particles[*particle_id];

                        /*
                         * Stress update.
                         * TODO: move this to its own function?
                         */
                        let model = &models[particle.model];
                        let stress = if !particle.failed {
                            model.constitutive_model.update_particle_stress(particle)
                        } else {
                            Matrix::zeros()
                        };

                        /*
                         * Scatter-style P2G
                         */
                        let ref_elt_pos_minus_particle_pos =
                            particle.dir_to_associated_grid_node(cell_width);
                        let w =
                            Kernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);

                        // MPM-MLS: the APIC affine matrix and the velocity gradient are the same.
                        let affine = particle.mass * particle.velocity_gradient
                            - (particle.volume0 * inv_d * dt) * stress;
                        let momentum = particle.mass * particle.velocity;

                        let psi_mass = if particle.phase > 0.0
                            && particle.crack_propagation_factor != 0.0
                            && !particle.failed
                        {
                            particle.mass
                        } else {
                            0.0
                        };
                        let psi_pos_momentum = psi_mass * particle.psi_pos;

                        grid.for_each_neighbor_packed_mut(
                            particle.grid_index,
                            |_cell_id, shift, cell| {
                                let dpt = ref_elt_pos_minus_particle_pos
                                    + shift.cast::<Real>() * cell_width;
                                #[cfg(feature = "dim2")]
                                let weight = w[0][shift.x] * w[1][shift.y];
                                #[cfg(feature = "dim3")]
                                let weight = w[0][shift.x] * w[1][shift.y] * w[2][shift.z];

                                cell.mass += weight * particle.mass;
                                cell.momentum += weight * (affine * dpt + momentum);
                                cell.psi_momentum += weight * psi_pos_momentum;
                                cell.psi_mass += weight * psi_mass;

                                // TODO: do this here or do this in another loop on grid nodes?
                                cell.velocity = (cell.momentum + cell.mass * gravity * dt)
                                    * utils::inv_exact(cell.mass);
                            },
                        );
                    }
                });
        }
    }
}
