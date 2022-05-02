use crate::dynamics::solver::MlsSolver;
use crate::dynamics::{GridNode, ParticleModelSet, ParticleSet};
use crate::geometry::SpGrid;
use crate::math::{Kernel, DIM};
use rayon::prelude::*;
use std::sync::atomic::Ordering;

impl MlsSolver {
    pub fn recompute_fluids_volumes(
        grid: &mut SpGrid<GridNode>,
        particles: &mut ParticleSet,
        models: &ParticleModelSet,
    ) {
        let cell_width = grid.cell_width();
        info!("^^^^^ Recomputing fluids volumes.");
        // P2G
        {
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

                            let ref_elt_pos_minus_particle_pos =
                                particle.dir_to_associated_grid_node(cell_width);
                            let w = Kernel::precompute_weights(
                                ref_elt_pos_minus_particle_pos,
                                cell_width,
                            );

                            grid.for_each_neighbor_packed_mut(
                                particle.grid_index,
                                |_cell_id, shift, cell| {
                                    #[cfg(feature = "dim2")]
                                    let weight = w[0][shift.x] * w[1][shift.y];
                                    #[cfg(feature = "dim3")]
                                    let weight = w[0][shift.x] * w[1][shift.y] * w[2][shift.z];
                                    cell.mass += weight * particle.mass;
                                },
                            );
                        }
                    });
            }
        }

        // G2P
        {
            let particles_ptr = std::sync::atomic::AtomicPtr::new(particles as *mut _);

            particles.particle_bins.par_iter().for_each(|particle_bin| {
                let particles: &mut ParticleSet =
                    unsafe { std::mem::transmute(particles_ptr.load(Ordering::Relaxed)) };

                for ii in 0..4 {
                    if particle_bin[ii] == usize::MAX {
                        break;
                    }

                    let particle = &mut particles.particles[particle_bin[ii]];
                    if !models[particle.model].constitutive_model.is_fluid() {
                        // Only affect the fluids.
                        // TODO: could we do something similar with non-fluids particles?
                        //       For example we could divide F by det(F) and mulitply by
                        //       the fun det(F) computed here?
                        continue;
                    }

                    let mut new_mass = 0.0;

                    let ref_elt_pos_minus_particle_pos =
                        particle.dir_to_associated_grid_node(cell_width);
                    let w = Kernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);

                    grid.for_each_neighbor_packed(particle.grid_index, |_cell_id, shift, cell| {
                        #[cfg(feature = "dim2")]
                        let weight = w[0][shift.x] * w[1][shift.y];
                        #[cfg(feature = "dim3")]
                        let weight = w[0][shift.x] * w[1][shift.y] * w[2][shift.z];

                        new_mass += weight * cell.mass;
                    });

                    let new_density = new_mass / cell_width.powi(DIM as i32);
                    let new_volume = particle.mass / new_density;
                    particle.deformation_gradient[(0, 0)] = new_volume / particle.volume0;
                }
            });
        }

        // Reset the modified cells.
        {
            let grid = &std::sync::atomic::AtomicPtr::new(grid as *mut _);

            particles.regions.par_iter().for_each(|region_id| {
                for i in 0..4u64.pow(DIM as u32) {
                    let cell_id = region_id.0 | (i << SpGrid::<()>::PACK_ALIGN);
                    let grid: &mut SpGrid<GridNode> =
                        unsafe { std::mem::transmute(grid.load(Ordering::Relaxed)) };
                    let cell = grid.get_packed_mut(cell_id);
                    cell.mass = 0.0;
                }
            });
        }
    }
}
