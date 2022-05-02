use crate::dynamics::solver::MlsSolver;
use crate::dynamics::{GridNode, ParticleSet};
use crate::geometry::SpGrid;
use crate::math::DIM;
use rayon::prelude::*;
use std::sync::atomic::Ordering;

impl MlsSolver {
    pub(crate) fn evolve_eigenerosion(grid: &SpGrid<GridNode>, particles: &mut ParticleSet) {
        let cell_width = grid.cell_width();
        let particles_ptr = &std::sync::atomic::AtomicPtr::new(particles as *mut _);

        particles.regions.par_iter().for_each(|region_id| {
            for i in 0..4u64.pow(DIM as u32) {
                let cell_id = region_id.0 | (i << SpGrid::<()>::PACK_ALIGN);
                let particles: &mut ParticleSet =
                    unsafe { std::mem::transmute(particles_ptr.load(Ordering::Relaxed)) };

                let base_cell = grid.shift_cell_neg_one(cell_id);
                let cell_out = grid.get_packed(base_cell);

                grid.for_each_neighbor_packed(base_cell, |_, _, cell_in| {
                    for out_i in cell_out.particles.0..cell_out.particles.1 {
                        for in_i in cell_in.particles.0..cell_in.particles.1 {
                            if let Some((particle_out, particle_in)) =
                                particles.get_sorted_mut2(out_i as usize, in_i as usize)
                            {
                                if particle_in.crack_propagation_factor != 0.0
                                    && particle_out.crack_propagation_factor != 0.0
                                    && particle_in.phase > 0.0
                                    && particle_out.phase > 0.0
                                    && !particle_in.failed
                                    && !particle_out.failed
                                    && na::distance(&particle_in.position, &particle_out.position)
                                        <= cell_width
                                {
                                    particle_out.parameter1 +=
                                        particle_in.mass * particle_in.psi_pos;
                                    particle_out.parameter2 += particle_in.mass;
                                }
                            }
                        }
                    }
                })
            }
        });

        particles.particles.par_iter_mut().for_each(|particle| {
            if particle.crack_propagation_factor != 0.0 {
                particle.parameter1 *=
                    particle.crack_propagation_factor * cell_width / particle.parameter2;
                if particle.parameter1 > particle.crack_threshold {
                    particle.phase = 0.0;
                }
            }
        });
    }
}
