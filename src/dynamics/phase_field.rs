use crate::dynamics::{GridNode, GridNodeCgPhase, ParticleSet};
use crate::geometry::SpGrid;
use crate::math::{Kernel, Real};
use na::{vector, Vector2};
use rayon::prelude::*;
use std::sync::atomic::Ordering;

const NUM_CG_STEPS: usize = 25;
const RESIDUAL_PHASE: Real = 0.0; // 0.001;

pub fn update_phase_field(
    dt: Real,
    grid: &mut SpGrid<GridNode>,
    grid_phase: &mut SpGrid<GridNodeCgPhase>,
    particles: &mut ParticleSet,
) {
    init_conjugate_gradient_p2g(dt, grid, grid_phase, particles);
    perform_preconditioning(dt, grid, grid_phase, particles);

    for kk in 0..NUM_CG_STEPS {
        let residual = conjugate_gradient_step(dt, grid, grid_phase, particles);
        info!("Conjugate gradient residual [{}]: {}", kk, residual);
        if residual < 1.0e-6 {
            break;
        }
    }

    collect_results_g2p(grid_phase, particles);
}

fn init_conjugate_gradient_p2g(
    dt: Real,
    grid: &SpGrid<GridNode>,
    grid_phase: &mut SpGrid<GridNodeCgPhase>,
    particles: &ParticleSet,
) {
    let cell_width = grid.cell_width();

    // Transfer (APIC).
    let grid_phase = &std::sync::atomic::AtomicPtr::new(grid_phase as *mut _);

    particles.active_cells.par_iter().for_each(|cell_id| {
        let grid_phase: &mut SpGrid<GridNodeCgPhase> =
            unsafe { std::mem::transmute(grid_phase.load(Ordering::Relaxed)) };

        let mut weight_sum = 0.0;
        let mut phase = 0.0;
        let mut r = 0.0;

        grid.for_each_neighbor_packed(*cell_id, |_, shift, adj_cell| {
            for particle in particles
                .iter_sorted_in_range(adj_cell.particles.0 as usize..adj_cell.particles.1 as usize)
            {
                if particle.m_c > 0.0 {
                    let dpt = particle.dir_to_closest_grid_node(cell_width)
                        - shift.cast::<Real>() * cell_width;
                    let weight = Kernel::stencil_with_dir(dpt, cell_width);
                    weight_sum += weight;
                    phase += weight * particle.phase;
                    // TODO: the reference implementation uses grid.phase_field instead of particle.phase? (see FractureSimulation.h:328)
                    // r += particle.volume_def_grad() * (particle.m_c + particle.phase / dt) * weight;
                }
            }
        });

        phase *= crate::utils::inv_exact(weight_sum);

        grid.for_each_neighbor_packed(*cell_id, |_, shift, adj_cell| {
            for particle in particles
                .iter_sorted_in_range(adj_cell.particles.0 as usize..adj_cell.particles.1 as usize)
            {
                let dpt = particle.dir_to_closest_grid_node(cell_width)
                    - shift.cast::<Real>() * cell_width;
                let weight = Kernel::stencil_with_dir(dpt, cell_width);

                if particle.m_c > 0.0 {
                    r += particle.volume_def_grad() * (particle.m_c + phase / dt) * weight;
                } else {
                    r += particle.volume_def_grad() * weight;
                }
            }
        });

        let cell = grid_phase.get_packed_mut(*cell_id);
        cell.cg_ap = 0.0;
        cell.cg_init_c = phase;
        cell.cg_c = 0.0;
        cell.cg_r = r;
        cell.cg_p = cell.cg_r;
    });
}

fn conjugate_gradient_step(
    dt: Real,
    grid: &mut SpGrid<GridNode>,
    grid_phase: &mut SpGrid<GridNodeCgPhase>,
    particles: &mut ParticleSet,
) -> Real {
    compute_ap_g2p2g(dt, grid, grid_phase, particles);

    // Perform the conjugate gradient step.
    let rz_pap: Vector2<Real> = particles
        .active_cells
        .par_iter()
        .map(|cell_id| {
            let cell = grid_phase.get_packed(*cell_id);
            let z = cell.cg_prec * cell.cg_r;
            let rz = cell.cg_r * z;
            let pap = cell.cg_p * cell.cg_ap;
            vector![rz, pap]
        })
        .sum();

    // info!("rz_pap: {}", rz_pap);
    if rz_pap[1] == 0.0 {
        return 0.0;
    }

    let grid_phase_ptr = &std::sync::atomic::AtomicPtr::new(grid_phase as *mut _);
    let alpha = rz_pap[0] / rz_pap[1];

    // FIXME: we could probably avoid this second sum reduction to compute beta.
    //        The idea is to note that we already have the denominator of beta `(r_k)²`.
    //        To compute the numerator of beta, we could use the fact that:
    //        r_{k + 1}² = (r_k - alpha * A * p)²
    //                   = r_k² - 2.0 * alpha * <r_k, Ap> + alpha² * (Ap)²
    //        So we just have to also compute `(r_k) . (Ap)` and `(Ap)²` in our
    //        first reduction and combine them with alpha after the reduction to obtain
    //        beta.
    let r1r1_r1z1: Vector2<Real> = particles
        .active_cells
        .par_iter()
        .map(|cell_id| {
            let grid_phase: &mut SpGrid<GridNodeCgPhase> =
                unsafe { std::mem::transmute(grid_phase_ptr.load(Ordering::Relaxed)) };
            let cell = grid_phase.get_packed_mut(*cell_id);
            cell.cg_c += alpha * cell.cg_p;
            // if (cell.cg_c - cell.cg_init_c).abs() > 0.5 {
            //     info!(
            //         "Current phase [{}]: {}, prev: {}",
            //         *cell_id, cell.cg_c, cell.cg_init_c
            //     );
            // }
            cell.cg_r -= alpha * cell.cg_ap;
            let z = cell.cg_prec * cell.cg_r;
            vector![cell.cg_r * cell.cg_r, cell.cg_r * z]
        })
        .sum();

    let beta = r1r1_r1z1[1] / rz_pap[0];

    particles.active_cells.par_iter().for_each(|cell_id| {
        let grid_phase: &mut SpGrid<GridNodeCgPhase> =
            unsafe { std::mem::transmute(grid_phase_ptr.load(Ordering::Relaxed)) };
        let cell = grid_phase.get_packed_mut(*cell_id);
        let z = cell.cg_prec * cell.cg_r;
        cell.cg_p = z + beta * cell.cg_p;
    });

    r1r1_r1z1[0]
}

fn perform_preconditioning(
    dt: Real,
    grid: &SpGrid<GridNode>,
    grid_phase: &mut SpGrid<GridNodeCgPhase>,
    particles: &ParticleSet,
) {
    let cell_width = grid.cell_width();
    let inv_d = Kernel::inv_d(cell_width);
    let grid_phase = &std::sync::atomic::AtomicPtr::new(grid_phase as *mut _);
    let l0 = 0.5 * cell_width;

    particles.active_cells.par_iter().for_each(|cell_id1| {
        let grid_phase: &mut SpGrid<GridNodeCgPhase> =
            unsafe { std::mem::transmute(grid_phase.load(Ordering::Relaxed)) };

        let mut mii = 0.0;
        let mut hii = 0.0;

        grid.for_each_neighbor_packed(*cell_id1, |_, shift1, adj_cell1| {
            for particle in particles.iter_sorted_in_range(
                adj_cell1.particles.0 as usize..adj_cell1.particles.1 as usize,
            ) {
                let dpt1 = particle.dir_to_closest_grid_node(cell_width)
                    - shift1.cast::<Real>() * cell_width;
                let weight1 = Kernel::stencil_with_dir(dpt1, cell_width);
                let volume = particle.volume_def_grad();

                if particle.m_c > 0.0 {
                    mii += volume
                        * ((4.0 * l0 * particle.m_c * (1.0 - RESIDUAL_PHASE) * particle.psi_pos)
                            / particle.g
                            + particle.m_c
                            + 1.0 / dt)
                        * weight1;
                    hii += volume
                        * (4.0 * l0 * l0 * particle.m_c)
                        * (weight1 * inv_d)
                        * (weight1 * inv_d)
                        * dpt1.dot(&dpt1);
                } else {
                    mii += volume
                        * ((4.0 * l0 * (1.0 - RESIDUAL_PHASE) * particle.psi_pos) / particle.g)
                        * weight1;
                    hii += volume
                        * (4.0 * l0 * l0)
                        * (weight1 * inv_d)
                        * (weight1 * inv_d)
                        * dpt1.dot(&dpt1);
                }
            }
        });

        let cell_i = grid_phase.get_packed_mut(*cell_id1);
        cell_i.cg_prec = crate::utils::inv_exact(mii + hii);
        cell_i.cg_p *= cell_i.cg_prec;
    });
}

fn compute_ap_g2p2g(
    dt: Real,
    grid: &SpGrid<GridNode>,
    grid_phase: &mut SpGrid<GridNodeCgPhase>,
    particles: &mut ParticleSet,
) {
    let cell_width = grid.cell_width();
    let inv_d = Kernel::inv_d(cell_width);
    let l0 = 0.5 * cell_width;

    particles.for_each_particles_mut(|particle| {
        // APIC grid-to-particle transfer.
        particle.phase_buf.fill(0.0);
        let volume = particle.volume_def_grad();

        grid_phase.for_each_neighbor_packed(particle.grid_index, |_, elt_shift, cell| {
            let dpt = elt_shift.cast::<Real>() * cell_width
                + particle.dir_to_closest_grid_node(cell_width);
            let weight = Kernel::stencil_with_dir(dpt, cell_width);

            if particle.m_c > 0.0 {
                particle.phase_buf +=
                    volume * (4.0 * l0 * l0 * particle.m_c) * cell.cg_p * (inv_d * weight) * dpt;
            } else {
                particle.phase_buf += volume * (4.0 * l0 * l0) * cell.cg_p * (inv_d * weight) * dpt;
            }
        })
    });

    let grid_phase = &std::sync::atomic::AtomicPtr::new(grid_phase as *mut _);

    particles.active_cells.par_iter().for_each(|cell_id| {
        let grid_phase: &mut SpGrid<GridNodeCgPhase> =
            unsafe { std::mem::transmute(grid_phase.load(Ordering::Relaxed)) };

        let curr_cg_p = grid_phase.get_packed(*cell_id).cg_p;
        let mut cg_ap = 0.0;

        grid.for_each_neighbor_packed(*cell_id, |_, shift, adj_cell| {
            for particle in particles
                .iter_sorted_in_range(adj_cell.particles.0 as usize..adj_cell.particles.1 as usize)
            {
                let dpt = particle.dir_to_closest_grid_node(cell_width)
                    - shift.cast::<Real>() * cell_width;
                let weight = Kernel::stencil_with_dir(dpt, cell_width);

                if particle.m_c > 0.0 {
                    cg_ap += particle.volume_def_grad()
                        * ((4.0 * l0 * particle.m_c * (1.0 - RESIDUAL_PHASE) * particle.psi_pos)
                            / particle.g
                            + particle.m_c
                            + 1.0 / dt)
                        * weight
                        * curr_cg_p; // [Mii] * p
                } else {
                    cg_ap += particle.volume_def_grad()
                        * ((4.0 * l0 * (1.0 - RESIDUAL_PHASE) * particle.psi_pos) / particle.g
                            + 1.0)
                        * weight
                        * curr_cg_p; // [Mii] * p
                }
                cg_ap += inv_d * weight * particle.phase_buf.dot(&dpt);
                // [Hij] * p
            }
        });

        let cell_i = grid_phase.get_packed_mut(*cell_id);
        cell_i.cg_ap = cg_ap;
    });
}

fn collect_results_g2p(grid_phase: &SpGrid<GridNodeCgPhase>, particles: &mut ParticleSet) {
    let cell_width = grid_phase.cell_width();

    particles.for_each_particles_mut(|particle| {
        let mut new_phase = if particle.m_c > 0.0 {
            particle.phase
        } else {
            0.0
        };

        grid_phase.for_each_neighbor_packed(particle.grid_index, |_, elt_shift, element| {
            let dpt = elt_shift.cast::<Real>() * cell_width
                + particle.dir_to_closest_grid_node(cell_width);
            let weight = Kernel::stencil_with_dir(dpt, cell_width);

            if particle.m_c > 0.0 {
                new_phase += (element.cg_c - element.cg_init_c) * weight;
            } else {
                new_phase += element.cg_c * weight;
            }
        });

        particle.phase = particle.phase.min(new_phase).max(0.0);
    });
}
