use super::MlsSolver;
use crate::dynamics::solver::{BoundaryHandling, RigidWorld, SimulationDofs};
use crate::dynamics::{GridNode, ParticleSet};
use crate::geometry::SpGrid;
use crate::math::{Real, DIM};
use crate::parry::query::PointProjection;
use na::Unit;
use rayon::prelude::*;
use std::sync::atomic::Ordering;

impl MlsSolver {
    #[allow(dead_code)] // May become handy later.
    pub(crate) fn compute_particle_normals(rigid_world: &RigidWorld, particles: &mut ParticleSet) {
        particles.particles.par_iter_mut().for_each(|particle| {
            let mut closest_dist = Real::MAX;
            let mut closest_proj = PointProjection::new(false, particle.position);

            for (_, collider) in rigid_world.colliders.iter() {
                let proj =
                    collider
                        .shape()
                        .project_point(collider.position(), &particle.position, false);

                let new_dist = na::distance(&proj.point, &particle.position);

                if new_dist < closest_dist {
                    closest_proj = proj;
                    closest_dist = new_dist;
                }
            }

            let normal = (particle.position - closest_proj.point) / closest_dist;
            if closest_proj.is_inside {
                particle.boundary_normal = -normal;
                particle.boundary_dist = -closest_dist;
            } else {
                particle.boundary_normal = normal;
                particle.boundary_dist = closest_dist;
            }
        })
    }

    pub(crate) fn grid_update(
        dt: Real,
        rigid_world: &RigidWorld,
        grid: &mut SpGrid<GridNode>,
        particles: &ParticleSet,
        boundary_handling: BoundaryHandling,
        simulation_dofs: SimulationDofs,
    ) {
        let cell_width = grid.cell_width();
        let grid = &std::sync::atomic::AtomicPtr::new(grid as *mut _);

        particles.regions.par_iter().for_each(|region_id| {
            for i in 0..4u64.pow(DIM as u32) {
                let cell_id = region_id.0 | (i << SpGrid::<()>::PACK_ALIGN);
                let grid: &mut SpGrid<GridNode> =
                    unsafe { std::mem::transmute(grid.load(Ordering::Relaxed)) };

                let cell_pos = grid.cell_center(cell_id);
                let cell = grid.get_packed_mut(cell_id);

                if simulation_dofs.contains(SimulationDofs::LOCK_X) {
                    cell.momentum.x = 0.0;
                    cell.velocity.x = 0.0;
                }
                if simulation_dofs.contains(SimulationDofs::LOCK_Y) {
                    cell.momentum.y = 0.0;
                    cell.velocity.y = 0.0;
                }
                #[cfg(feature = "dim3")]
                if simulation_dofs.contains(SimulationDofs::LOCK_Z) {
                    info!("Locking z");
                    cell.momentum.z = 0.0;
                    cell.velocity.z = 0.0;
                }

                for (_, collider) in rigid_world.colliders.iter() {
                    let proj =
                        collider
                            .shape()
                            .project_point(collider.position(), &cell_pos, false);

                    if !cell.boundary() {
                        cell.set_boundary(proj.is_inside);
                    }

                    match boundary_handling {
                        BoundaryHandling::Stick => {
                            if proj.is_inside {
                                cell.velocity.fill(0.0);
                            }
                        }
                        BoundaryHandling::Friction => {
                            if let Some((mut normal, dist)) =
                                Unit::try_new_and_get(cell_pos - proj.point, 1.0e-5)
                            {
                                if proj.is_inside {
                                    normal = -normal;
                                }

                                let normal_vel = cell.velocity.dot(&normal);

                                if normal_vel < 0.0 {
                                    let dist_with_margin = dist - cell_width;
                                    if proj.is_inside || dist_with_margin <= 0.0 {
                                        let tangent_vel =
                                            cell.velocity - normal_vel * normal.into_inner();
                                        let tangent_vel_norm = tangent_vel.norm();

                                        cell.velocity = tangent_vel;

                                        if tangent_vel_norm > 1.0e-10 {
                                            // Friction.
                                            cell.velocity = tangent_vel / tangent_vel_norm
                                                * (tangent_vel_norm
                                                    + normal_vel * collider.friction())
                                                .max(0.0);
                                        }
                                    } else if -normal_vel * dt > dist_with_margin {
                                        cell.velocity -= (dist_with_margin / dt + normal_vel)
                                            * normal.into_inner();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        })
    }
}
