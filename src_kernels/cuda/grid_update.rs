use crate::gpu_collider::{GpuCollider, GpuColliderSet};
use crate::gpu_grid::{GpuGrid, GpuGridNode, GpuGridProjectionStatus};
use crate::BlockHeaderId;
use cuda_std::thread;
use cuda_std::*;
use na::{vector, Unit};
use sparkl_core::dynamics::solver::BoundaryHandling;
use sparkl_core::math::{Point, Real, Vector};

#[kernel] // NOTE: must be called with 4x4x4 (in 3D) or 4x4 (in 2D) threads per block.
pub unsafe fn grid_update(
    dt: Real,
    mut next_grid: GpuGrid,
    colliders_ptr: *const GpuCollider,
    num_colliders: usize,
    gravity: Vector<Real>,
) {
    let bid = BlockHeaderId(thread::block_idx_x());
    #[cfg(feature = "dim2")]
    let shift = vector![
        thread::thread_idx_x() as usize,
        thread::thread_idx_y() as usize
    ];
    #[cfg(feature = "dim3")]
    let shift = vector![
        thread::thread_idx_x() as usize,
        thread::thread_idx_y() as usize,
        thread::thread_idx_z() as usize
    ];

    let block = next_grid.active_block_unchecked(bid);

    let cell_packed_id = bid.to_physical().node_id_unchecked(shift);
    let cell_pos_int = block.virtual_id.unpack_pos_on_signed_grid() * 4 + shift.cast::<i64>();
    let cell_width = next_grid.cell_width();

    if let Some(cell) = next_grid.get_node_mut(cell_packed_id) {
        let cell_pos = cell_pos_int.cast::<Real>() * cell_width;
        let collider_set = GpuColliderSet {
            ptr: colliders_ptr,
            len: num_colliders,
        };
        update_single_cell(
            dt,
            cell,
            cell_pos.into(),
            cell_width,
            &collider_set,
            gravity,
        )
    }
}

fn update_single_cell(
    dt: Real,
    cell: &mut GpuGridNode,
    cell_pos: Point<Real>,
    cell_width: Real,
    colliders: &GpuColliderSet,
    gravity: Vector<Real>,
) {
    let mut cell_velocity = (cell.momentum_velocity + cell.mass * gravity * dt)
        * sparkl_core::utils::inv_exact(cell.mass);

    if cell.projection_status == GpuGridProjectionStatus::NotComputed {
        let mut best_proj = None;
        for (i, collider) in colliders.iter().enumerate() {
            if collider.grid_boundary_handling == BoundaryHandling::None {
                continue;
            }

            if let Some(proj) = collider.shape.project_point_with_max_dist(
                &collider.position,
                &cell_pos,
                false,
                cell_width * 2.0,
            ) {
                let projection_scaled_dir = cell_pos - proj.point;
                let projection_dist = projection_scaled_dir.norm();
                if projection_dist < best_proj.map(|(_, dist, _, _)| dist).unwrap_or(Real::MAX) {
                    best_proj = Some((projection_scaled_dir, projection_dist, proj.is_inside, i));
                }
            }
        }

        if let Some((scaled_dir, _, is_inside, id)) = best_proj {
            cell.projection_status = if is_inside {
                GpuGridProjectionStatus::Inside(id)
            } else {
                GpuGridProjectionStatus::Outside(id)
            };
            cell.projection_scaled_dir = scaled_dir;
        } else {
            cell.projection_status = GpuGridProjectionStatus::TooFar;
        }
    }

    match cell.projection_status {
        GpuGridProjectionStatus::Inside(collider_id)
        | GpuGridProjectionStatus::Outside(collider_id) => {
            let is_inside = matches!(cell.projection_status, GpuGridProjectionStatus::Inside(_));
            let collider = colliders.get(collider_id).unwrap();

            match collider.grid_boundary_handling {
                BoundaryHandling::Stick => {
                    if is_inside {
                        cell_velocity = na::zero();
                    }
                }
                BoundaryHandling::Friction | BoundaryHandling::FrictionZUp => {
                    if let Some((mut normal, mut dist)) =
                        Unit::try_new_and_get(cell.projection_scaled_dir, 1.0e-5)
                    {
                        if is_inside {
                            normal = -normal;
                            dist = 0.0;
                        }

                        #[cfg(feature = "dim2")]
                        let apply_friction = true; // In 2D, Friction and FrictionZUp act the same.
                        #[cfg(feature = "dim3")]
                        let apply_friction = collider.grid_boundary_handling
                            == BoundaryHandling::Friction
                            || (collider.grid_boundary_handling == BoundaryHandling::FrictionZUp
                                && normal.z >= 0.0);

                        if apply_friction {
                            let normal_vel = cell_velocity.dot(&normal);
                            let dist_with_margin = (dist - cell_width * 0.5).max(0.0);

                            if -normal_vel * dt > dist_with_margin {
                                // NOTE: if we enter this code, normal_vel is negative.
                                let tangent_vel = cell_velocity - normal_vel * normal.into_inner();
                                let tangent_vel_norm = tangent_vel.norm();

                                let new_normal_vel = normal_vel.max(-dist_with_margin / dt);

                                // NOTE: removed_normal_vel is always positive.
                                let removed_normal_vel = new_normal_vel - normal_vel;

                                let new_tangent_vel = if tangent_vel_norm > 1.0e-6 {
                                    tangent_vel / tangent_vel_norm
                                        * (tangent_vel_norm
                                            - removed_normal_vel * collider.friction)
                                            .max(0.0)
                                } else {
                                    Vector::zeros()
                                };

                                cell_velocity =
                                    new_normal_vel * normal.into_inner() + new_tangent_vel;
                            }
                        }
                    }
                }
                BoundaryHandling::None => {}
            }
        }
        _ => {}
    }

    cell.momentum_velocity = cell_velocity;
    cell.psi_momentum_velocity *= sparkl_core::utils::inv_exact(cell.psi_mass);
}
