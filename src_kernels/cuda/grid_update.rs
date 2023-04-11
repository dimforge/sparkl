use crate::{
    gpu_grid::{GpuGrid, GpuGridNode, GpuGridProjectionStatus},
    gpu_rigid_world::GpuRigidWorld,
    BlockHeaderId,
};
use cuda_std::{thread, *};
use na::{vector, Unit};
use sparkl_core::{
    dynamics::solver::BoundaryCondition,
    math::{Point, Real, Vector},
};

#[kernel] // NOTE: must be called with 4x4x4 (in 3D) or 4x4 (in 2D) threads per block.
pub unsafe fn grid_update(
    dt: Real,
    mut next_grid: GpuGrid,
    rigid_world: GpuRigidWorld,
    gravity: Vector<Real>,
    enable_cdf: bool,
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
        if false {
            // Todo: use this after the CDF transition
            update_cell(dt, cell, gravity);
        } else {
            let cell_pos = cell_pos_int.cast::<Real>() * cell_width;

            update_single_cell(dt, cell, cell_pos.into(), cell_width, &rigid_world, gravity);
        }
    }
}

fn update_single_cell(
    dt: Real,
    cell: &mut GpuGridNode,
    cell_pos: Point<Real>,
    cell_width: Real,
    rigid_world: &GpuRigidWorld,
    gravity: Vector<Real>,
) {
    let mut cell_velocity = (cell.momentum_or_velocity + cell.mass * gravity * dt)
        * sparkl_core::utils::inv_exact(cell.mass);

    if cell.projection_status == GpuGridProjectionStatus::NotComputed {
        let mut best_proj = None;
        for (i, collider) in rigid_world.iter_colliders().enumerate() {
            if collider.boundary_condition == BoundaryCondition::None || collider.enable_cdf {
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
            let collider = rigid_world.collider(collider_id as u32);

            match collider.boundary_condition {
                BoundaryCondition::Stick => {
                    if is_inside {
                        cell_velocity = na::zero();
                    }
                }
                BoundaryCondition::Slip => todo!(),
                BoundaryCondition::Friction | BoundaryCondition::FrictionZUp => {
                    if let Some((mut normal, dist)) =
                        Unit::try_new_and_get(cell.projection_scaled_dir, 1.0e-5)
                    {
                        if is_inside {
                            normal = -normal;
                        }

                        #[cfg(feature = "dim2")]
                        let apply_friction = true; // In 2D, Friction and FrictionZUp act the same.
                        #[cfg(feature = "dim3")]
                        let apply_friction = collider.boundary_condition
                            == BoundaryCondition::Friction
                            || (collider.boundary_condition == BoundaryCondition::FrictionZUp
                                && normal.z >= 0.0);

                        if apply_friction {
                            let normal_vel = cell_velocity.dot(&normal);

                            if normal_vel < 0.0 {
                                let dist_with_margin = dist - cell_width;
                                if is_inside || dist_with_margin <= 0.0 {
                                    let tangent_vel =
                                        cell_velocity - normal_vel * normal.into_inner();
                                    let tangent_vel_norm = tangent_vel.norm();

                                    cell_velocity = tangent_vel;

                                    if tangent_vel_norm > 1.0e-10 {
                                        let friction = collider.friction;
                                        cell_velocity = tangent_vel / tangent_vel_norm
                                            * (tangent_vel_norm + normal_vel * friction).max(0.0);
                                    }
                                } else if -normal_vel * dt > dist_with_margin {
                                    cell_velocity -=
                                        (dist_with_margin / dt + normal_vel) * normal.into_inner();
                                }
                            }
                        }
                    }
                }
                BoundaryCondition::None => {}
            }
        }
        _ => {}
    }

    cell.momentum_or_velocity = cell_velocity;
    cell.psi_momentum_or_velocity *= sparkl_core::utils::inv_exact(cell.psi_mass);
}

fn update_cell(dt: Real, cell: &mut GpuGridNode, gravity: Vector<Real>) {
    cell.momentum_or_velocity = (cell.momentum_or_velocity + cell.mass * gravity * dt)
        * sparkl_core::utils::inv_exact(cell.mass);
    cell.psi_momentum_or_velocity *= sparkl_core::utils::inv_exact(cell.psi_mass);
}
