use crate::gpu_collider::{GpuCollider, GpuColliderSet};
use crate::gpu_grid::{GpuGrid, GpuGridNode};
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
    boundary_handling: BoundaryHandling,
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
            boundary_handling,
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
    boundary_handling: BoundaryHandling,
    gravity: Vector<Real>,
) {
    let mut cell_velocity = (cell.momentum_velocity + cell.mass * gravity * dt)
        * sparkl_core::utils::inv_exact(cell.mass);

    // TODO: replace this by a proper boundary  handling.
    for collider in colliders.iter() {
        if let Some(proj) = collider.shape.project_point_with_max_dist(
            &collider.position,
            &cell_pos,
            false,
            cell_width * 2.0,
        ) {
            match boundary_handling {
                BoundaryHandling::Stick => {
                    if proj.is_inside {
                        cell_velocity.fill(0.0);
                        break; // No need to detect other contact because we already have a zero velocity.
                    }
                }
                BoundaryHandling::Friction => {
                    if let Some((mut normal, dist)) =
                        Unit::try_new_and_get(cell_pos - proj.point, 1.0e-5)
                    {
                        if proj.is_inside {
                            normal = -normal;
                        }

                        let normal_vel = cell_velocity.dot(&normal);

                        if normal_vel < 0.0 {
                            let dist_with_margin = dist - cell_width;
                            if proj.is_inside || dist_with_margin <= 0.0 {
                                let tangent_vel = cell_velocity - normal_vel * normal.into_inner();
                                let tangent_vel_norm = tangent_vel.norm();

                                cell_velocity = tangent_vel;

                                if tangent_vel_norm > 1.0e-10 {
                                    // Friction.
                                    cell_velocity = tangent_vel / tangent_vel_norm
                                        * (tangent_vel_norm + normal_vel * collider.friction)
                                            .max(0.0);
                                }
                            } else if -normal_vel * dt > dist_with_margin {
                                cell_velocity -=
                                    (dist_with_margin / dt + normal_vel) * normal.into_inner();
                            }
                        }
                    }
                }
            }
        }
    }

    cell.momentum_velocity = cell_velocity;
    cell.psi_momentum_velocity *= sparkl_core::utils::inv_exact(cell.psi_mass);
}
