use crate::gpu_collider::{GpuCollider, GpuColliderSet};
use crate::gpu_grid::{GpuGrid, GpuGridNode, GpuGridProjectionStatus};
use crate::BlockHeaderId;
use cuda_std::thread;
use cuda_std::*;
use na::{vector, Unit};
use sparkl_core::dynamics::solver::BoundaryHandling;
use sparkl_core::math::{Point, Real, Vector};

#[cfg_attr(target_os = "cuda", kernel)] // NOTE: must be called with 4x4x4 (in 3D) or 4x4 (in 2D) threads per block.
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

fn sdf(cell_width: Real, colliders: &GpuColliderSet, query: Point<Real>) -> Option<Real> {
    colliders
        .iter()
        .filter(|collider| collider.grid_boundary_handling != BoundaryHandling::None)
        .filter_map(|collider| {
            collider.shape.project_point_with_max_dist(
                &collider.position,
                &query,
                false,
                cell_width * 2.0,
            )
        })
        .map(|projection| {
            (projection.point - query).norm() * if projection.is_inside { -1. } else { 1. }
        })
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
}

// approximated with centered finite difference
fn sdf_gradient(cell_width: Real, colliders: &GpuColliderSet, query: Point<Real>) -> Vector<Real> {
    const FACTOR: Real = 0.1;
    let sdf = |offset| sdf(cell_width, colliders, query + offset);
    let partial = |direction: Unit<Vector<Real>>| {
        let offset_pos = direction.scale(cell_width * FACTOR);
        let offset_neg = -offset_pos;
        match (sdf(offset_pos), sdf(offset_neg)) {
            (Some(sample_pos), Some(sample_neg)) => {
                (sample_pos - sample_neg) / cell_width / FACTOR / 2.
            }
            _ => 0.,
        }
    };
    Vector::new(
        partial(Vector::x_axis()),
        partial(Vector::y_axis()),
        #[cfg(feature = "dim3")]
        partial(Vector::z_axis()),
    )
    .try_normalize(1.0e-5)
    .unwrap_or(Vector::zeros())
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

        cell.collision_normal = sdf_gradient(cell_width, colliders, cell_pos);
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
                    if cell.collision_normal != Vector::zeros() {
                        let normal = cell.collision_normal;
                        let dist = cell.projection_scaled_dir.norm();

                        #[cfg(feature = "dim2")]
                        let apply_friction = true; // In 2D, Friction and FrictionZUp act the same.
                        #[cfg(feature = "dim3")]
                        let apply_friction = collider.grid_boundary_handling
                            == BoundaryHandling::Friction
                            || (collider.grid_boundary_handling == BoundaryHandling::FrictionZUp
                                && normal.z >= 0.0);

                        if apply_friction {
                            let normal_vel = cell_velocity.dot(&normal);

                            if normal_vel < 0.0 {
                                let dist_with_margin = dist - cell_width;
                                if is_inside || dist_with_margin <= 0.0 {
                                    let tangent_vel = cell_velocity - normal_vel * normal;
                                    let tangent_vel_norm = tangent_vel.norm();

                                    cell_velocity = tangent_vel;

                                    if tangent_vel_norm > 1.0e-10 {
                                        let friction = collider.friction;
                                        cell_velocity = tangent_vel / tangent_vel_norm
                                            * (tangent_vel_norm + normal_vel * friction).max(0.0);
                                    }
                                } else if -normal_vel * dt > dist_with_margin {
                                    cell_velocity -= (dist_with_margin / dt + normal_vel) * normal;
                                }
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
