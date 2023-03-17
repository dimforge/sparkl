use crate::{GpuGrid, NewGpuColliderSet};
use cuda_std::{thread, *};
use na::vector;
use sparkl_core::math::Real;

#[kernel]
pub unsafe fn update_cdf(mut next_grid: GpuGrid, collider_set: NewGpuColliderSet) {
    let cell_width = next_grid.cell_width();

    #[cfg(feature = "dim2")]
    let shift = vector![thread::thread_idx_x() as i64, thread::thread_idx_y() as i64];
    #[cfg(feature = "dim3")]
    let shift = vector![
        thread::thread_idx_x() as i64,
        thread::thread_idx_y() as i64,
        thread::thread_idx_z() as i64
    ];

    let particle_index = thread::block_idx_x();
    let particle = collider_set.rigid_particle(particle_index as usize);

    let collider_index = particle.collider_index;
    let collider = collider_set.collider(collider_index as usize);

    let particle_position = collider.position * particle.position;

    let node_coord = particle_position.map(|e| (e / cell_width).round() as i64 - 1) + shift;
    let node_position = node_coord.cast::<Real>() * cell_width;

    if let Some(node_id) = next_grid.get_node_id_at_coord(node_coord) {
        if let Some(node) = next_grid.get_node_mut(node_id) {
            let signed_distance = particle.normal.dot(&(node_position - particle_position));

            node.cdf_data.update(signed_distance, collider_index);
        }
    }
}
