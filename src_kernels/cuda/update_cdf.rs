use crate::{GpuGrid, NewGpuColliderSet};
use cuda_std::{thread, *};
use na::vector;
use sparkl_core::math::Real;

#[kernel]
pub unsafe fn update_cdf(mut grid: GpuGrid, collider_set: NewGpuColliderSet) {
    let particle_index = thread::block_idx_x();

    #[cfg(feature = "dim2")]
    let shift = vector![thread::thread_idx_x() as i32, thread::thread_idx_y() as i32];
    #[cfg(feature = "dim3")]
    let shift = vector![
        thread::thread_idx_x() as i32,
        thread::thread_idx_y() as i32,
        thread::thread_idx_z() as i32
    ];

    let cell_width = grid.cell_width();

    let particle = collider_set.rigid_particle(particle_index as usize);
    let collider = collider_set.collider(particle.collider_index as usize);

    let particle_position = collider.position * particle.position;

    let node_coord = (particle_position / cell_width).map(|e| e.round() as i32 - 1) + shift;
    let node_position = node_coord.map(|e| e as Real) * cell_width;

    if let Some(node_id) = grid.get_node_id_at_coord(node_coord) {
        if let Some(node) = grid.get_node_mut(node_id) {
            let signed_distance = particle.normal.dot(&(node_position - particle_position));
            let unsigned_distance = signed_distance.abs();
            let tag = if signed_distance >= 0.0 { 1 } else { 0 };
            let affinity = 1;
            let color = ((tag << 1) & affinity) << (particle.collider_index << 1);

            node.cdf_data.update(unsigned_distance, color);
        }
    }
}
