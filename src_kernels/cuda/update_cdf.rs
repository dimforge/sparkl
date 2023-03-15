use crate::{GpuGrid, NewGpuColliderSet, NBH_SHIFTS};
use cuda_std::{thread, *};
use sparkl_core::math::Real;

#[kernel]
pub unsafe fn update_cdf(mut grid: GpuGrid, collider_set: NewGpuColliderSet) {
    let particle_index = thread::block_idx_x();

    let particle = collider_set.rigid_particle(particle_index as usize);
    let collider = collider_set.collider(particle.collider_index as usize);

    let cell_width = grid.cell_width();
    let particle_position = collider.position * particle.position;

    let closest_node_coord = (particle_position / cell_width).map(|e| e.round() as i32 - 1);

    for shift in NBH_SHIFTS {
        let node_coord = closest_node_coord + shift.cast::<i32>();
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
}
