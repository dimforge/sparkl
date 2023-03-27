use crate::{GpuGrid, NewGpuColliderSet};
use cuda_std::{thread, *};
use na::vector;
use parry::shape::Triangle;
use sparkl_core::math::{Point, Real};

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
    let particle = collider_set.rigid_particle(particle_index);

    let collider_index = particle.collider_index;
    let collider = collider_set.collider(collider_index);

    let triangle_index = particle.triangle_index;
    let triangle = collider_set.triangle(triangle_index, &collider.position);

    let particle_position = collider.position * particle.position;
    let normal = triangle.normal().unwrap().into_inner();

    let node_coord = particle_position.map(|e| (e / cell_width).round() as i64 - 1) + shift;
    let node_position = node_coord.cast::<Real>() * cell_width;

    if let Some(node_id) = next_grid.get_node_id_at_coord(node_coord) {
        if let Some(node) = next_grid.get_node_mut(node_id) {
            let signed_distance = (node_position - particle_position).dot(&normal);
            let projected_point = node_position - signed_distance * normal;

            if inside_triangle(projected_point, triangle) {
                node.cdf.update(signed_distance, collider_index);
            }
        }
    }
}

fn inside_triangle(point: Point<Real>, triangle: Triangle) -> bool {
    // determine the barycentric coordinates to determine if the point is inside the triangle
    // from: https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    let ab = triangle.b - triangle.a;
    let ac = triangle.c - triangle.a;
    let ap = point - triangle.a;

    let d00 = ab.dot(&ab);
    let d01 = ab.dot(&ac);
    let d11 = ac.dot(&ac);
    let d20 = ap.dot(&ab);
    let d21 = ap.dot(&ac);
    let denom = d00 * d11 - d01 * d01;

    let alpha = (d11 * d20 - d01 * d21) / denom;
    let beta = (d00 * d21 - d01 * d20) / denom;
    let gamma = 1.0 - alpha - beta;

    // slight tolerance to avoid discarding valid points on the edge
    // this might not bee needed at all
    // Todo: reconsider this
    let min = -0.0000001;
    let max = 1.0000001;

    min <= alpha && alpha <= max && min <= beta && beta <= max && min <= gamma && gamma <= max
}
