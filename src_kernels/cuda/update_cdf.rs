use crate::{GpuColliderSet, GpuGrid, GpuGridNode};
use cuda_std::{thread, *};
use na::vector;
use parry::shape::{Segment, Triangle};
use sparkl_core::math::{Point, Real};

#[kernel]
pub unsafe fn update_cdf(mut next_grid: GpuGrid, collider_set: GpuColliderSet) {
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

    let particle_position = collider.position * particle.position;

    let node_coord = particle_position.map(|e| (e / cell_width).round() as i64 - 1) + shift;
    let node_position = node_coord.cast::<Real>() * cell_width;

    if let Some(node_id) = next_grid.get_node_id_at_coord(node_coord) {
        if let Some(node) = next_grid.get_node_mut(node_id) {
            #[cfg(feature = "dim2")]
            {
                let segment_index = particle.segment_or_triangle_index;
                let segment = collider_set.segment(segment_index, &collider.position);
                let normal = segment.normal().unwrap().into_inner();

                let signed_distance = (node_position - particle_position).dot(&normal);
                let projected_point = node_position - signed_distance * normal;

                if inside_segment(projected_point, segment) {
                    node.cdf.update(signed_distance, collider_index);
                }
            }
            #[cfg(feature = "dim3")]
            {
                let triangle_index = particle.segment_or_triangle_index;
                let triangle = collider_set.triangle(triangle_index, &collider.position);
                let normal = triangle.normal().unwrap().into_inner();

                let signed_distance = (node_position - particle_position).dot(&normal);
                let projected_point = node_position - signed_distance * normal;

                if inside_triangle(projected_point, triangle) {
                    node.cdf.update(signed_distance, collider_index);
                }
            }
        }
    }
}

fn inside_segment(point: Point<Real>, segment: Segment) -> bool {
    let ap = point - segment.a;
    let ab = segment.b - segment.a;

    let d_ap = ap.dot(&ap);
    let d_ab = ab.dot(&ab);

    let alpha = d_ap / d_ab;

    0.0 <= alpha && alpha <= 1.0
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
