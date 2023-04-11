use crate::core::math::{Point, Real};
use kernels::RigidParticle;
use na::distance;
use parry::shape::{Segment, Triangle, TypedShape};
use rapier::geometry::Collider;
use std::ops::Range;

// Todo: move this code somewhere else

pub fn generate_collider_mesh(
    collider: &Collider,
    vertices: &mut Vec<Point<Real>>,
    indices: &mut Vec<u32>,
) -> Range<usize> {
    let vertex_offset = vertices.len() as u32;
    let first_index = indices.len();

    #[cfg(feature = "dim2")]
    match collider.shape().as_typed_shape() {
        TypedShape::Segment(segment) => {
            vertices.extend([segment.a, segment.b]);
            indices.extend([0, 1].iter().map(|index| index + vertex_offset));
        }
        TypedShape::Cuboid(cuboid) => {
            let a = Point::new(-cuboid.half_extents.x, -cuboid.half_extents.y);
            let b = Point::new(cuboid.half_extents.x, -cuboid.half_extents.y);
            let c = Point::new(cuboid.half_extents.x, cuboid.half_extents.y);
            let d = Point::new(-cuboid.half_extents.x, cuboid.half_extents.y);

            vertices.extend([a, b, c, d]);
            indices.extend(
                [0, 1, 1, 2, 2, 3, 3, 0]
                    .iter()
                    .map(|index| index + vertex_offset),
            );
        }
        TypedShape::Polyline(polyline) => {
            vertices.extend(polyline.vertices());
            indices.extend(
                polyline
                    .indices()
                    .iter()
                    .flatten()
                    .map(|index| index + vertex_offset),
            );
        }
        TypedShape::HeightField(heightfield) => {
            extend_polyline(heightfield.to_polyline(), vertices, indices);
        }
        _ => {}
    }

    #[cfg(feature = "dim3")]
    match collider.shape().as_typed_shape() {
        TypedShape::Triangle(&triangle) => {
            vertices.extend(triangle.vertices());
            indices.extend([0, 1, 2].iter().map(|index| index + vertex_offset));
        }
        TypedShape::Cuboid(cuboid) => {
            extend_trimesh(cuboid.to_trimesh(), vertices, indices);
        }
        TypedShape::Capsule(capsule) => {
            extend_trimesh(capsule.to_trimesh(20, 10), vertices, indices);
        }
        TypedShape::TriMesh(trimesh) => {
            vertices.extend(trimesh.vertices());
            indices.extend(
                trimesh
                    .flat_indices()
                    .iter()
                    .map(|index| index + vertex_offset),
            );
        }
        TypedShape::HeightField(heightfield) => {
            extend_trimesh(heightfield.to_trimesh(), vertices, indices);
        }
        _ => {}
    }

    let last_index = indices.len();

    first_index..last_index
}

fn extend_polyline(
    (points, segments): (Vec<Point<Real>>, Vec<[u32; 2]>),
    vertices: &mut Vec<Point<Real>>,
    indices: &mut Vec<u32>,
) {
    let vertex_offset = vertices.len() as u32;

    vertices.extend(points);
    indices.extend(segments.iter().flatten().map(|index| index + vertex_offset));
}

fn extend_trimesh(
    (points, triangles): (Vec<Point<Real>>, Vec<[u32; 3]>),
    vertices: &mut Vec<Point<Real>>,
    indices: &mut Vec<u32>,
) {
    let vertex_offset = vertices.len() as u32;

    vertices.extend(points);
    indices.extend(
        triangles
            .iter()
            .flatten()
            .map(|index| index + vertex_offset),
    );
}

pub fn generate_rigid_particles(
    index_range: Range<usize>,
    vertices: &[Point<Real>],
    indices: &[u32],
    rigid_particles: &mut Vec<RigidParticle>,
    collider_index: u32,
    cell_width: Real,
) {
    // let cell_width = cell_width / 2.0;

    #[cfg(feature = "dim2")]
    for (segment_index, segment) in indices[index_range.clone()].chunks(2).enumerate() {
        let segment = Segment {
            a: vertices[segment[0] as usize],
            b: vertices[segment[1] as usize],
        };

        let segment_index = (2 * segment_index + index_range.start) as u32;

        cover_segment(
            segment,
            rigid_particles,
            cell_width,
            collider_index,
            segment_index,
        );
    }

    #[cfg(feature = "dim3")]
    for (triangle_index, triangle) in indices[index_range.clone()].chunks(3).enumerate() {
        let triangle = Triangle {
            a: vertices[triangle[0] as usize],
            b: vertices[triangle[1] as usize],
            c: vertices[triangle[2] as usize],
        };

        let triangle_index = (3 * triangle_index + index_range.start) as u32;

        cover_triangle(
            triangle,
            rigid_particles,
            cell_width,
            collider_index,
            triangle_index,
        );
    }
}

fn cover_segment(
    segment: Segment,
    rigid_particles: &mut Vec<RigidParticle>,
    cell_width: Real,
    collider_index: u32,
    segment_index: u32,
) {
    let base = segment.b - segment.a;
    let base_length = base.norm();
    let base_dir = base / base_length;

    // Calculate the step increment on the base.
    let base_step_count = (base_length / cell_width).ceil();
    let base_step = base_dir * base_length / base_step_count;

    for i in 0..=base_step_count as u32 {
        let particle_position = segment.a + (i as f32) * base_step;

        rigid_particles.push(RigidParticle {
            position: particle_position,
            collider_index,
            segment_or_triangle_index: segment_index,
            color_index: 0,
        });
    }
}

// Cover the triangle with rigid particles. They should be spaced apart no more than the cell_width.
fn cover_triangle(
    triangle: Triangle,
    rigid_particles: &mut Vec<RigidParticle>,
    cell_width: Real,
    collider_index: u32,
    triangle_index: u32,
) {
    // select the longest edge as the base
    let distance_ab = distance(&triangle.b, &triangle.a);
    let distance_bc = distance(&triangle.c, &triangle.b);
    let distance_ca = distance(&triangle.a, &triangle.c);
    let max = distance_ab.max(distance_bc).max(distance_ca);

    let triangle = if max == distance_bc {
        Triangle {
            a: triangle.b,
            b: triangle.c,
            c: triangle.a,
        }
    } else if max == distance_ca {
        Triangle {
            a: triangle.c,
            b: triangle.a,
            c: triangle.b,
        }
    } else {
        triangle
    };

    let ac = triangle.c - triangle.a;
    let base = triangle.b - triangle.a;
    let base_length = base.norm();
    let base_dir = base / base_length;

    // Calculate the step increment on the base.
    let base_step_count = (base_length / cell_width).ceil();
    let base_step = base_dir * base_length / base_step_count;

    // Project C on the base AB.
    let ac_offset_length = ac.dot(&base_dir);
    let bc_offset_length = base_length - ac_offset_length;
    // Compute the triangle’s height vector.
    let height = ac - base_dir * ac_offset_length;
    let height_length = height.norm();
    let height_dir = height / height_length;
    // Calculate the tangents.
    let tan_alpha = height_length / ac_offset_length;
    let tan_beta = height_length / bc_offset_length;

    for i in 0..=base_step_count as u32 {
        let base_position = triangle.a + (i as f32) * base_step;

        let height_ac = tan_alpha * distance(&triangle.a, &base_position);
        let height_bc = tan_beta * distance(&triangle.b, &base_position);
        let height_length = height_ac.min(height_bc);

        // Calculate the step increment on the height.
        let height_step_count = (height_length / cell_width).ceil();
        let height_step = height_dir * height_length / height_step_count;

        for j in 0..=height_step_count as u32 {
            let particle_position = base_position + (j as f32) * height_step;

            rigid_particles.push(RigidParticle {
                position: particle_position,
                collider_index,
                segment_or_triangle_index: triangle_index,
                color_index: j as u32,
            });
        }
    }
}
