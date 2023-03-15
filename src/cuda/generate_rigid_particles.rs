use crate::core::{
    math::{Point, Real},
    rigid_particles::RigidParticle,
};
use na::distance;
use parry::shape::{TriMesh, Triangle, TypedShape};
use rapier::geometry::Collider;
use std::iter::Map;
use std::slice::Iter;

pub fn generate_rigid_particles(
    rigid_particles: &mut Vec<RigidParticle>,
    cell_width: Real,
    collider_index: u32,
    collider: &Collider,
) {
    match collider.shape().as_typed_shape() {
        TypedShape::Cuboid(cuboid) => {
            #[cfg(feature = "dim2")]
            {
                let a = Point::new(-cuboid.half_extents.x, -cuboid.half_extents.y);
                let b = Point::new(cuboid.half_extents.x, -cuboid.half_extents.y);
                let c = Point::new(cuboid.half_extents.x, cuboid.half_extents.y);
                let d = Point::new(-cuboid.half_extents.x, cuboid.half_extents.y);

                cover_triangle(
                    rigid_particles,
                    cell_width,
                    collider_index,
                    Triangle::new(a, b, c),
                );
                cover_triangle(
                    rigid_particles,
                    cell_width,
                    collider_index,
                    Triangle::new(a, c, d),
                );
            }
            #[cfg(feature = "dim3")]
            {
                let (points, triangles) = cuboid.to_trimesh();

                for triangle in triangles {
                    cover_triangle(
                        rigid_particles,
                        cell_width,
                        collider_index,
                        Triangle::new(
                            points[triangle[0] as usize],
                            points[triangle[1] as usize],
                            points[triangle[2] as usize],
                        ),
                    );
                }
            }
        }
        TypedShape::Capsule(capsule) => {
            #[cfg(feature = "dim3")]
            {
                let (points, triangles) = capsule.to_trimesh(20, 10);

                for triangle in triangles {
                    cover_triangle(
                        rigid_particles,
                        cell_width,
                        collider_index,
                        Triangle::new(
                            points[triangle[0] as usize],
                            points[triangle[1] as usize],
                            points[triangle[2] as usize],
                        ),
                    );
                }
            }
        }
        TypedShape::TriMesh(trimesh) => {
            for triangle in trimesh.triangles() {
                cover_triangle(rigid_particles, cell_width, collider_index, triangle);
            }
        }
        TypedShape::HeightField(heightfield) =>
        {
            #[cfg(feature = "dim3")]
            for triangle in heightfield.triangles() {
                cover_triangle(rigid_particles, cell_width, collider_index, triangle);
            }
        }
        TypedShape::Triangle(&triangle) => {
            cover_triangle(rigid_particles, cell_width, collider_index, triangle);
        }
        _ => {}
    }
}

fn cover_triangle(
    rigid_particles: &mut Vec<RigidParticle>,
    cell_width: Real,
    collider_index: u32,
    triangle: Triangle,
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
    // Calculate the step increments on the base and the height.
    let base_step = cell_width * base_dir;
    let height_step = cell_width * height_dir;

    let mut triangle_d = triangle.a;

    // step along the base in cell_width increments
    while distance(&triangle.a, &triangle_d) <= base_length {
        let height_ac = tan_alpha * distance(&triangle.a, &triangle_d);
        let height_bc = tan_beta * distance(&triangle.b, &triangle_d);
        let min_height = height_ac.min(height_bc);

        let mut particle_position = triangle_d;

        let mut color_index = 0;

        // step along the height in cell_width increments
        while distance(&triangle_d, &particle_position) <= min_height {
            rigid_particles.push(RigidParticle {
                position: particle_position,
                collider_index,
                color_index,
            });

            color_index += 1;

            particle_position += height_step;
        }

        triangle_d += base_step;
    }
}
