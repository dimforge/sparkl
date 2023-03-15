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
        TypedShape::Capsule(capsule) => {
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
        TypedShape::TriMesh(trimesh) => {
            for triangle in trimesh.triangles() {
                cover_triangle(rigid_particles, cell_width, collider_index, triangle);
            }
        }
        TypedShape::HeightField(heightfield) => {
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
    // step along the AB edge in cell_width increments (primary_dir)
    // additionally, step perpendicular to this edge in cell_width increments (secondary_dir)
    // push all points inside the triangle onto the rigid particles vec

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

    let ab = triangle.b - triangle.a;
    let ac = triangle.c - triangle.a;
    let bc = triangle.c - triangle.b;
    let normal = ab.cross(&ac).normalize();
    let alpha = Real::acos(ab.dot(&ac) / ab.norm() / ac.norm());
    let beta = Real::acos(-ab.dot(&bc) / ab.norm() / bc.norm());
    let tan_alpha = alpha.tan();
    let tan_beta = beta.tan();

    let primary_length = ab.norm();
    let primary_dir = ab.normalize();
    let primary_step = primary_dir * cell_width;

    let secondary_dir = normal.cross(&primary_dir).normalize();
    let secondary_step = secondary_dir * cell_width;

    let mut edge_position = triangle.a;

    while distance(&triangle.a, &edge_position) <= primary_length {
        let secondary_length_a = tan_alpha * distance(&triangle.a, &edge_position);
        let secondary_length_b = tan_beta * distance(&triangle.b, &edge_position);
        let secondary_length = secondary_length_a.min(secondary_length_b);

        let mut particle_position = edge_position;

        let mut color_index = 0;

        while distance(&edge_position, &particle_position) <= secondary_length {
            rigid_particles.push(RigidParticle {
                position: particle_position,
                collider_index,
                color_index,
            });

            color_index += 1;

            particle_position += secondary_step;
        }

        edge_position += primary_step;
    }
}
