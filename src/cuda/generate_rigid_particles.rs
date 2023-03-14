use crate::core::{
    math::{Point, Real},
    rigid_particles::RigidParticle,
};
use parry::shape::Triangle;
use rapier::geometry::Collider;

pub fn generate_rigid_particles(
    rigid_particles: &mut Vec<RigidParticle>,
    cell_width: Real,
    collider_index: u32,
    collider: &Collider,
) {
    if let Some(cuboid) = collider.shape().as_cuboid() {
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
    } else if let Some(heightfield) = collider.shape().as_heightfield() {
        for triangle in heightfield.triangles() {
            cover_triangle(rigid_particles, cell_width, collider_index, triangle);
        }
    } else if let Some(trimesh) = collider.shape().as_trimesh() {
        for triangle in trimesh.triangles() {
            cover_triangle(rigid_particles, cell_width, collider_index, triangle);
        }
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

    let primary_dir = (triangle.b - triangle.a).normalize();
    let secondary_dir = triangle.normal().unwrap().cross(&primary_dir);

    let primary_step = primary_dir * cell_width;
    let secondary_step = secondary_dir * cell_width;

    let mut edge_pos = triangle.a;

    while {
        edge_pos += primary_step;
        triangle.contains_point(&edge_pos)
    } {
        let mut particle_pos = edge_pos;

        while {
            particle_pos += secondary_step;
            triangle.contains_point(&particle_pos)
        } {
            rigid_particles.push(RigidParticle {
                position: particle_pos,
                collider_index,
            })
        }
    }
}
