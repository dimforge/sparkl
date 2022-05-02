use super::na::{Isometry2, Point2};
use na::{point, vector};
use sparkl2d::dynamics::{Particle, ParticleModelHandle};
use sparkl2d::parry::shape::Shape;

pub fn cube_particles(
    origin: Point2<f32>,
    ni: usize,
    nj: usize,
    model: ParticleModelHandle,
    particle_rad: f32,
    particle_density: f32,
    randomize: bool,
) -> Vec<Particle> {
    let mut particles = Vec::new();
    let mut rand32 = oorandom::Rand32::new(42);

    for i in 0..ni {
        for j in 0..nj {
            let shift = if randomize {
                let x = rand32.rand_float() * ni as f32 * particle_rad * 2.0;
                let y = rand32.rand_float() * nj as f32 * particle_rad * 2.0;
                vector![x, y]
            } else {
                let x = (i as f32) * particle_rad * 2.0;
                let y = (j as f32) * particle_rad * 2.0;
                vector![x, y]
            };
            let pos = origin + shift;
            particles.push(Particle::new(model, pos, particle_rad, particle_density));
        }
    }

    particles
}

pub fn sample_shape(
    shape: &dyn Shape,
    position: Isometry2<f32>,
    model: ParticleModelHandle,
    particle_rad: f32,
    particle_density: f32,
    randomize: bool,
) -> Vec<Particle> {
    let mut particles = vec![];
    let mut rand32 = oorandom::Rand32::new(42);
    let aabb = shape.compute_local_aabb();

    let mut x = aabb.mins.x;
    let mut y = aabb.mins.y;

    while x <= aabb.maxs.x {
        while y <= aabb.maxs.y {
            let pt = if randomize {
                point![
                    aabb.mins.x + rand32.rand_float() * (aabb.maxs.x - aabb.mins.x),
                    aabb.mins.y + rand32.rand_float() * (aabb.maxs.y - aabb.mins.y)
                ]
            } else {
                point![x, y]
            };

            if shape.contains_local_point(&pt) {
                particles.push(Particle::new(
                    model,
                    position * pt,
                    particle_rad,
                    particle_density,
                ));
            }

            y += particle_rad * 2.0;
        }

        x += particle_rad * 2.0;
        y = aabb.mins.y;
    }

    particles
}
