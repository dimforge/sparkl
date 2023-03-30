use crate::helper;
use na::{point, vector, DMatrix};
use rapier3d::math::Point;
use rapier3d::prelude::{
    ColliderBuilder, ColliderSet, ImpulseJointSet, MultibodyJointSet, RigidBodySet,
};
use rapier_testbed3d::{Testbed, TestbedApp};
use sparkl3d::prelude::*;
use sparkl3d::third_party::rapier::MpmTestbedPlugin;

pub fn init_world(testbed: &mut Testbed) {
    /*
     * World
     */
    let mut models = ParticleModelSet::new();
    let mut particles = ParticleSet::new();

    let mut colliders = ColliderSet::new();
    let ground_half_side = 20.0;

    let (nx, ny) = (40, 40);
    let mut heigths = DMatrix::zeros(nx + 1, ny + 1);

    for i in 0..=nx {
        for j in 0..=ny {
            heigths[(i, j)] = 0.0;
        }
    }

    let quad_size = 5.0_f32;
    let a = Point::new(-quad_size, 0.0, 0.0);
    let b = Point::new(quad_size, 0.0, 0.0);
    let c = Point::new(quad_size, quad_size, 0.0);
    let d = Point::new(-quad_size, quad_size, 0.0);

    colliders.insert(
        ColliderBuilder::heightfield(
            heigths.into(),
            vector![ground_half_side * 2.0, 10.0, ground_half_side * 2.0],
        )
        .build(),
    );
    colliders.insert(ColliderBuilder::triangle(a, b, c).build());
    colliders.insert(ColliderBuilder::triangle(a, c, d).build());
    colliders.insert(
        ColliderBuilder::triangle(a, b, c)
            .rotation(vector![0.0, std::f32::consts::PI / 2.0, 0.0])
            .build(),
    );
    colliders.insert(
        ColliderBuilder::triangle(a, c, d)
            .rotation(vector![0.0, std::f32::consts::PI / 2.0, 0.0])
            .build(),
    );

    const NU: Real = 0.2;
    const E: Real = 1.0e7;

    let plasticity = DruckerPragerPlasticity::new(E, NU);
    let sand_model = models.insert(ParticleModel::with_plasticity(
        CorotatedLinearElasticity::new(E, NU),
        plasticity,
    ));
    let n = 50;
    let cell_width = 0.2;
    let particle_rad = cell_width / 4.0;
    let offset = -particle_rad * n as f32;
    let sand_particles = helper::cube_particles(
        point![offset, 6.0, offset],
        n,
        n,
        n,
        sand_model,
        particle_rad,
        2700.0,
        false,
    );

    particles.insert_batch(sand_particles);

    let bodies = RigidBodySet::new();
    let impulse_joints = ImpulseJointSet::new();
    let multibody_joints = MultibodyJointSet::new();

    let plugin = MpmTestbedPlugin::new(models, particles, cell_width);
    testbed.add_plugin(plugin);
    testbed.set_world(bodies, colliders, impulse_joints, multibody_joints);
    testbed.integration_parameters_mut().dt = 1.0 / 60.0;
    testbed.look_at(
        Point::new(ground_half_side, 4.0, ground_half_side + 20.0),
        point![ground_half_side, 1.0, ground_half_side],
    );
}

fn main() {
    let testbed = TestbedApp::from_builders(0, vec![("Elasticity", init_world)]);
    testbed.run()
}