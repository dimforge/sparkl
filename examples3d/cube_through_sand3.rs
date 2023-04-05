use crate::helper;
use na::{point, vector, DMatrix};
use rapier3d::prelude::*;
use rapier_testbed3d::{Testbed, TestbedApp};
use sparkl3d::prelude::*;
use sparkl3d::third_party::rapier::MpmTestbedPlugin;

pub fn init_world(testbed: &mut Testbed) {
    let mut models = ParticleModelSet::new();
    let mut particles = ParticleSet::new();
    let mut colliders = ColliderSet::new();
    let mut bodies = RigidBodySet::new();
    let impulse_joints = ImpulseJointSet::new();
    let multibody_joints = MultibodyJointSet::new();

    let cell_width = 0.2;
    let particle_rad = cell_width / 4.0;
    let ground_half_side = 20.0;
    let sand_width = 100;
    let sand_height = 50;
    let sand_depth = 50;

    let (nx, ny) = (40, 40);
    let mut heigths = DMatrix::zeros(nx + 1, ny + 1);

    for i in 0..=nx {
        for j in 0..=ny {
            heigths[(i, j)] = -(i as f32 * std::f32::consts::PI / (nx as f32)).sin();
        }
    }
    colliders.insert(
        ColliderBuilder::heightfield(
            heigths.into(),
            vector![ground_half_side * 2.0, 10.0, ground_half_side * 2.0],
        )
        .translation(vector![0.0, 10.0, 0.0])
        .build(),
    );

    const NU: Real = 0.2;
    const E: Real = 1.0e7;

    let plasticity = DruckerPragerPlasticity::new(E, NU);
    let sand_model = models.insert(ParticleModel::with_plasticity(
        CorotatedLinearElasticity::new(E, NU),
        plasticity,
    ));
    let sand_particles = helper::cube_particles(
        point![
            sand_width as f32 * -particle_rad,
            2.0,
            sand_depth as f32 * -particle_rad
        ],
        sand_width,
        sand_height,
        sand_depth,
        sand_model,
        particle_rad,
        2700.0,
        false,
    );
    particles.insert_batch(sand_particles);

    // let block_model = models.insert(ParticleModel::new(CorotatedLinearElasticity::new(E, NU)));
    // let mut block_particles = helper::cube_particles(
    //     point![-10.0, cell_width * 3.0 + 2.0, 0.0],
    //     25, // 40,
    //     25, // 100,
    //     25, // 40,
    //     block_model,
    //     particle_rad,
    //     2700.0,
    //     false,
    // );

    // for p in &mut block_particles {
    //     p.kinematic_vel = Some(vector![10.0, 0.0, 0.0]);
    // }

    // particles.insert_batch(block_particles);

    let block_body = RigidBodyBuilder::new(RigidBodyType::Dynamic)
        .translation(vector![-10.0, 3.0, 0.0])
        .linvel(vector![10.0, 0.0, 0.0])
        .angvel(vector![0.0, 0.0, 0.0])
        .gravity_scale(0.0)
        .build();

    let block_collider = ColliderBuilder::cuboid(1.0, 1.0, 1.0).density(1.0).build();

    let block_body_handle = bodies.insert(block_body);
    colliders.insert_with_parent(block_collider, block_body_handle, &mut bodies);

    let plugin = MpmTestbedPlugin::new(models, particles, cell_width);
    testbed.add_plugin(plugin);
    testbed.set_world(bodies, colliders, impulse_joints, multibody_joints);
    testbed.integration_parameters_mut().dt = 1.0 / 60.0;
    testbed.look_at(point![0.0, 4.0, 50.0], point![0.0, 1.0, 0.0]);
}

fn main() {
    let testbed = TestbedApp::from_builders(0, vec![("Elasticity", init_world)]);
    testbed.run()
}
