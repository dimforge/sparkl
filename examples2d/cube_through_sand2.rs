use crate::helper;
use na::{point, vector};
use rapier2d::prelude::*;
use rapier_testbed2d::{Testbed, TestbedApp};
use sparkl2d::cuda::CudaColliderOptions;
use sparkl2d::prelude::*;
use sparkl2d::third_party::rapier::MpmTestbedPlugin;

pub fn init_world(testbed: &mut Testbed) {
    let mut models = ParticleModelSet::new();
    let mut particles = ParticleSet::new();
    let mut colliders = ColliderSet::new();
    let mut bodies = RigidBodySet::new();

    let cell_width = 0.1;
    let particle_rad = cell_width / 2.0;
    let particles_width = 100;
    let particles_height = 50;

    let height_offset = 10.0 * particle_rad;
    let width = 30.0;
    let height = (particles_height as f32 + 1.0) * particle_rad * 2.0;
    let thickness = 3.0 * cell_width;

    colliders.insert(
        ColliderBuilder::cuboid(width, thickness)
            .translation(vector![0.0, -thickness])
            .build(),
    );

    let block_body = RigidBodyBuilder::new(RigidBodyType::Dynamic)
        .translation(vector![-6.0, 2.0])
        .linvel(vector![20.0, 0.0])
        .angvel(0.0)
        .gravity_scale(1.0)
        .build();

    let block_collider = ColliderBuilder::cuboid(1.0, 1.0).density(10000.0).build();

    let block_body_handle = bodies.insert(block_body);
    let block_collider_handle =
        colliders.insert_with_parent(block_collider, block_body_handle, &mut bodies);

    let collider_options = vec![CudaColliderOptions {
        handle: block_collider_handle,
        enable_cdf: true,
        ..Default::default()
    }];

    let sand_model = models.insert(ParticleModel::with_plasticity(
        CorotatedLinearElasticity::new(1.0e6, 0.2),
        DruckerPragerPlasticity::new(1.0e6, 0.2),
    ));

    let block1 = helper::cube_particles(
        point![5.0 - (particles_width as f32) * particle_rad, height_offset],
        particles_width,
        particles_height,
        sand_model,
        particle_rad,
        1000.0,
        false,
    );

    particles.insert_batch(block1);

    let mut plugin = MpmTestbedPlugin::new(models, particles, cell_width);
    plugin.solver_params.force_fluids_volume_recomputation = true;
    plugin.collider_options = collider_options;
    testbed.add_plugin(plugin);
    testbed.set_world(
        bodies,
        colliders,
        ImpulseJointSet::new(),
        MultibodyJointSet::new(),
    );
    testbed.integration_parameters_mut().dt = 1.0 / 60.0;
    testbed.look_at(Point::new(0.0, 3.0), 30.0);
}

fn main() {
    let testbed = TestbedApp::from_builders(0, vec![("Elasticity", init_world)]);
    testbed.run()
}
