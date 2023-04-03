use crate::helper;
use na::{point, vector};
use rapier2d::prelude::*;
use rapier_testbed2d::{Testbed, TestbedApp};
use sparkl2d::prelude::*;
use sparkl2d::third_party::rapier::MpmTestbedPlugin;

pub fn init_world(testbed: &mut Testbed) {
    /*
     * World
     */
    let mut models = ParticleModelSet::new();
    let mut particles = ParticleSet::new();
    let mut colliders = ColliderSet::new();

    let cell_width = 0.1;
    let particle_rad = cell_width / 2.0;
    let particles_width = 100;
    let particles_height = 50;

    let height_offset = 50.0 * particle_rad;
    let width = (particles_width as f32 + 1.0) * particle_rad;
    let height = (particles_height as f32 + 1.0) * particle_rad + height_offset / 2.0;
    let thickness = 2.0 * particle_rad;

    colliders.insert(
        ColliderBuilder::cuboid(width, thickness)
            .translation(vector![0.0, -thickness])
            .friction(0.0)
            .build(),
    );

    colliders.insert(
        ColliderBuilder::cuboid(thickness, height)
            .translation(vector![-width - thickness, height - 2.0 * thickness])
            .friction(0.0)
            .build(),
    );

    colliders.insert(
        ColliderBuilder::cuboid(thickness, height)
            .translation(vector![width + thickness, height - 2.0 * thickness])
            .friction(0.0)
            .build(),
    );

    let fluids_model = models.insert(ParticleModel::new(MonaghanSphEos {
        pressure0: 1.0e4,
        gamma: 7,
        viscosity: 1.01e-3,
        max_neg_pressure: 1.0,
    }));

    let block1 = helper::cube_particles(
        point![-width + 2.0 * particle_rad, height_offset],
        particles_width,
        particles_height,
        fluids_model,
        particle_rad,
        1000.0,
        false,
    );

    particles.insert_batch(block1);

    let bodies = RigidBodySet::new();

    let mut plugin = MpmTestbedPlugin::new(models, particles, cell_width);
    plugin.solver_params.force_fluids_volume_recomputation = true;

    testbed.add_plugin(plugin);
    testbed.set_world(
        bodies,
        colliders,
        ImpulseJointSet::new(),
        MultibodyJointSet::new(),
    );
    testbed.integration_parameters_mut().dt = 1.0 / 60.0;
    // testbed.look_at(Point::new(0.0, 16.0, 0.0), point![6.0, 10.0, 6.0]);
}

fn main() {
    let testbed = TestbedApp::from_builders(0, vec![("Elasticity", init_world)]);
    testbed.run()
}
