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

    let cell_width = 0.1;
    let mut colliders = ColliderSet::new();
    let ground_height = cell_width * 10.0;
    let ground_shift = cell_width * 40.0;
    colliders.insert(
        ColliderBuilder::cuboid(1000.0, ground_height)
            .translation(vector![0.0, ground_shift - ground_height])
            .friction(0.0)
            .build(),
    );
    colliders.insert(
        ColliderBuilder::cuboid(ground_height, 1000.0)
            .translation(vector![ground_shift - ground_height, 0.0])
            .friction(0.0)
            .build(),
    );

    colliders.insert(
        ColliderBuilder::cuboid(ground_height, 1000.0)
            .translation(vector![
                ground_shift - ground_height + ground_shift * 8.0,
                0.0
            ])
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
        point![
            ground_shift + cell_width * 2.0 + cell_width / 4.0,
            ground_shift + cell_width * 2.0 + cell_width / 4.0
        ],
        300,
        300,
        fluids_model,
        cell_width / 4.0,
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
