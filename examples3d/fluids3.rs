use crate::helper;
use na::point;
use rapier3d::math::Point;
use rapier3d::prelude::{ColliderSet, ImpulseJointSet, MultibodyJointSet, RigidBodySet};
use rapier_testbed3d::{Testbed, TestbedApp};
use sparkl3d::prelude::*;
use sparkl3d::third_party::rapier::MpmTestbedPlugin;

pub fn init_world(testbed: &mut Testbed) {
    /*
     * World
     */

    let mut models = ParticleModelSet::new();
    let mut particles = ParticleSet::new();

    let cell_width = 0.8;
    let fluid_model = models.insert(ParticleModel::new(MonaghanSphEos {
        pressure0: 1.0e6,
        gamma: 7,
        viscosity: 1.01e-3,
        max_neg_pressure: 1.0,
    }));

    let fluid = helper::cube_particles(
        point![1.6, 1.6, 1.6],
        38,
        20,
        20,
        fluid_model,
        0.1,
        1000.0,
        false,
    );
    particles.insert_batch(fluid);

    let bodies = RigidBodySet::new();
    let colliders = ColliderSet::new();
    let impulse_joints = ImpulseJointSet::new();
    let multibody_joints = MultibodyJointSet::new();

    let mut plugin = MpmTestbedPlugin::new(models, particles, cell_width);
    plugin.solver_params.force_fluids_volume_recomputation = true;
    testbed.add_plugin(plugin);
    testbed.set_world(bodies, colliders, impulse_joints, multibody_joints);
    testbed.integration_parameters_mut().dt = 1.0 / 60.0;
    testbed.look_at(Point::new(0.0, 16.0, 0.0), point![6.0, 10.0, 6.0]);
}

fn main() {
    let testbed = TestbedApp::from_builders(0, vec![("Elasticity", init_world)]);
    testbed.run()
}
