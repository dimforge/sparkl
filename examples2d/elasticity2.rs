use crate::helper;
use na::{point, vector};
use rapier2d::prelude::{
    ColliderBuilder, ColliderSet, ImpulseJointSet, MultibodyJointSet, RigidBodySet,
};
use rapier_testbed2d::{Testbed, TestbedApp};
use sparkl2d::parry::shape::SharedShape;
use sparkl2d::prelude::*;
use sparkl2d::third_party::rapier::MpmTestbedPlugin;

const E: Real = 2.0e4;
const NU: Real = 0.35;
const COHESION: Real = 0.0;
const FRICTION_ANGLE: Real = 33.0;
const DILATANCY_ANGLE: Real = 0.0;
const TENSILE_STRENGTH: Real = 2.0e5;

pub fn init_world(testbed: &mut Testbed) {
    /*
     * World
     */
    let mut particles = ParticleSet::new();
    let mut models = ParticleModelSet::new();
    let mut colliders = ColliderSet::new();

    let cell_width = 0.05;
    let particle_rad = cell_width / 4.0;

    let height_offset = 0.5;
    let width = 6.0;
    let height = 3.0;
    let thickness = 0.1;

    colliders.insert(
        ColliderBuilder::cuboid(width, thickness)
            .translation(vector![width, -thickness])
            .friction(0.0)
            .build(),
    );

    colliders.insert(
        ColliderBuilder::cuboid(thickness, height)
            .translation(vector![-thickness, height - 2.0 * thickness])
            .friction(0.0)
            .build(),
    );

    colliders.insert(
        ColliderBuilder::cuboid(thickness, height)
            .translation(vector![2.0 * width + thickness, height - 2.0 * thickness])
            .friction(0.0)
            .build(),
    );

    let w = cell_width * 20.0;
    let star = vec![
        point![w, -w],
        point![w * 0.5, 0.0],
        point![w, w],
        point![0.0, w * 0.5],
        point![-w, w],
        point![-w * 0.5, 0.0],
        point![-w, -w],
        point![0.0, -w * 0.5],
    ];
    let indices: Vec<_> = (0u32..star.len() as u32)
        .map(|e| [e, (e + 1) % star.len() as u32])
        .collect();
    let shape = SharedShape::convex_decomposition(&star, &indices);

    let mut rng = oorandom::Rand32::new(42);
    let mut gen = || height_offset + cell_width * 40.0 * (rng.rand_u32() % 5 + 1) as Real;
    let model = ParticleModel::with_plasticity(
        CorotatedLinearElasticity::new(E, NU),
        RankinePlasticity::new(E, NU, 1.0e2, 5.0),
    );
    let model = models.insert(model);

    for _ in 0..5 {
        let mut block1 = helper::sample_shape(
            &*shape.0,
            Isometry::translation(gen(), gen()),
            model,
            particle_rad,
            2.0,
            false,
        );

        for p in &mut block1 {
            p.crack_propagation_factor = 0.9;
            p.crack_threshold = 1.0;
            p.m_c = 0.01;
            p.g = 20000.0;
        }

        particles.insert_batch(block1);
    }

    let plugin = MpmTestbedPlugin::with_boundaries(models, particles, cell_width, None, Some(1));
    // plugin.solver_params.damage_model = DamageModel::Eigenerosion;
    testbed.add_plugin(plugin);
    testbed.set_world(
        RigidBodySet::new(),
        colliders,
        ImpulseJointSet::new(),
        MultibodyJointSet::new(),
    );
    testbed.integration_parameters_mut().dt = 1.0 / 60.0;
    // testbed.physics_state_mut().gravity.y = -981.0;
}

fn main() {
    let testbed = TestbedApp::from_builders(0, vec![("Elasticity", init_world)]);
    testbed.run()
}
