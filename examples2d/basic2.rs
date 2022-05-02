use crate::helper;
use na::{point, vector};
use rapier2d::prelude::{
    ColliderBuilder, ColliderSet, ImpulseJointSet, MultibodyJointSet, RigidBodySet,
};
use rapier_testbed2d::{Testbed, TestbedApp};
use sparkl2d::parry::shape::SharedShape;
use sparkl2d::prelude::*;
use sparkl2d::third_party::rapier::MpmTestbedPlugin;

const E: Real = 1.0e5;
const NU: Real = 0.35;
const COHESION: Real = 0.0;
const FRICTION_ANGLE: Real = 33.0;
const DILATANCY_ANGLE: Real = 0.0;
const TENSILE_STRENGTH: Real = 2.0e5;

pub fn init_world(testbed: &mut Testbed) {
    /*
     * World
     */
    let mut models = ParticleModelSet::new();
    let mut particles = ParticleSet::new();

    let cell_width = 1.0 / 128.0;
    let mut colliders = ColliderSet::new();
    let ground_shift = cell_width * 10.0;
    let ground_half_height = 0.05;
    let ground_half_width = 0.35;

    let mut heigths = vec![];
    let n = 40;
    for i in 0..=n {
        heigths.push(-(i as f32 * std::f32::consts::PI / (n as f32)).sin());
    }
    colliders.insert(
        ColliderBuilder::heightfield(heigths.into(), vector![2.0, 1.0])
            .translation(vector![0.5, 1.5])
            .build(),
    );

    for _ in 0..1 {
        let snow_model = models.insert(ParticleModel::with_plasticity(
            CorotatedLinearElasticity::new(1.0e5, 0.2),
            SnowPlasticity::new(),
        ));

        let snow = helper::sample_shape(
            &*SharedShape::cuboid(0.1, 0.2), // w, w / 2.0),
            Isometry::translation(cell_width * 40.0, ground_shift + 0.6 + 0.2),
            snow_model,
            cell_width / 4.0,
            1000.0,
            false,
        );
        let mut sand = snow.clone();

        particles.insert_batch(snow);

        let sand_model = models.insert(ParticleModel::with_plasticity(
            CorotatedLinearElasticity::new(1.0e5, 0.2),
            DruckerPragerPlasticity::new(1.0e5, 0.2),
        ));

        for p in &mut sand {
            p.model = sand_model;
            p.position.y += 0.5;
        }
        particles.insert_batch(sand);
    }

    /*
     * Create a breakable star.
     */
    let w = cell_width * 10.0;
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

    let model = ParticleModel::with_failure(
        CorotatedLinearElasticity::new(1.0e5, 0.2),
        MaximumStressFailure::new(1.0e5, Real::MAX),
    );
    let star_model = models.insert(model);

    let star = helper::sample_shape(
        &*shape.0,
        Isometry::translation(cell_width * 40.0, 1.7),
        star_model,
        cell_width / 4.0,
        4000.0,
        false,
    );
    particles.insert_batch(star);

    let mut plugin =
        MpmTestbedPlugin::with_boundaries(models, particles, cell_width, None, Some(1));
    plugin.solver_params.max_num_substeps = 50;
    testbed.add_plugin(plugin);
    testbed.set_world(
        RigidBodySet::new(),
        colliders,
        ImpulseJointSet::new(),
        MultibodyJointSet::new(),
    );
    testbed.integration_parameters_mut().dt = 1.0 / 60.0;
}

fn main() {
    let testbed = TestbedApp::from_builders(0, vec![("Elasticity", init_world)]);
    testbed.run()
}
