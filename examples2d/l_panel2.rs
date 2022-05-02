use crate::helper;
use na::{point, vector};
use rapier2d::prelude::{
    ColliderBuilder, ColliderSet, ImpulseJointSet, MultibodyJointSet, RigidBodySet,
};
use rapier_testbed2d::{Testbed, TestbedApp};
use sparkl2d::parry::shape::SharedShape;
use sparkl2d::prelude::*;
use sparkl2d::third_party::rapier::MpmTestbedPlugin;

const E: Real = 25.85e9;
const NU: Real = 0.18;
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

    let cell_width = 0.005;
    let mut colliders = ColliderSet::new();
    let ground_height = cell_width * 10.0;
    let ground_shift = cell_width * 40.0;
    colliders.insert(
        ColliderBuilder::cuboid(1000.0, ground_height)
            .translation(vector![0.0, ground_shift - ground_height])
            .build(),
    );

    let panel = vec![
        point![0.0, 0.0],
        point![0.25, 0.0],
        point![0.25, 0.25],
        point![0.5, 0.25],
        point![0.5, 0.5],
        point![0.0, 0.5],
    ];
    let indices: Vec<_> = (0u32..panel.len() as u32)
        .map(|e| [e, (e + 1) % panel.len() as u32])
        .collect();
    let shape = SharedShape::convex_decomposition(&panel, &indices);

    let panel_model1 = models.insert(ParticleModel::new(CorotatedLinearElasticity::new(E, NU)));
    let panel_model2 = models.insert(ParticleModel::with_failure(
        CorotatedLinearElasticity::new(E, NU),
        MaximumStressFailure::new(2.7e6, Real::MAX),
    ));

    let panel_particles = helper::sample_shape(
        &*shape.0,
        Isometry::identity(),
        panel_model1,
        cell_width / 4.0,
        2500.0,
        false,
    );
    // let kinematic_pt_id = panel_particles
    //     .iter()
    //     .map(|p| (p.position - point![0.47, 0.25]).norm())
    //     .enumerate()
    //     .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    //     .unwrap();
    // panel_particles[kinematic_pt_id.0].kinematic_vel = Some(vector![0.0, 0.1]);

    let l_panel_origins = vec![
        point![ground_shift, ground_shift],
        point![ground_shift * 8.0, ground_shift],
    ];

    // Panel 1 (global crack propagation).
    {
        let mut panel = panel_particles.clone();
        for p in &mut panel {
            p.model = panel_model1;
            p.position += l_panel_origins[0].coords;
            p.crack_propagation_factor = 4.5;
            p.crack_threshold = 89.0;

            p.m_c = 0.0;
            p.g = 10.0;
        }

        particles.insert_batch(panel);
    }

    // Panel 2 (local cracks propagation)
    {
        let mut panel = panel_particles;
        for p in &mut panel {
            p.model = panel_model2;
            p.position += l_panel_origins[1].coords;
        }

        particles.insert_batch(panel);
    }

    let mut plugin =
        MpmTestbedPlugin::with_boundaries(models, particles, cell_width, None, Some(1));
    plugin.hooks = Box::new(BoundaryConditions {
        origins: l_panel_origins,
    });
    plugin.solver_params.boundary_handling = BoundaryHandling::Stick;
    plugin.solver_params.damage_model = DamageModel::Eigenerosion;
    testbed.add_plugin(plugin);
    testbed.set_world(
        RigidBodySet::new(),
        colliders,
        ImpulseJointSet::new(),
        MultibodyJointSet::new(),
    );
    testbed.integration_parameters_mut().dt = 1.0 / 6000.0;
    testbed.physics_state_mut().gravity.fill(0.0);
}

fn main() {
    let testbed = TestbedApp::from_builders(0, vec![("Elasticity", init_world)]);
    testbed.run()
}

struct BoundaryConditions {
    pub origins: Vec<Point<Real>>,
}

impl MpmHooks for BoundaryConditions {
    fn post_grid_update_hook(&mut self, grid: &mut SpGrid<GridNode>) {
        for origin in &self.origins {
            let moved_point = origin + vector![0.47, 0.25];
            let cell_id = grid.cell_at_point(&moved_point);
            // TODO: in the paper, the prescribed velocity is 0.01 instead of 0.1
            grid.get_packed_mut(cell_id).velocity = vector![0.0, 0.1];
        }
    }
}
