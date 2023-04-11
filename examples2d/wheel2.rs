use crate::helper;
use na::{point, vector};
use rapier2d::prelude::*;
use rapier_testbed2d::{Testbed, TestbedApp};
use sparkl2d::{cuda::CudaColliderOptions, prelude::*, third_party::rapier::MpmTestbedPlugin};

pub fn init_world(testbed: &mut Testbed) {
    let mut models = ParticleModelSet::new();
    let mut particles = ParticleSet::new();
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut collider_options = vec![];
    let impulse_joints = ImpulseJointSet::new();
    let multibody_joints = MultibodyJointSet::new();

    let paddle_count = 10;
    let paddle_start = 1.0;
    let paddle_size = 2.0;
    let paddle_mass = 10000.0;
    let wheel_offset = 1.0;

    let slide_height = 0.6;
    let slide_size_x = 20.0;
    let slide_size_y = 3.0;
    let slide_offset_x = 0.8;
    let slide_offset_y = wheel_offset + 2.0 * paddle_size + 0.2;

    let cell_width = 0.1;
    let particle_rad = cell_width / 2.0;
    let block_count = 20;

    let ground_size = 10.0;
    let ground_thickness = 3.0 * cell_width;

    const NU: Real = 0.2;
    const E: Real = 1.0e6;

    for i in 0..block_count {
        let sand_model = models.insert(ParticleModel::with_plasticity(
            CorotatedLinearElasticity::new(E, NU),
            DruckerPragerPlasticity::new(E, NU),
        ));

        let block_begin_x = slide_offset_x + i as f32 * slide_size_x / block_count as f32;
        let block_begin_y = slide_offset_y + i as f32 * slide_size_y / block_count as f32;
        let block_end_x = block_begin_x + slide_size_x / block_count as f32;
        let block_end_y = block_begin_y + slide_size_y / block_count as f32;

        let sand_shape = SharedShape::convex_decomposition(
            &vec![
                point![block_begin_x, block_begin_y],
                point![block_end_x, block_end_y],
                point![block_end_x, block_end_y + slide_height],
                point![block_begin_x, block_begin_y + slide_height],
            ],
            &vec![[0, 1], [1, 2], [2, 3], [3, 0]],
        );
        let sand_block = helper::sample_shape(
            &*sand_shape.0,
            Isometry::default(),
            sand_model,
            particle_rad,
            1000.0,
            false,
        );
        particles.insert_batch(sand_block);
    }

    colliders.insert(
        ColliderBuilder::cuboid(ground_size, ground_thickness)
            .translation(vector![0.0, -ground_thickness])
            .build(),
    );

    let slide_handle = colliders.insert(
        ColliderBuilder::polyline(
            vec![
                point![slide_offset_x, slide_offset_y - 0.01],
                point![
                    slide_offset_x + slide_size_x * 1.1,
                    slide_offset_y + slide_size_y * 1.1 - 0.01
                ],
            ],
            None,
        )
        .friction(0.0)
        .build(),
    );
    collider_options.push(CudaColliderOptions {
        handle: slide_handle,
        enable_cdf: true,
        ..Default::default()
    });

    let wheel = RigidBodyBuilder::new(RigidBodyType::Dynamic)
        .translation(vector![0.0, wheel_offset + paddle_size])
        .lock_translations()
        .build();

    let wheel_handle = bodies.insert(wheel);

    let mut inner_points = vec![];

    for i in 0..paddle_count {
        let angle1 = std::f32::consts::TAU * i as Real / paddle_count as Real;
        let angle2 = std::f32::consts::TAU * (i as Real - 0.5) / paddle_count as Real;
        let point1 = point![paddle_start * angle1.cos(), paddle_start * angle1.sin()];
        let point2 = point![paddle_size * angle2.cos(), paddle_size * angle2.sin()];

        inner_points.push(point1);

        let principal_inertia = paddle_mass * paddle_size * paddle_size / 12.0;

        let paddle = ColliderBuilder::polyline(vec![point1, point2], None)
            .mass_properties(MassProperties::new(
                na::center(&point1, &point2),
                paddle_mass,
                principal_inertia,
            ))
            .build();

        let paddle_handle = colliders.insert_with_parent(paddle, wheel_handle, &mut bodies);

        collider_options.push(CudaColliderOptions {
            handle: paddle_handle,
            enable_cdf: true,
            ..Default::default()
        });
    }

    inner_points.push(inner_points[0]);

    let inner = ColliderBuilder::polyline(inner_points, None).build();
    let inner_handle = colliders.insert_with_parent(inner, wheel_handle, &mut bodies);
    collider_options.push(CudaColliderOptions {
        handle: inner_handle,
        enable_cdf: true,
        ..Default::default()
    });

    let mut plugin = MpmTestbedPlugin::new(models, particles, cell_width);
    plugin.collider_options = collider_options;
    testbed.add_plugin(plugin);
    testbed.set_world(bodies, colliders, impulse_joints, multibody_joints);
    testbed.integration_parameters_mut().dt = 1.0 / 240.0;
    testbed.look_at(Point::new(0.0, 3.0), 30.0);
}

fn main() {
    let testbed = TestbedApp::from_builders(0, vec![("Elasticity", init_world)]);
    testbed.run()
}
