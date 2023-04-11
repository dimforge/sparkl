use crate::helper;
use na::{point, vector};
use rapier3d::prelude::*;
use rapier_testbed3d::{Testbed, TestbedApp};
use sparkl3d::{cuda::CudaColliderOptions, prelude::*, third_party::rapier::MpmTestbedPlugin};

pub fn init_world(testbed: &mut Testbed) {
    let mut models = ParticleModelSet::new();
    let mut particles = ParticleSet::new();
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut collider_options = vec![];
    let impulse_joints = ImpulseJointSet::new();
    let multibody_joints = MultibodyJointSet::new();

    let paddle_width = 2.5;
    let paddle_count = 10;
    let paddle_start = 1.0;
    let paddle_size = 2.0;
    let paddle_mass = 300.0;
    let wheel_offset = 1.0;

    let slide_width = 2.0;
    let slide_height = 0.6;
    let slide_size_x = 20.0;
    let slide_size_y = 3.0;
    let slide_offset_x = 0.4;
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
                point![block_begin_x, block_begin_y, slide_width],
                point![block_end_x, block_end_y, slide_width],
                point![block_begin_x, block_begin_y + slide_height, slide_width],
                point![block_end_x, block_end_y + slide_height, slide_width],
                point![block_begin_x, block_begin_y, -slide_width],
                point![block_end_x, block_end_y, -slide_width],
                point![block_begin_x, block_begin_y + slide_height, -slide_width],
                point![block_end_x, block_end_y + slide_height, -slide_width],
            ],
            &vec![
                [2, 6, 7],
                [2, 3, 7],
                [0, 4, 5],
                [0, 1, 5],
                [0, 2, 6],
                [0, 4, 6],
                [1, 3, 7],
                [1, 5, 7],
                [0, 2, 3],
                [0, 1, 3],
                [4, 6, 7],
                [4, 5, 7],
            ],
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
        ColliderBuilder::cuboid(ground_size, ground_thickness, ground_size)
            .translation(vector![0.0, -ground_thickness, 0.0])
            .build(),
    );

    let slide_handle = colliders.insert(
        ColliderBuilder::trimesh(
            vec![
                point![slide_offset_x, slide_offset_y - 0.1, -slide_width - 0.1],
                point![slide_offset_x, slide_offset_y - 0.1, slide_width + 0.1],
                point![
                    slide_offset_x + slide_size_x,
                    slide_offset_y + slide_size_y - 0.1,
                    slide_width + 0.1
                ],
                point![
                    slide_offset_x + slide_size_x,
                    slide_offset_y + slide_size_y - 0.1,
                    -slide_width - 0.1
                ],
                point![
                    slide_offset_x,
                    slide_offset_y + slide_height + 0.1,
                    -slide_width - 0.1
                ],
                point![
                    slide_offset_x,
                    slide_offset_y + slide_height + 0.1,
                    slide_width + 0.1
                ],
                point![
                    slide_offset_x + slide_size_x,
                    slide_offset_y + slide_size_y + slide_height + 0.1,
                    slide_width + 0.1
                ],
                point![
                    slide_offset_x + slide_size_x,
                    slide_offset_y + slide_size_y + slide_height + 0.1,
                    -slide_width - 0.1
                ],
            ],
            vec![
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 7],
                [0, 7, 4],
                [1, 6, 2],
                [1, 5, 6],
                [2, 7, 3],
                [2, 6, 7],
            ],
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
        .translation(vector![0.0, wheel_offset + paddle_size, 0.0])
        .lock_translations()
        .lock_rotations()
        .enabled_rotations(false, false, true)
        .build();

    let wheel_handle = bodies.insert(wheel);

    let mut inner_points = vec![];
    let mut left_points = vec![];
    let mut right_points = vec![];

    for i in 0..paddle_count {
        let angle1 = std::f32::consts::TAU * i as Real / paddle_count as Real;
        let angle2 = std::f32::consts::TAU * (i as Real - 0.5) / paddle_count as Real;
        let point1 = point![
            paddle_start * angle1.cos(),
            paddle_start * angle1.sin(),
            -paddle_width
        ];
        let point2 = point![
            paddle_size * angle2.cos(),
            paddle_size * angle2.sin(),
            -paddle_width
        ];
        let point3 = point![
            paddle_start * angle1.cos(),
            paddle_start * angle1.sin(),
            paddle_width
        ];
        let point4 = point![
            paddle_size * angle2.cos(),
            paddle_size * angle2.sin(),
            paddle_width
        ];

        let com = na::center(&na::center(&point1, &point2), &na::center(&point3, &point4));

        inner_points.push(point1);
        inner_points.push(point3);
        left_points.push(point2);
        left_points.push(point![0.0, 0.0, -paddle_width]);
        right_points.push(point4);
        right_points.push(point![0.0, 0.0, paddle_width]);

        let ix = (paddle_size * paddle_size) / 3.0;
        let iy = 0.0;
        let iz = (paddle_size * paddle_size) / 3.0;

        let principal_inertia = paddle_mass * vector![iy + iz, ix + iz, ix + iy];

        let paddle = ColliderBuilder::trimesh(
            vec![point1, point2, point3, point4],
            vec![[0, 1, 2], [1, 3, 2]],
        )
        .mass_properties(MassProperties::new(com, paddle_mass, principal_inertia))
        .build();

        let paddle_handle = colliders.insert_with_parent(paddle, wheel_handle, &mut bodies);

        collider_options.push(CudaColliderOptions {
            handle: paddle_handle,
            enable_cdf: true,
            ..Default::default()
        });
    }

    inner_points.push(inner_points[0]);
    inner_points.push(inner_points[1]);
    left_points.push(left_points[0]);
    right_points.push(right_points[0]);

    let inner_indices = (0..paddle_count)
        .map(|i| 2 * i)
        .flat_map(|i| [[i, i + 2, i + 1], [i + 1, i + 2, i + 3]])
        .collect::<Vec<_>>();

    let left_indices = (0..paddle_count)
        .map(|i| 2 * i)
        .flat_map(|i| [[i, i + 2, i + 1]])
        .collect::<Vec<_>>();

    let right_indices = (0..paddle_count)
        .map(|i| 2 * i)
        .flat_map(|i| [[i, i + 2, i + 1]])
        .collect::<Vec<_>>();

    let inner = ColliderBuilder::trimesh(inner_points, inner_indices).build();
    let inner_handle = colliders.insert_with_parent(inner, wheel_handle, &mut bodies);
    collider_options.push(CudaColliderOptions {
        handle: inner_handle,
        enable_cdf: true,
        ..Default::default()
    });

    let left = ColliderBuilder::trimesh(left_points, left_indices).build();
    let left_handle = colliders.insert_with_parent(left, wheel_handle, &mut bodies);
    collider_options.push(CudaColliderOptions {
        handle: left_handle,
        enable_cdf: true,
        ..Default::default()
    });

    let right = ColliderBuilder::trimesh(right_points, right_indices).build();
    let right_handle = colliders.insert_with_parent(right, wheel_handle, &mut bodies);
    collider_options.push(CudaColliderOptions {
        handle: right_handle,
        enable_cdf: true,
        ..Default::default()
    });

    let mut plugin = MpmTestbedPlugin::new(models, particles, cell_width);
    plugin.collider_options = collider_options;
    testbed.add_plugin(plugin);
    testbed.set_world(bodies, colliders, impulse_joints, multibody_joints);
    testbed.integration_parameters_mut().dt = 1.0 / 240.0;
    testbed.look_at(point![-10.0, 10.0, 10.0], point![0.0, 5.0, 0.0]);
}

fn main() {
    let testbed = TestbedApp::from_builders(0, vec![("Elasticity", init_world)]);
    testbed.run()
}
