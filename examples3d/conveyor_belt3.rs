use crate::helper;
use na::{point, vector};
use rapier3d::parry;
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
    let slide_height = 0.8;
    let slide_size_x = 20.0;
    let slide_size_y = 3.0;
    let slide_offset_x = 0.4;
    let slide_offset_y = wheel_offset + 2.0 * paddle_size + 0.2;

    let cell_width = 0.1;
    let particle_rad = cell_width / 2.0;
    let block_count = 20;

    let ground_size = 5.0;
    let ground_thickness = 3.0 * cell_width;

    const NU: Real = 0.2;
    const E: Real = 1.0e7;

    for i in 0..block_count {
        let mut plasticity = DruckerPragerPlasticity::new(E, NU);
        plasticity.h0 = (65.0 as Real).to_radians();
        let sand_model = models.insert(ParticleModel::with_plasticity(
            CorotatedLinearElasticity::new(E, NU),
            plasticity,
        ));

        let block_begin_x = slide_offset_x + 0.1 + i as f32 * slide_size_x / block_count as f32;
        let block_begin_y = slide_offset_y + i as f32 * slide_size_y / block_count as f32;
        let block_end_x = block_begin_x + 0.1 + slide_size_x / block_count as f32;
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

    // let ground_handle = colliders.insert(
    //     ColliderBuilder::cuboid(ground_size * 2.0, ground_thickness, ground_size)
    //         .translation(vector![25.0, 3.0, 0.0])
    //         .friction(10.0)
    //         .build(),
    // );
    // collider_options.push(CudaColliderOptions {
    //     handle: ground_handle,
    //     flip_interior: false,
    //     enable_cdf: false,
    //     boundary_condition: BoundaryCondition::Stick,
    // });

    let slide_handle = colliders.insert(
        ColliderBuilder::trimesh(
            vec![
                point![slide_offset_x, slide_offset_y - 0.1, -slide_width - 0.1],
                point![slide_offset_x, slide_offset_y - 0.1, slide_width + 0.4],
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
                    slide_offset_y + slide_height + 0.4,
                    -slide_width - 0.1
                ],
                point![
                    slide_offset_x,
                    slide_offset_y + slide_height + 0.4,
                    slide_width + 0.1
                ],
                point![
                    slide_offset_x + slide_size_x,
                    slide_offset_y + slide_size_y + slide_height + 0.4,
                    slide_width + 0.1
                ],
                point![
                    slide_offset_x + slide_size_x,
                    slide_offset_y + slide_size_y + slide_height + 0.4,
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
                // [2, 7, 3],
                // [2, 6, 7],
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

    // Small paddle pushing the sand up.
    let ramp_segment = vector![slide_size_x, slide_size_y, 0.0];
    let num_paddles = 7;

    for i in 0..num_paddles {
        let shift = (ramp_segment / (num_paddles as f32)) * i as f32;
        let paddle_body = bodies.insert(
            RigidBodyBuilder::new(RigidBodyType::KinematicVelocityBased).linvel(ramp_segment * 0.1),
        );

        let paddle_handle = colliders.insert_with_parent(
            ColliderBuilder::trimesh(
                vec![
                    point![slide_offset_x, slide_offset_y - 0.1, -slide_width - 0.1] + shift,
                    point![slide_offset_x, slide_offset_y - 0.1, slide_width + 0.1] + shift,
                    point![
                        slide_offset_x,
                        slide_offset_y + slide_height + 0.1,
                        -slide_width - 0.1
                    ] + shift,
                    point![
                        slide_offset_x,
                        slide_offset_y + slide_height + 0.1,
                        slide_width + 0.1
                    ] + shift,
                ],
                vec![[0, 1, 2], [1, 3, 2]],
            )
            .friction(10.0)
            .build(),
            paddle_body,
            &mut bodies,
        );
        collider_options.push(CudaColliderOptions {
            handle: paddle_handle,
            enable_cdf: true,
            ..Default::default()
        });
    }

    let paddle_handle = colliders.insert(
        ColliderBuilder::trimesh(
            vec![
                point![slide_offset_x, slide_offset_y - 0.1, -slide_width - 0.1],
                point![slide_offset_x, slide_offset_y - 0.1, slide_width + 0.1],
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
            ],
            vec![[0, 1, 2], [1, 3, 2]],
        )
        .friction(10.0)
        .build(),
    );
    collider_options.push(CudaColliderOptions {
        handle: paddle_handle,
        enable_cdf: true,
        ..Default::default()
    });

    // Cone.
    let mut cone_vtx = vec![];
    let mut cone_idx = vec![];
    let nsubdiv = 10;
    let dtheta = std::f32::consts::PI * 2.0 / nsubdiv as f32;
    parry::transformation::utils::push_circle(0.8, nsubdiv, dtheta, 2.5, &mut cone_vtx);
    let nidx = cone_vtx.len() as u32;
    parry::transformation::utils::push_circle(0.8, nsubdiv, dtheta, 3.0, &mut cone_vtx);
    parry::transformation::utils::push_circle(5.0, nsubdiv, dtheta, 5.0, &mut cone_vtx);
    parry::transformation::utils::push_ring_indices(0, nidx, nsubdiv, &mut cone_idx);
    parry::transformation::utils::push_ring_indices(nidx, nidx * 2, nsubdiv, &mut cone_idx);
    cone_idx.iter_mut().for_each(|tri| tri.swap(0, 1));
    let cone_handle = colliders.insert(
        ColliderBuilder::trimesh(cone_vtx, cone_idx)
            .friction(0.1)
            .translation(vector![24.0, 2.0, 0.0]),
    );
    collider_options.push(CudaColliderOptions {
        handle: cone_handle,
        enable_cdf: true,
        ..Default::default()
    });

    // Receptabcle.
    let mut ground_vtx = vec![];
    let mut ground_idx = vec![];
    parry::transformation::utils::push_circle(
        6.0,
        4,
        std::f32::consts::PI / 2.0,
        0.0,
        &mut ground_vtx,
    );
    parry::transformation::utils::push_circle(
        6.0,
        4,
        std::f32::consts::PI / 2.0,
        2.0,
        &mut ground_vtx,
    );
    parry::transformation::utils::push_ring_indices(0, 4, 4, &mut ground_idx);
    parry::transformation::utils::push_filled_circle_indices(0, 4, &mut ground_idx);
    ground_idx.iter_mut().for_each(|tri| tri.swap(0, 1));
    let ground_handle = colliders.insert(
        ColliderBuilder::trimesh(ground_vtx, ground_idx)
            .friction(0.1)
            .translation(vector![24.0, 1.0, 0.0]),
    );
    collider_options.push(CudaColliderOptions {
        handle: ground_handle,
        enable_cdf: true,
        ..Default::default()
    });

    let mut plugin = MpmTestbedPlugin::new(models, particles, cell_width);
    plugin.collider_options = collider_options;
    testbed.add_plugin(plugin);
    testbed.set_world(bodies, colliders, impulse_joints, multibody_joints);
    testbed.integration_parameters_mut().dt = 1.0 / 60.0;
    testbed.look_at(point![-10.0, 10.0, 10.0], point![0.0, 5.0, 0.0]);
}

fn main() {
    let testbed = TestbedApp::from_builders(0, vec![("Elasticity", init_world)]);
    testbed.run()
}
