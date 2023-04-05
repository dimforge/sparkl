use crate::{
    core::prelude::BoundaryCondition,
    cuda::generate_rigid_particles::{generate_collider_mesh, generate_rigid_particles},
    kernels::GpuCollider,
};
use cust::{error::CudaResult, memory::DeviceBuffer};
use kernels::{GpuColliderShape, GpuRigidBody, GpuRigidWorld, RigidParticle};
use parry::{
    math::{Point, Real},
    shape::{CudaHeightField, CudaTriMesh},
    utils::CudaArray1,
};
use rapier::prelude::{ColliderHandle, ColliderSet, RigidBodyHandle, RigidBodySet};
use std::collections::HashMap;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CudaColliderOptions {
    pub handle: ColliderHandle,
    pub penalty_stiffness: f32,
    pub flip_interior: bool,
    pub boundary_condition: BoundaryCondition,
}

impl Default for CudaColliderOptions {
    fn default() -> Self {
        Self {
            handle: ColliderHandle::invalid(),
            penalty_stiffness: 0.0,
            flip_interior: false,
            boundary_condition: BoundaryCondition::Friction,
        }
    }
}

pub struct CudaRigidWorld {
    pub penalty_stiffness: Real,
    pub gpu_rigid_bodies: Vec<GpuRigidBody>,
    pub gpu_colliders: Vec<GpuCollider>,
    pub rigid_particles: Vec<RigidParticle>,
    rigid_body_buffer: DeviceBuffer<GpuRigidBody>,
    collider_buffer: DeviceBuffer<GpuCollider>,
    rigid_particles_buffer: DeviceBuffer<RigidParticle>,
    vertex_buffer: DeviceBuffer<Point<Real>>,
    index_buffer: DeviceBuffer<u32>,
    // maps the CPU side rigid body handle to the GPU side rigid body index
    rigid_body_map: HashMap<RigidBodyHandle, u32>,
    // NOTE: keep this to keep the cuda buffers allocated.
    // Todo: remove this once the CDF is the default
    _heightfield_buffers: Vec<CudaHeightField>,
    _trimesh_buffers: Vec<CudaTriMesh>,
    _polyline_buffers: Vec<CudaArray1<Point<Real>>>,
}

impl CudaRigidWorld {
    pub fn new(
        rigid_body_set: Option<&RigidBodySet>,
        collider_set: &ColliderSet,
        collider_options: Vec<CudaColliderOptions>,
        cell_width: Real,
    ) -> CudaResult<Self> {
        let mut rigid_body_map = HashMap::new();
        let mut gpu_rigid_bodies = vec![];
        let mut gpu_colliders = vec![];
        let mut rigid_particles = vec![];
        let mut vertices = vec![];
        let mut indices = vec![];

        let mut heightfield_buffers = vec![];
        let mut trimesh_buffers = vec![];
        let mut polyline_buffers = vec![];

        if let Some(rigid_body_set) = rigid_body_set {
            for (handle, rigid_body) in rigid_body_set.iter() {
                rigid_body_map.insert(handle, gpu_rigid_bodies.len() as u32);

                gpu_rigid_bodies.push(GpuRigidBody {
                    position: *rigid_body.position(),
                    linvel: *rigid_body.linvel(),
                    angvel: *rigid_body.angvel(),
                    mass: rigid_body.mass(),
                    center_of_mass: *rigid_body.center_of_mass(),
                });
            }
        }

        for (handle, collider) in collider_set.iter() {
            let options = collider_options
                .iter()
                .find(|opt| opt.handle == handle)
                .copied()
                .unwrap_or_else(Default::default);

            let collider_index = handle.into_raw_parts().0;

            let index_range = generate_collider_mesh(collider, &mut vertices, &mut indices);

            generate_rigid_particles(
                index_range,
                &vertices,
                &indices,
                &mut rigid_particles,
                collider_index,
                cell_width,
            );

            let (rigid_body_index, position) = if let Some(handle) = collider.parent() {
                let rigid_body_index = rigid_body_map.get(&handle).copied();
                let position = *collider.position_wrt_parent().unwrap();
                (rigid_body_index, position)
            } else {
                (None, *collider.position())
            };

            let shape = if let Some(cuboid) = collider.shape().as_cuboid() {
                GpuColliderShape::Cuboid(*cuboid)
            } else if let Some(heightfield) = collider.shape().as_heightfield() {
                let cuda_heightfield = heightfield.to_cuda()?;
                let heightfield = cuda_heightfield.as_device_ptr();
                heightfield_buffers.push(cuda_heightfield);

                GpuColliderShape::HeightField {
                    heightfield,
                    bellow_heightfield_is_solid: true,
                    flip_interior: options.flip_interior,
                }
            } else if let Some(trimesh) = collider.shape().as_trimesh() {
                let cuda_trimesh = trimesh.to_cuda()?;
                let trimesh = cuda_trimesh.as_device_ptr();
                trimesh_buffers.push(cuda_trimesh);

                GpuColliderShape::TriMesh {
                    trimesh,
                    flip_interior: options.flip_interior,
                }
            } else if let Some(polyline) = collider.shape().as_polyline() {
                let cuda_vertices = CudaArray1::new(polyline.vertices())?;
                let vertices = cuda_vertices.as_device_ptr();
                polyline_buffers.push(cuda_vertices);

                GpuColliderShape::Polyline {
                    vertices,
                    flip_interior: options.flip_interior,
                }
            } else {
                GpuColliderShape::Any
            };

            gpu_colliders.push(GpuCollider {
                shape,
                position,
                friction: collider.friction(),
                penalty_stiffness: options.penalty_stiffness,
                boundary_condition: options.boundary_condition,
                rigid_body_index,
            });
        }

        println!("GPU rigid bodies count: {}", gpu_rigid_bodies.len());
        println!("GPU colliders count: {}", gpu_colliders.len());
        println!("Rigid particles count: {}", rigid_particles.len());
        println!("Vertex count: {}", vertices.len());
        println!("Index count: {}", indices.len());

        let rigid_body_buffer = DeviceBuffer::from_slice(&gpu_rigid_bodies)?;
        let collider_buffer = DeviceBuffer::from_slice(&gpu_colliders)?;
        let rigid_particles_buffer = DeviceBuffer::from_slice(&rigid_particles)?;
        let vertex_buffer = DeviceBuffer::from_slice(&vertices)?;
        let index_buffer = DeviceBuffer::from_slice(&indices)?;

        Ok(Self {
            penalty_stiffness: 0.0,
            gpu_rigid_bodies,
            gpu_colliders,
            rigid_particles,
            collider_buffer,
            rigid_body_buffer,
            rigid_particles_buffer,
            vertex_buffer,
            index_buffer,
            rigid_body_map,
            _heightfield_buffers: heightfield_buffers,
            _polyline_buffers: polyline_buffers,
            _trimesh_buffers: trimesh_buffers,
        })
    }

    // Todo: consider updating the entire state of the rigid world instead of just the rigid bodies
    pub fn update_rigid_bodies(&mut self, rigid_bodies: &RigidBodySet) -> CudaResult<()> {
        for (handle, rigid_body) in rigid_bodies.iter() {
            if let Some(&rigid_body_index) = self.rigid_body_map.get(&handle) {
                self.gpu_rigid_bodies[rigid_body_index as usize] = GpuRigidBody {
                    position: *rigid_body.position(),
                    linvel: *rigid_body.linvel(),
                    angvel: *rigid_body.angvel(),
                    mass: rigid_body.mass(),
                    center_of_mass: *rigid_body.center_of_mass(),
                };
            }
        }

        self.rigid_body_buffer = DeviceBuffer::from_slice(&self.gpu_rigid_bodies)?;

        Ok(())
    }

    pub fn device_elements(&mut self) -> GpuRigidWorld {
        GpuRigidWorld {
            penalty_stiffness: self.penalty_stiffness,
            rigid_body_ptr: self.rigid_body_buffer.as_device_ptr(),
            collider_ptr: self.collider_buffer.as_device_ptr(),
            rigid_particle_ptr: self.rigid_particles_buffer.as_device_ptr(),
            vertex_ptr: self.vertex_buffer.as_device_ptr(),
            index_ptr: self.index_buffer.as_device_ptr(),
            collider_count: self.gpu_colliders.len() as u32,
        }
    }
}