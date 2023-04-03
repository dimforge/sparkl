use crate::{
    core::{prelude::BoundaryCondition, rigid_particles::RigidParticle},
    cuda::generate_rigid_particles::{generate_collider_mesh, generate_rigid_particles},
    kernels::GpuCollider,
};
use cust::{error::CudaResult, memory::DeviceBuffer};
use kernels::{GpuColliderSet, GpuColliderShape};
use parry::{
    math::{Point, Real},
    shape::{CudaHeightField, CudaTriMesh},
    utils::CudaArray1,
};
use rapier::geometry::{ColliderHandle, ColliderSet};

pub struct CudaColliderSet {
    pub gpu_colliders: Vec<GpuCollider>,
    pub rigid_particles: Vec<RigidParticle>,
    collider_buffer: DeviceBuffer<GpuCollider>,
    rigid_particles_buffer: DeviceBuffer<RigidParticle>,
    vertex_buffer: DeviceBuffer<Point<Real>>,
    index_buffer: DeviceBuffer<u32>,
    // NOTE: keep this to keep the cuda buffers allocated.
    // Todo: remove this once the CDF is the default
    _heightfield_buffers: Vec<CudaHeightField>,
    _trimesh_buffers: Vec<CudaTriMesh>,
    _polyline_buffers: Vec<CudaArray1<Point<Real>>>,
}

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

impl CudaColliderSet {
    pub fn new() -> CudaResult<Self> {
        Self::from_collider_set(&ColliderSet::new(), vec![], 0.0)
    }

    pub fn from_collider_set(
        collider_set: &ColliderSet,
        options: Vec<CudaColliderOptions>,
        cell_width: Real,
    ) -> CudaResult<Self> {
        let mut gpu_colliders = vec![];
        let mut heightfield_buffers = vec![];
        let mut trimesh_buffers = vec![];
        let mut polyline_buffers = vec![];
        let mut rigid_particles = vec![];
        let mut vertices = vec![];
        let mut indices = vec![];

        for (handle, collider) in collider_set.iter() {
            let options = options
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

            if let Some(cuboid) = collider.shape().as_cuboid() {
                let gpu_collider = GpuCollider {
                    shape: GpuColliderShape::Cuboid(*cuboid),
                    position: *collider.position(),
                    friction: collider.friction(),
                    penalty_stiffness: options.penalty_stiffness,
                    boundary_condition: options.boundary_condition,
                };
                gpu_colliders.push(gpu_collider);
            } else if let Some(heightfield) = collider.shape().as_heightfield() {
                let cuda_heightfield = heightfield.to_cuda()?;
                let cuda_heightfield_pointer = cuda_heightfield.as_device_ptr();
                let gpu_collider = GpuCollider {
                    shape: GpuColliderShape::HeightField {
                        heightfield: cuda_heightfield_pointer,
                        bellow_heightfield_is_solid: true,
                        flip_interior: options.flip_interior,
                    },
                    position: *collider.position(),
                    friction: collider.friction(),
                    penalty_stiffness: options.penalty_stiffness,
                    boundary_condition: options.boundary_condition,
                };
                heightfield_buffers.push(cuda_heightfield);
                gpu_colliders.push(gpu_collider);
            } else if let Some(trimesh) = collider.shape().as_trimesh() {
                let cuda_trimesh = trimesh.to_cuda()?;
                let cuda_trimesh_pointer = cuda_trimesh.as_device_ptr();
                let gpu_collider = GpuCollider {
                    shape: GpuColliderShape::TriMesh {
                        trimesh: cuda_trimesh_pointer,
                        flip_interior: options.flip_interior,
                    },
                    position: *collider.position(),
                    friction: collider.friction(),
                    penalty_stiffness: options.penalty_stiffness,
                    boundary_condition: options.boundary_condition,
                };
                trimesh_buffers.push(cuda_trimesh);
                gpu_colliders.push(gpu_collider);
            } else if let Some(polyline) = collider.shape().as_polyline() {
                let cuda_vertices = CudaArray1::new(polyline.vertices())?;
                let gpu_collider = GpuCollider {
                    shape: GpuColliderShape::Polyline {
                        vertices: cuda_vertices.as_device_ptr(),
                        flip_interior: options.flip_interior,
                    },
                    position: *collider.position(),
                    friction: collider.friction(),
                    penalty_stiffness: options.penalty_stiffness,
                    boundary_condition: options.boundary_condition,
                };
                polyline_buffers.push(cuda_vertices);
                gpu_colliders.push(gpu_collider);
            } else {
                let gpu_collider = GpuCollider {
                    shape: GpuColliderShape::Any,
                    position: *collider.position(),
                    friction: collider.friction(),
                    penalty_stiffness: options.penalty_stiffness,
                    boundary_condition: options.boundary_condition,
                };
                gpu_colliders.push(gpu_collider);
            }
        }

        println!("GPU colliders count: {}", gpu_colliders.len());
        println!("Rigid particles count: {}", rigid_particles.len());
        println!("Vertex count: {}", vertices.len());
        println!("Index count: {}", indices.len());

        let collider_buffer = DeviceBuffer::from_slice(&gpu_colliders)?;
        let rigid_particles_buffer = DeviceBuffer::from_slice(&rigid_particles)?;
        let vertex_buffer = DeviceBuffer::from_slice(&vertices)?;
        let index_buffer = DeviceBuffer::from_slice(&indices)?;

        Ok(Self {
            gpu_colliders,
            rigid_particles,
            collider_buffer,
            rigid_particles_buffer,
            vertex_buffer,
            index_buffer,
            _heightfield_buffers: heightfield_buffers,
            _polyline_buffers: polyline_buffers,
            _trimesh_buffers: trimesh_buffers,
        })
    }

    pub fn device_elements(&mut self) -> GpuColliderSet {
        GpuColliderSet {
            collider_ptr: self.collider_buffer.as_device_ptr(),
            collider_count: self.gpu_colliders.len() as u32,
            rigid_particle_ptr: self.rigid_particles_buffer.as_device_ptr(),
            vertex_ptr: self.vertex_buffer.as_device_ptr(),
            index_ptr: self.index_buffer.as_device_ptr(),
        }
    }
}
