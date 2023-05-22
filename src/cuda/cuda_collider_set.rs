use crate::core::prelude::BoundaryHandling;
use crate::kernels::GpuCollider;
use cust::{
    error::CudaResult,
    memory::{DeviceBuffer, DevicePointer},
};
use kernels::GpuColliderShape;
use parry::math::{Point, Real};
use parry::shape::{CudaHeightField, CudaTriMesh};
use parry::utils::CudaArray1;
use rapier::geometry::{ColliderHandle, ColliderSet};

pub struct CudaColliderSet {
    buffer: DeviceBuffer<GpuCollider>,
    // NOTE: keep this to keep the cuda buffers allocated.
    _heightfield_buffers: Vec<CudaHeightField>,
    _trimesh_buffers: Vec<CudaTriMesh>,
    _polyline_buffers: Vec<CudaArray1<Point<Real>>>,
    len: usize,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CudaColliderOptions {
    pub handle: ColliderHandle,
    pub penalty_stiffness: f32,
    pub flip_interior: bool,
    pub grid_boundary_handling: BoundaryHandling,
}

impl Default for CudaColliderOptions {
    fn default() -> Self {
        Self {
            handle: ColliderHandle::invalid(),
            penalty_stiffness: 0.0,
            flip_interior: false,
            grid_boundary_handling: BoundaryHandling::Friction,
        }
    }
}

impl CudaColliderSet {
    pub fn new() -> CudaResult<Self> {
        Self::from_collider_set(&ColliderSet::new(), vec![])
    }

    pub fn from_collider_set(
        collider_set: &ColliderSet,
        options: Vec<CudaColliderOptions>,
    ) -> CudaResult<Self> {
        let mut gpu_colliders = vec![];
        let mut heightfield_buffers = vec![];
        let mut trimesh_buffers = vec![];
        let mut polyline_buffers = vec![];

        for (handle, collider) in collider_set.iter() {
            let options = options
                .iter()
                .find(|opt| opt.handle == handle)
                .copied()
                .unwrap_or_else(Default::default);

            if let Some(cuboid) = collider.shape().as_cuboid() {
                let gpu_collider = GpuCollider {
                    shape: GpuColliderShape::Cuboid(*cuboid),
                    position: *collider.position(),
                    friction: collider.friction(),
                    penalty_stiffness: options.penalty_stiffness,
                    grid_boundary_handling: options.grid_boundary_handling,
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
                    grid_boundary_handling: options.grid_boundary_handling,
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
                    grid_boundary_handling: options.grid_boundary_handling,
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
                    grid_boundary_handling: options.grid_boundary_handling,
                };
                polyline_buffers.push(cuda_vertices);
                gpu_colliders.push(gpu_collider);
            }
        }

        let buffer = DeviceBuffer::from_slice(&gpu_colliders)?;
        Ok(Self {
            buffer,
            _heightfield_buffers: heightfield_buffers,
            _polyline_buffers: polyline_buffers,
            _trimesh_buffers: trimesh_buffers,
            len: gpu_colliders.len(),
        })
    }

    pub fn device_elements(&mut self) -> (DevicePointer<GpuCollider>, usize) {
        (self.buffer.as_device_ptr(), self.len)
    }
}
