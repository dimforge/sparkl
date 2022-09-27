use crate::kernels::GpuCollider;
use cust::{
    error::CudaResult,
    memory::{DeviceBuffer, DevicePointer},
};
use kernels::GpuColliderShape;
use parry::math::{Point, Real};
use parry::shape::CudaHeightField;
use parry::utils::CudaArray1;
use rapier::geometry::{ColliderHandle, ColliderSet};

pub struct CudaColliderSet {
    buffer: DeviceBuffer<GpuCollider>,
    // NOTE: keep this to keep the cuda buffers allocated.
    _heightfield_buffers: Vec<CudaHeightField>,
    _polyline_buffers: Vec<CudaArray1<Point<Real>>>,
    len: usize,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CudaColliderOptions {
    pub handle: ColliderHandle,
    pub penalty_stiffness: f32,
    pub flip_interior: bool,
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
        let mut polyline_buffers = vec![];

        for (handle, collider) in collider_set.iter() {
            let options = options.iter().find(|opt| opt.handle == handle);

            if let Some(cuboid) = collider.shape().as_cuboid() {
                let gpu_collider = GpuCollider {
                    shape: GpuColliderShape::Cuboid(*cuboid),
                    position: *collider.position(),
                    friction: collider.friction(),
                    penalty_stiffness: options.map(|opt| opt.penalty_stiffness).unwrap_or(0.0),
                };
                gpu_colliders.push(gpu_collider);
            } else if let Some(heightfield) = collider.shape().as_heightfield() {
                let cuda_heightfield = heightfield.to_cuda()?;
                let cuda_heightfield_pointer = cuda_heightfield.as_device_ptr();
                let gpu_collider = GpuCollider {
                    shape: GpuColliderShape::HeightField {
                        heightfield: cuda_heightfield_pointer,
                        bellow_heightfield_is_solid: true,
                        flip_interior: options.map(|opt| opt.flip_interior).unwrap_or(false),
                    },
                    position: *collider.position(),
                    friction: collider.friction(),
                    penalty_stiffness: options.map(|opt| opt.penalty_stiffness).unwrap_or(0.0),
                };
                heightfield_buffers.push(cuda_heightfield);
                gpu_colliders.push(gpu_collider);
            } else if let Some(polyline) = collider.shape().as_polyline() {
                let cuda_vertices = CudaArray1::new(polyline.vertices())?;
                let gpu_collider = GpuCollider {
                    shape: GpuColliderShape::Polyline {
                        vertices: cuda_vertices.as_device_ptr(),
                        flip_interior: options.map(|opt| opt.flip_interior).unwrap_or(false),
                    },
                    position: *collider.position(),
                    friction: collider.friction(),
                    penalty_stiffness: options.map(|opt| opt.penalty_stiffness).unwrap_or(0.0),
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
            len: gpu_colliders.len(),
        })
    }

    pub fn device_elements(&mut self) -> (DevicePointer<GpuCollider>, usize) {
        (self.buffer.as_device_ptr(), self.len)
    }
}
