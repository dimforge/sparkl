use crate::kernels::GpuCollider;
use cust::{
    error::CudaResult,
    memory::{DeviceBuffer, DevicePointer},
};
use kernels::GpuColliderShape;
use parry::shape::CudaHeightField;
use rapier::geometry::ColliderSet;

pub struct CudaColliderSet {
    buffer: DeviceBuffer<GpuCollider>,
    // NOTE: keep this to keep the cuda buffers allocated.
    _collider_buffers: Vec<CudaHeightField>,
    len: usize,
}

impl CudaColliderSet {
    pub fn new() -> CudaResult<Self> {
        Self::from_collider_set(&ColliderSet::new())
    }

    pub fn from_collider_set(collider_set: &ColliderSet) -> CudaResult<Self> {
        let mut gpu_colliders = vec![];
        let mut collider_buffers = vec![];

        for (_, collider) in collider_set.iter() {
            if let Some(cuboid) = collider.shape().as_cuboid() {
                let gpu_collider = GpuCollider {
                    shape: GpuColliderShape::Cuboid(*cuboid),
                    position: *collider.position(),
                    friction: collider.friction(),
                };
                gpu_colliders.push(gpu_collider);
            } else if let Some(heightfield) = collider.shape().as_heightfield() {
                let cuda_heightfield = heightfield.to_cuda()?;
                let cuda_heightfield_pointer = cuda_heightfield.as_device_ptr();
                let gpu_collider = GpuCollider {
                    shape: GpuColliderShape::HeightField {
                        heightfield: cuda_heightfield_pointer,
                        bellow_heightfield_is_solid: true,
                    },
                    position: *collider.position(),
                    friction: collider.friction(),
                };
                collider_buffers.push(cuda_heightfield);
                gpu_colliders.push(gpu_collider);
            }
        }

        let buffer = DeviceBuffer::from_slice(&gpu_colliders)?;
        Ok(Self {
            buffer,
            _collider_buffers: collider_buffers,
            len: gpu_colliders.len(),
        })
    }

    pub fn device_elements(&mut self) -> (DevicePointer<GpuCollider>, usize) {
        (self.buffer.as_device_ptr(), self.len)
    }
}
