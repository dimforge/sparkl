use bytemuck::Zeroable;
use cust::{
    error::CudaResult,
    memory::{CopyDestination, DeviceBuffer, DeviceCopy},
};
use kernels::DevicePointer;
use std::ops::Range;

/// A device buffer with the ability to be resized.
pub struct CudaVec<T: DeviceCopy> {
    len: usize,
    buffer: DeviceBuffer<T>,
}

impl<T: DeviceCopy> CudaVec<T> {
    pub fn from_slice(data: &[T]) -> CudaResult<Self> {
        Ok(Self {
            len: data.len(),
            buffer: DeviceBuffer::from_slice(data)?,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    pub fn buffer(&self) -> &DeviceBuffer<T> {
        &self.buffer
    }

    pub fn as_device_ptr(&self) -> DevicePointer<T> {
        self.buffer.as_device_ptr()
    }

    pub fn truncate(&mut self, new_len: usize) {
        self.len = self.len.min(new_len);
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn to_vec(&self) -> CudaResult<Vec<T>>
    where
        T: Zeroable,
    {
        let mut out = vec![T::zeroed(); self.len];
        self.buffer.index(..self.len).copy_to(&mut out)?;
        Ok(out)
    }

    pub fn append(&mut self, data: &[T]) -> CudaResult<()>
    where
        T: Zeroable,
    {
        let num_added = data.len();
        let new_len = self.len() + num_added;

        if new_len >= self.capacity() {
            // We need to grow the buffers.
            let new_capacity = (self.capacity() * 2).max(new_len);
            let new_buffer = DeviceBuffer::zeroed(new_capacity)?;

            new_buffer
                .index(..self.capacity())
                .copy_from(&self.buffer)?;

            self.buffer = new_buffer;
        }

        // At this point, we know we have enough capacity in our buffers
        // to insert the new particles.
        self.buffer
            .index(self.len..self.len + num_added)
            .copy_from(data)?;
        self.len = new_len;

        Ok(())
    }

    pub fn remove_range(&mut self, range: Range<usize>) -> CudaResult<()> {
        assert!(range.end <= self.len(), "Range index out of bounds.");

        if range.is_empty() {
            return Ok(());
        }

        let num_to_remove = range.len();
        let mut idx_to_copy = range.end;

        while idx_to_copy + num_to_remove <= self.len() {
            let mut to_remove = self
                .buffer
                .index(idx_to_copy - num_to_remove..idx_to_copy - num_to_remove + num_to_remove);
            let to_copy = self.buffer.index(idx_to_copy..idx_to_copy + num_to_remove);
            to_remove.copy_from(&to_copy)?;
            idx_to_copy += num_to_remove;
        }

        let rest_to_move = self.len() - idx_to_copy;

        if rest_to_move > 0 {
            let mut to_remove = self
                .buffer
                .index(idx_to_copy - num_to_remove..idx_to_copy - num_to_remove + rest_to_move);
            let to_copy = self.buffer.index(idx_to_copy..idx_to_copy + rest_to_move);
            to_remove.copy_from(&to_copy)?;
        }

        self.len -= num_to_remove;
        Ok(())
    }
}
