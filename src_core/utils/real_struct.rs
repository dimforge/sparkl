use crate::math::{Point, Real, Vector};
use core::mem;

/// A structure that is composed entirely of real numbers.
pub unsafe trait RealStruct: bytemuck::Pod {
    const SIZE: usize = mem::size_of::<Self>() / mem::size_of::<Real>();

    fn cast(ptr: *const Real) -> *const Self {
        ptr as *const Self
    }

    fn cast_mut(ptr: *mut Real) -> *mut Self {
        ptr as *mut Self
    }

    fn cast_slice(slice: &[Real]) -> &[Self] {
        bytemuck::cast_slice(slice)
    }

    fn cast_slice_mut(slice: &mut [Real]) -> &mut [Self] {
        bytemuck::cast_slice_mut(slice)
    }
}

unsafe impl RealStruct for Real {}
unsafe impl RealStruct for Vector<Real> {}
unsafe impl RealStruct for Point<Real> {}
