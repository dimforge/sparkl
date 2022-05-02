use core::marker::PhantomData;
use parry::query::PointProjection;
use parry::shape::{Cuboid, CudaHeightFieldPointer};
use sparkl_core::math::{Isometry, Point, Real};

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone)]
#[repr(C)]
pub enum GpuColliderShape {
    Cuboid(Cuboid),
    HeightField {
        heightfield: CudaHeightFieldPointer,
        bellow_heightfield_is_solid: bool,
    },
}

impl GpuColliderShape {
    #[cfg(not(target_os = "cuda"))]
    pub fn project_point_with_max_dist(
        &self,
        _position: &Isometry<Real>,
        _point: &Point<Real>,
        _solid: bool,
        _max_dist: Real,
    ) -> Option<PointProjection> {
        unreachable!()
    }

    #[cfg(target_os = "cuda")]
    pub fn project_point_with_max_dist(
        &self,
        position: &Isometry<Real>,
        point: &Point<Real>,
        solid: bool,
        max_dist: Real,
    ) -> Option<PointProjection> {
        use parry::query::PointQuery;

        match self {
            Self::Cuboid(s) => Some(s.project_point(position, point, solid)),
            Self::HeightField {
                heightfield,
                bellow_heightfield_is_solid,
            } => {
                let local_point = position.inverse_transform_point(point);
                let mut local_aabb = heightfield.local_aabb();
                local_aabb.mins.y = -Real::MAX;
                local_aabb.maxs.y = Real::MAX;
                let mut local_proj =
                    heightfield.project_local_point_with_max_dist(&local_point, solid, max_dist)?;

                if *bellow_heightfield_is_solid {
                    local_proj.is_inside = local_aabb.contains_local_point(&local_point)
                        && local_point.y <= local_proj.point.y;
                }

                Some(local_proj.transform_by(position))
            }
        }
    }
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GpuCollider {
    pub shape: GpuColliderShape,
    pub position: Isometry<Real>,
    pub friction: Real,
}

pub struct GpuColliderSet {
    pub ptr: *const GpuCollider,
    pub len: usize,
}

impl GpuColliderSet {
    pub fn get(&self, i: usize) -> Option<&GpuCollider> {
        if i >= self.len {
            None
        } else {
            unsafe { Some(&*self.ptr.add(i)) }
        }
    }

    pub fn iter(&self) -> GpuColliderIter {
        GpuColliderIter {
            ptr: self.ptr,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

pub struct GpuColliderIter<'a> {
    ptr: *const GpuCollider,
    len: usize,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Iterator for GpuColliderIter<'a> {
    type Item = &'a GpuCollider;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            let curr = self.ptr;

            unsafe {
                self.ptr = self.ptr.offset(1);
                self.len -= 1;
                Some(&*curr)
            }
        }
    }
}
