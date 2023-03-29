use crate::cuda::AtomicInt;
use na::ComplexField;
use sparkl_core::math::Real;
use sparkl_core::prelude::{CdfColor, ParticleCdf};

const FREE: u32 = 0;
const LOCKED: u32 = 1;

pub const ENABLE_CDF: bool = true;

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct NodeCdf {
    // The unsigned distance to the closest collider.
    pub unsigned_distance: Real,
    // The affinity and tag (inside/ outside) information stored for up to 16 colliders.
    pub color: CdfColor,
    pub closest_collider_index: u32,
    lock: u32,
}

impl Default for NodeCdf {
    fn default() -> Self {
        Self {
            unsigned_distance: f32::MAX,
            color: Default::default(),
            closest_collider_index: u32::MAX,
            lock: FREE,
        }
    }
}

impl NodeCdf {
    pub fn update(&mut self, signed_distance: Real, collider_index: u32) {
        let unsigned_distance = signed_distance.abs();

        unsafe {
            while self.lock.global_atomic_exch_acq(LOCKED) == LOCKED {}

            self.color.set_affinity(collider_index);

            // only update the tag information if the new distance is smaller than the old one
            if unsigned_distance < self.unsigned_distance {
                self.color.update_tag(collider_index, signed_distance);
                self.unsigned_distance = unsigned_distance;
                self.closest_collider_index = collider_index;
            }

            self.lock.global_atomic_exch_rel(FREE);
        }
    }
    pub fn signed_distance(&self, collider_index: u32) -> Option<f32> {
        let sign = self.color.sign(collider_index);

        if sign == 0.0 {
            None
        } else {
            Some(sign * self.unsigned_distance)
        }
    }

    pub fn is_compatible(&self, particle_cdf: &ParticleCdf) -> bool {
        self.color.is_compatible(particle_cdf.color)
    }
}
