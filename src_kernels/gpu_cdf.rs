use crate::cuda::AtomicInt;
use cuda_std::GpuFloat;
use na::{matrix, vector, Matrix, Matrix3, Matrix4, Vector3, Vector4};
use sparkl_core::math::{Real, Vector};

const FREE: u32 = 0;
const LOCKED: u32 = 1;

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct CdfColor(pub u32);

impl CdfColor {
    pub fn new(affinity: u32, tag: u32, collider_index: u32) -> Self {
        Self((affinity << collider_index) | (tag << (collider_index + 16)))
    }

    pub fn affinities(&self) -> u32 {
        self.0 & 0xFFFF
    }

    pub fn tags(&self) -> u32 {
        (self.0 >> 16) & 0xFFFF
    }

    pub fn affinity(&self, collider_index: u32) -> u32 {
        1 & (self.0 >> collider_index)
    }

    pub fn tag(&self, collider_index: u32) -> u32 {
        1 & (self.0 >> (collider_index + 16))
    }

    pub fn sign(&self, collider_index: u32) -> Real {
        let affinity = self.affinity(collider_index);
        let tag = self.tag(collider_index);

        affinity as Real * if tag == 1 { 1.0 } else { -1.0 }
    }

    pub fn set_affinity(&mut self, collider_index: u32) {
        self.0 |= 1 << collider_index;
    }

    pub fn update_tag(&mut self, collider_index: u32, signed_distance: Real) {
        let value = if signed_distance >= 0.0 { 1 } else { 0 };
        // sets the bit at collider_index + 16 to the value
        let offset = collider_index + 16;
        self.0 = self.0 & !(1 << offset) | (value << offset);
    }

    pub fn is_compatible(&self, other: CdfColor) -> bool {
        // check whether all tags match for all shared affinities
        let shared_affinities = self.affinities() & other.affinities();
        shared_affinities & self.tags() == shared_affinities & other.tags()
    }
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct NodeCdf {
    // The unsigned distance to the closest collider.
    pub unsigned_distance: Real,
    // The affinity and tag (inside/ outside) information stored for up to 16 colliders.
    pub color: CdfColor,
    lock: u32,
}

impl Default for NodeCdf {
    fn default() -> Self {
        Self {
            unsigned_distance: 0.0,
            color: Default::default(),
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
            }

            self.lock.global_atomic_exch_rel(FREE);
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct InterpolatedCdfData {
    color: CdfColor,
    weighted_tags: [Real; 16],
    #[cfg(feature = "dim2")]
    weight_matrix: Matrix3<Real>,
    #[cfg(feature = "dim2")]
    weight_vector: Vector3<Real>,
    #[cfg(feature = "dim3")]
    weight_matrix: Matrix4<Real>,
    #[cfg(feature = "dim3")]
    weight_vector: Vector4<Real>,
}

impl Default for InterpolatedCdfData {
    fn default() -> Self {
        Self {
            color: Default::default(),
            weighted_tags: [na::zero(); 16],
            weight_matrix: na::zero(),
            weight_vector: na::zero(),
        }
    }
}

impl InterpolatedCdfData {
    pub fn interpolate_color(&mut self, node_cdf: NodeCdf, weight: Real) {
        let unsigned_distance = node_cdf.unsigned_distance;

        // or together all affinities
        self.color.0 |= node_cdf.color.affinities();

        for collider_index in 0..16 {
            // make sure to only include tags into the weight, that are actually valid
            // weight them by their signed distances to prevent floating point errors
            let sign = self.color.sign(collider_index as u32);

            self.weighted_tags[collider_index] += weight * sign * unsigned_distance;
        }
    }

    pub fn compute_tags(&mut self) {
        // turn the weighted tags into the proper tags of the particle
        for collider_index in 0..16 {
            let weighted_tag = self.weighted_tags[collider_index];
            self.color.update_tag(collider_index as u32, weighted_tag);
        }
    }

    pub fn interpolate_distance_and_normal(
        &mut self,
        node_cdf: NodeCdf,
        weight: Real,
        difference: Vector<Real>,
    ) {
        // for now lets assume we only have a single collider
        let affinity = node_cdf.color.affinity(0);

        if affinity == 0 {
            return;
        }

        let particle_tag = self.color.tag(0);
        let node_tag = node_cdf.color.tag(0);

        let sign = if particle_tag == node_tag { 1.0 } else { -1.0 };
        let unsigned_distance = node_cdf.unsigned_distance;

        let distance = sign * unsigned_distance;
        let outer_product = difference * difference.transpose();

        #[cfg(feature = "dim2")]
        {
            self.weight_vector += weight * vector![distance, difference.x, difference.y];
        }
        #[cfg(feature = "dim3")]
        {
            self.weight_vector +=
                weight * vector![distance, difference.x, difference.y, difference.z];
        }

        #[cfg(feature = "dim2")]
        {
            self.weight_matrix += weight
                * matrix![
                    1.0,          difference.x,          difference.y;
                    difference.x, outer_product[(0, 0)], outer_product[(0, 1)];
                    difference.y, outer_product[(1, 0)], outer_product[(1, 1)]
                ];
        }

        #[cfg(feature = "dim3")]
        {
            self.weight_matrix += weight
                * distance
                * matrix![
                    1.0,          difference.x,          difference.y,          difference.z;
                    difference.x, outer_product[(0, 0)], outer_product[(0, 1)], outer_product[(0, 2)];
                    difference.y, outer_product[(1, 0)], outer_product[(1, 1)], outer_product[(1, 2)];
                    difference.z, outer_product[(2, 0)], outer_product[(2, 1)], outer_product[(2, 2)]
                ];
        }
    }

    pub fn compute_particle_cdf(&self) -> ParticleCdf {
        if let Some(inverse_matrix) = self.weight_matrix.try_inverse() {
            let result = inverse_matrix * self.weight_vector;

            let signed_distance = result.x;
            let gradient = result.remove_row(0);
            let unsigned_distance = signed_distance.abs();
            let normal = gradient.normalize();

            ParticleCdf {
                color: self.color,
                unsigned_distance,
                normal,
            }
        } else {
            ParticleCdf {
                color: self.color,
                unsigned_distance: na::zero(),
                normal: na::zero(),
            }
        }
    }
}

pub struct ParticleCdf {
    color: CdfColor,
    unsigned_distance: Real,
    normal: Vector<Real>,
}

impl ParticleCdf {
    pub fn is_compatible(&self, node_cdf: &NodeCdf) -> bool {
        self.color.is_compatible(node_cdf.color)
    }
}
