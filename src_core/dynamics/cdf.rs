use crate::math::Real;

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Zeroable)]
pub struct CdfColor(pub u32, pub u32);

impl CdfColor {
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

    pub fn check_and_correct_penetration(&mut self, previous_color: CdfColor) -> bool {
        let shared_affinities = self.affinities() & previous_color.affinities();
        let difference =
            (shared_affinities & self.tags()) ^ (shared_affinities & previous_color.tags());

        // correct the tags that penetrate a collider
        self.0 = ((self.tags() ^ difference) << 16) | self.affinities();
        self.1 = difference;

        difference != 0
    }
}
