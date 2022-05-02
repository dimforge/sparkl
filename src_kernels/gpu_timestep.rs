use sparkl_core::math::Real;

#[cfg_attr(
    not(target_os = "cuda"),
    derive(cust::DeviceCopy)
)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(transparent)]
pub struct GpuTimestepLength(pub u64);

impl Default for GpuTimestepLength {
    fn default() -> Self {
        GpuTimestepLength(0)
    }
}

impl GpuTimestepLength {
    pub const FACTOR: f32 = 1_000_000_000_000.0;

    pub fn from_sec(dt: Real) -> Self {
        Self((dt * Self::FACTOR) as u64)
    }

    pub fn into_sec(self) -> f32 {
        self.0 as f32 / Self::FACTOR
    }
}
