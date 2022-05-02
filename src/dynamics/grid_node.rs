use crate::math::{Real, Vector};

bitflags::bitflags! {
    pub struct GridNodeFlags: u32 {
        const NONE = 0;
        const ACTIVE = 1 << 0;
        const BOUNDARY = 1 << 1;
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct GridNode {
    pub mass: Real,
    pub momentum: Vector<Real>,
    pub velocity: Vector<Real>,
    pub particles: (u32, u32),
    pub flags: GridNodeFlags,
    pub psi_momentum: Real,
    pub psi_mass: Real,
}

unsafe impl bytemuck::Zeroable for GridNode {}
unsafe impl bytemuck::Pod for GridNode {}

impl Default for GridNode {
    fn default() -> Self {
        Self {
            mass: 0.0,
            momentum: na::zero(),
            velocity: na::zero(),
            particles: (0, 0),
            flags: GridNodeFlags::NONE,
            psi_momentum: 0.0,
            psi_mass: 0.0,
        }
    }
}

impl GridNode {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn active(&self) -> bool {
        self.flags.contains(GridNodeFlags::ACTIVE)
    }

    pub fn set_active(&mut self, active: bool) {
        self.flags.set(GridNodeFlags::ACTIVE, active);
    }

    pub fn boundary(&self) -> bool {
        self.flags.contains(GridNodeFlags::BOUNDARY)
    }

    pub fn set_boundary(&mut self, boundary: bool) {
        self.flags.set(GridNodeFlags::BOUNDARY, boundary);
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct GridNodeCgPhase {
    pub cg_init_c: Real,
    pub cg_c: Real,
    pub cg_ap: Real,
    pub cg_p: Real,
    pub cg_r: Real,
    pub cg_prec: Real,
}

unsafe impl bytemuck::Zeroable for GridNodeCgPhase {}
unsafe impl bytemuck::Pod for GridNodeCgPhase {}

impl Default for GridNodeCgPhase {
    fn default() -> Self {
        Self {
            cg_init_c: 0.0,
            cg_c: 0.0,
            cg_ap: 0.0,
            cg_r: 0.0,
            cg_p: 0.0,
            cg_prec: 0.0,
        }
    }
}
