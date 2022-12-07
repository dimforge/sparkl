use crate::math::{Matrix, Point, Real, Vector};
use crate::utils::RealStruct;
#[cfg(not(feature = "std"))]
use na::ComplexField;

/*
 *========================
 *                       *
 * MENDATORY COMPONENTS  *
 *                       *
 *========================
 */

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ParticlePosition {
    pub point: Point<Real>,
}
unsafe impl RealStruct for ParticlePosition {}

impl ParticlePosition {
    pub fn closest_grid_pos(&self, cell_width: Real) -> Point<Real> {
        (self.point / cell_width).map(|e| e.round()) * cell_width
    }

    pub fn associated_cell_index_in_block_off_by_two(&self, cell_width: Real) -> Point<u32> {
        (self.point / cell_width).map(|e| {
            let assoc_cell = e.round() as i32 - 2;
            // TODO: use assoc_cell.unstable_div_floor(4) instead once it’s stabilized.
            let assoc_block = (assoc_cell as Real / 4.0).floor() as i32 * 4;
            (assoc_cell - assoc_block) as u32 // Will always be positive.
        })
    }

    pub fn associated_grid_pos(&self, cell_width: Real) -> Point<Real> {
        ((self.point / cell_width).map(|e| e.round()) - Vector::repeat(1.0)) * cell_width
    }

    pub fn dir_to_closest_grid_node(&self, cell_width: Real) -> Vector<Real> {
        self.closest_grid_pos(cell_width) - self.point
    }

    pub fn dir_to_associated_grid_node(&self, cell_width: Real) -> Vector<Real> {
        self.associated_grid_pos(cell_width) - self.point
    }
}

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ParticleVelocity {
    pub vector: Vector<Real>,
}

unsafe impl RealStruct for ParticleVelocity {}

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ParticleVolume {
    pub mass: Real,
    pub volume0: Real,
    pub radius0: Real,
    pub deformation_gradient: Matrix<Real>, // Elastic part of the deformation gradient.
    pub plastic_deformation_gradient_det: Real, // Determinant of the plastic part of the deformation gradient.
}
unsafe impl RealStruct for ParticleVolume {}
impl Default for ParticleVolume {
    fn default() -> Self {
        Self {
            mass: 0.0,
            volume0: 0.0,
            radius0: 0.0,
            deformation_gradient: Matrix::identity(),
            plastic_deformation_gradient_det: 1.0,
        }
    }
}

impl ParticleVolume {
    pub fn density0(&self) -> Real {
        self.mass / self.volume0
    }

    pub fn fluid_deformation_gradient_det(&self) -> Real {
        self.deformation_gradient[(0, 0)]
    }

    pub fn volume_fluid(&self) -> Real {
        self.volume0 * self.fluid_deformation_gradient_det()
    }

    pub fn volume_def_grad(&self) -> Real {
        self.volume0 * self.deformation_gradient.determinant()
    }

    pub fn density_fluid(&self) -> Real {
        self.density0() / self.fluid_deformation_gradient_det()
    }

    pub fn density_def_grad(&self) -> Real {
        self.density0() / self.deformation_gradient.determinant()
    }
}

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Zeroable)]
#[repr(C)]
pub struct ParticleStatus {
    // TODO: these three bool should be a single enum, or a single bitflags.
    pub failed: bool,
    pub is_static: bool,
    pub kinematic_vel_enabled: bool,
    pub kinematic_vel: Vector<Real>,
    pub model_index: usize,
}
impl Default for ParticleStatus {
    fn default() -> Self {
        Self {
            failed: false,
            is_static: false,
            kinematic_vel_enabled: false,
            kinematic_vel: Vector::zeros(),
            model_index: 0,
        }
    }
}

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Zeroable)]
#[repr(C)]
pub struct ParticlePhase {
    pub phase: Real,
    pub psi_pos: Real,
}

impl Default for ParticlePhase {
    fn default() -> Self {
        Self {
            phase: 1.0,
            psi_pos: 0.0,
        }
    }
}

/*
 *=======================
 *                      *
 * OPTIONAL COMPONENTS  *
 *                      *
 *=======================
 */

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ParticleContact {
    pub boundary_normal: Vector<Real>,
    pub boundary_dist: Real, // Negative if penetration.
}
unsafe impl RealStruct for ParticleContact {}
impl Default for ParticleContact {
    fn default() -> Self {
        Self {
            boundary_normal: Vector::zeros(),
            boundary_dist: 0.0,
        }
    }
}

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ParticleFracture {
    // Crack propagation
    pub crack_propagation_factor: Real,
    pub crack_threshold: Real,
}
unsafe impl RealStruct for ParticleFracture {}
impl Default for ParticleFracture {
    fn default() -> Self {
        Self {
            crack_propagation_factor: 0.0,
            crack_threshold: Real::MAX,
        }
    }
}

#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ParticleData {
    // Plasticity and hardening.
    pub nacc_alpha: Real,
    pub plastic_hardening: Real,
    pub elastic_hardening: Real,
    pub log_vol_gain: Real,
}
unsafe impl RealStruct for ParticleData {}

impl Default for ParticleData {
    fn default() -> Self {
        Self {
            nacc_alpha: -0.01,
            plastic_hardening: 1.0,
            elastic_hardening: 1.0,
            log_vol_gain: 0.0,
        }
    }
}
