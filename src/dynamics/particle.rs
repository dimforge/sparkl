use crate::dynamics::{GridNode, ParticleModelHandle};
use crate::geometry::SpGrid;
use crate::math::{Matrix, Point, Real, Vector, DIM};

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Particle {
    pub model: ParticleModelHandle,

    pub mass: Real,
    pub volume0: Real,
    pub radius0: Real,

    pub position: Point<Real>,
    pub velocity: Vector<Real>,
    pub kinematic_vel: Option<Vector<Real>>,
    pub velocity_gradient: Matrix<Real>, // MPM-MLS: equivalent to APIC affine matrix.
    pub deformation_gradient: Matrix<Real>, // Elastic part of the deformation gradient.
    pub plastic_deformation_gradient_det: Real, // Determinant of the plastic part of the deformation gradient.

    pub grid_index: u64,
    pub failed: bool,
    pub is_static: bool,

    // Crack propagation
    pub psi_pos: Real,
    pub parameter1: Real,
    pub parameter2: Real,
    pub crack_propagation_factor: Real,
    pub crack_threshold: Real,

    // Boundary handling
    pub boundary_normal: Vector<Real>,
    pub boundary_dist: Real, // Negative if penetration.

    // Phase field variables.
    pub m_c: Real,
    pub g: Real,
    pub phase: Real,
    pub phase_buf: Vector<Real>,

    // Plasticity and hardening.
    pub nacc_alpha: Real,
    pub plastic_hardening: Real,
    pub elastic_hardening: Real,
    pub log_vol_gain: Real,

    // User-data
    pub user_data: u64,
    pub debug_val: Real,
}

impl Particle {
    pub fn new(
        model: ParticleModelHandle,
        position: Point<Real>,
        radius: Real,
        density0: Real,
    ) -> Self {
        Self::with_internal_energy(model, position, radius, density0, 0.0)
    }

    pub fn with_internal_energy(
        model: ParticleModelHandle,
        position: Point<Real>,
        radius: Real,
        density0: Real,
        internal_energy_per_volume_unit: Real,
    ) -> Self {
        let volume0 = (radius * 2.0).powi(DIM as i32);

        Self {
            model,
            kinematic_vel: None,
            position,
            mass: volume0 * density0,
            volume0,
            radius0: radius,
            velocity: Vector::zeros(),
            velocity_gradient: Matrix::zeros(),
            deformation_gradient: Matrix::identity(),
            plastic_deformation_gradient_det: 1.0,
            failed: false,
            is_static: false,
            grid_index: 0,
            psi_pos: 0.0,
            parameter1: 0.0,
            parameter2: 0.0,
            crack_propagation_factor: 0.0,
            crack_threshold: Real::MAX,
            boundary_normal: Vector::zeros(),
            boundary_dist: 0.0,
            g: 0.0,
            m_c: Real::MAX,
            phase: 1.0,
            nacc_alpha: -0.01,
            phase_buf: na::zero(),
            plastic_hardening: 1.0,
            elastic_hardening: 1.0,
            log_vol_gain: 0.0,
            user_data: 0,
            debug_val: 0.0,
        }
    }

    pub fn closest_grid_pos(&self, cell_width: Real) -> Point<Real> {
        (self.position / cell_width).map(|e| e.round()) * cell_width
    }

    pub fn associated_grid_pos(&self, cell_width: Real) -> Point<Real> {
        ((self.position / cell_width).map(|e| e.round()) - Vector::repeat(1.0)) * cell_width
    }

    pub fn dir_to_closest_grid_node(&self, cell_width: Real) -> Vector<Real> {
        self.closest_grid_pos(cell_width) - self.position
    }

    pub fn dir_to_associated_grid_node(&self, cell_width: Real) -> Vector<Real> {
        self.associated_grid_pos(cell_width) - self.position
    }

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

    pub fn region_index(&self) -> u64 {
        self.grid_index & SpGrid::<GridNode>::REGION_ID_MASK
    }
}
