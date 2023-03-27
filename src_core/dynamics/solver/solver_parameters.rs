use crate::math::{Real, Vector};

bitflags::bitflags! {
    #[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
    #[repr(C)]
    pub struct SimulationDofs: u32 {
        const LOCK_NONE = 0;
        const LOCK_X = 1 << 0;
        const LOCK_Y = 1 << 1;
        #[cfg(feature = "dim3")]
        const LOCK_Z = 1 << 2;
    }
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub enum BoundaryHandling {
    Stick,
    Friction,
    FrictionZUp, // A bit of a hack until we have a more generic solution
    None,
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub enum BoundaryCondition {
    Stick,
    Slip,
    Friction(Real),
}

impl BoundaryCondition {
    pub fn project(&self, velocity: Vector<Real>, normal: Vector<Real>) -> Vector<Real> {
        let normal_velocity = velocity.dot(&normal);
        let tangential_velocity = velocity - normal_velocity * normal;

        match self {
            Self::Stick => na::zero(),
            Self::Slip => tangential_velocity,
            Self::Friction(coefficient) => {
                if normal_velocity > 0.0 {
                    velocity
                } else {
                    let tangent = tangential_velocity
                        .try_normalize(1.0e-10)
                        .unwrap_or(na::zero());
                    (tangential_velocity.norm() + coefficient * normal_velocity).max(0.0) * tangent
                }
            }
        }
    }
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub enum DamageModel {
    CdMpm,
    Eigenerosion,
    ModifiedEigenerosion,
    None,
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "cuda", derive(cust_core::DeviceCopy))]
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct SolverParameters {
    pub dt: Real,
    pub max_substep_dt: Real,
    pub max_num_substeps: u32,
    pub boundary_handling: BoundaryHandling,
    pub damage_model: DamageModel,
    pub force_fluids_volume_recomputation: bool,
    pub enable_boundary_particle_projection: bool,
    pub stop_after_one_substep: bool,
    pub simulation_dofs: SimulationDofs,
}

impl Default for SolverParameters {
    fn default() -> Self {
        SolverParameters {
            dt: 1.0 / 60.0,
            max_substep_dt: Real::MAX,
            max_num_substeps: 1000,
            boundary_handling: BoundaryHandling::Friction,
            damage_model: DamageModel::None,
            force_fluids_volume_recomputation: false,
            enable_boundary_particle_projection: false,
            stop_after_one_substep: false,
            simulation_dofs: SimulationDofs::LOCK_NONE,
        }
    }
}
