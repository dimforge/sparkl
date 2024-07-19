use crate::core::prelude::{ActiveTimestepBounds, CoreConstitutiveModel};
use crate::dynamics::Particle;
use crate::math::{Matrix, Real};

pub trait ConstitutiveModel: Send + Sync {
    fn is_fluid(&self) -> bool;

    fn elastic_energy_density(&self, _particle: &Particle) -> Real {
        0.0
    }

    fn pos_energy(&self, _particle: &Particle) -> Real {
        0.0
    }
    fn update_particle_stress(&self, particle: &Particle) -> Matrix<Real>;
    fn update_internal_energy_and_pressure(
        &self,
        _particle: &mut Particle,
        _dt: Real,
        _cell_width: Real,
    ) {
    }
    fn active_timestep_bounds(&self) -> ActiveTimestepBounds;
    fn timestep_bound(&self, particle: &Particle, cell_width: Real) -> Real;
    fn to_core_model(&self) -> Option<CoreConstitutiveModel>;
}
