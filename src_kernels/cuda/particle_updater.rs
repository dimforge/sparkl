use crate::GpuParticleModel;
use sparkl_core::math::{Matrix, Real, Vector};
use sparkl_core::prelude::{
    ActiveTimestepBounds, ParticlePhase, ParticlePosition, ParticleStatus, ParticleVelocity,
    ParticleVolume,
};

#[cfg(target_os = "cuda")]
use na::ComplexField;

pub trait ParticleUpdater {
    unsafe fn estimate_particle_timestep_length(
        &self,
        cell_width: Real,
        particle_id: u32,
        particle_status: &ParticleStatus,
        particle_volume: &ParticleVolume,
        particle_vel: &ParticleVelocity,
    ) -> (ActiveTimestepBounds, Real);

    unsafe fn update_particle_and_compute_kirchhoff_stress(
        &self,
        dt: Real,
        cell_width: Real,
        particle_id: u32,
        particle_status: &mut ParticleStatus,
        particle_pos: &mut ParticlePosition,
        particle_vel: &mut ParticleVelocity,
        particle_volume: &mut ParticleVolume,
        particle_phase: &mut ParticlePhase,
        velocity_gradient: &mut Matrix<Real>,
        velocity: Vector<Real>,
        velocity_gradient_det: Real,
        psi_pos: Real,
    ) -> Option<Matrix<Real>>;
}

pub struct DefaultParticleUpdater {
    pub models: *mut GpuParticleModel,
}

impl ParticleUpdater for DefaultParticleUpdater {
    #[inline(always)]
    unsafe fn estimate_particle_timestep_length(
        &self,
        cell_width: Real,
        particle_id: u32,
        particle_status: &ParticleStatus,
        particle_volume: &ParticleVolume,
        particle_vel: &ParticleVelocity,
    ) -> (ActiveTimestepBounds, Real) {
        let model = &*self.models.add(particle_status.model_index);
        let active_timestep_bounds = model.constitutive_model.active_timestep_bounds();

        if active_timestep_bounds.contains(ActiveTimestepBounds::CONSTITUTIVE_MODEL_BOUND) {
            (
                active_timestep_bounds,
                model.constitutive_model.timestep_bound(
                    particle_id,
                    particle_volume,
                    particle_vel,
                    cell_width,
                ),
            )
        } else {
            (active_timestep_bounds, Real::MAX)
        }
    }

    #[inline(always)]
    unsafe fn update_particle_and_compute_kirchhoff_stress(
        &self,
        dt: Real,
        cell_width: Real,
        particle_id: u32,
        particle_status: &mut ParticleStatus,
        particle_pos: &mut ParticlePosition,
        particle_vel: &mut ParticleVelocity,
        particle_volume: &mut ParticleVolume,
        particle_phase: &mut ParticlePhase,
        velocity_gradient: &mut Matrix<Real>,
        velocity: Vector<Real>,
        velocity_gradient_det: Real,
        _psi_pos: Real,
    ) -> Option<Matrix<Real>> {
        let model = &*self.models.add(particle_status.model_index);

        particle_vel.vector = velocity;

        let is_fluid = model.constitutive_model.is_fluid();

        /*
         * Modified Eigenerosion.
         */
        // TODO:???move this to a new failure model?
        // if damage_model == DamageModel::ModifiedEigenerosion
        //     && particle.crack_propagation_factor != 0.0
        //     && particle_status.phase > 0.0
        // {
        //     let crack_energy = particle.crack_propagation_factor * cell_width * psi_pos_momentum;
        //     if crack_energy > particle.crack_threshold {
        //         particle_status.phase = 0.0;
        //     }
        // }

        /*
         * Advection.
         */
        if particle_status.kinematic_vel_enabled {
            particle_vel.vector = particle_status.kinematic_vel;
        }

        if particle_vel
            .vector
            .iter()
            .any(|x| x.abs() * dt >= cell_width)
        {
            particle_vel
                .vector
                .apply(|x| *x = x.signum() * cell_width / dt);
        }

        particle_pos.point += particle_vel.vector * dt;

        /*
         * Deformation gradient update.
         */
        if !is_fluid {
            particle_volume.deformation_gradient +=
                (*velocity_gradient * dt) * particle_volume.deformation_gradient;
        } else {
            particle_volume.deformation_gradient[(0, 0)] +=
                (velocity_gradient_det * dt) * particle_volume.deformation_gradient[(0, 0)];
            model
                .constitutive_model
                .update_internal_energy_and_pressure(
                    particle_id,
                    particle_volume,
                    dt,
                    cell_width,
                    &velocity_gradient,
                );
        }

        if let Some(plasticity) = &model.plastic_model {
            plasticity.update_particle(particle_id, particle_volume, particle_status.phase);
        }

        if particle_status.is_static {
            particle_vel.vector.fill(0.0);
            velocity_gradient.fill(0.0);
        }

        if particle_volume.density_def_grad() == 0.0
        || particle_status.failed
        // Isolated particles tend to accumulate a huge amount of numerical
        // error, leading to completely broken deformation gradients.
        // Don???t let them destroy the whole simulation by setting them as failed.
        || (!is_fluid && particle_volume.deformation_gradient[(0, 0)].abs() > 1.0e4)
        {
            particle_status.failed = true;
            particle_volume.deformation_gradient = Matrix::identity();
            return None;
        }

        /*
         * Update Pos energy.
         * TODO: refactor to its own function.
         * TODO:???should the crack propagation be part of its own kernel?
         */
        {
            let energy = model.constitutive_model.pos_energy(
                particle_id,
                particle_volume,
                particle_status.phase,
            );
            particle_phase.psi_pos = particle_phase.psi_pos.max(energy);
        }

        /*
         * Apply failure model.
         * FIXME: refactor to its own function.
         */
        {
            if let Some(failure_model) = &model.failure_model {
                let stress = model.constitutive_model.kirchhoff_stress(
                    particle_id,
                    particle_volume,
                    particle_status.phase,
                    &velocity_gradient,
                );
                if failure_model.particle_failed(&stress) {
                    particle_status.phase = 0.0;
                }
            }
        }

        // /*
        //  * Particle projection.
        //  * TODO: refactor to its own function.
        //  */
        // if enable_boundary_particle_projection {
        //     for (_, collider) in rigid_world.colliders.iter() {
        //         let proj =
        //             collider
        //                 .shape()
        //                 .project_point(collider.position(), &particle_pos_vel.position, false);
        //
        //         if proj.is_inside {
        //             particle.velocity += (proj.point - particle_pos_vel.position) / dt;
        //             particle_pos_vel.position = proj.point;
        //         }
        //     }
        // }

        // MPM-MLS: the APIC affine matrix and the velocity gradient are the same.
        if !particle_status.failed {
            Some(model.constitutive_model.kirchhoff_stress(
                particle_id,
                particle_volume,
                particle_status.phase,
                &velocity_gradient,
            ))
        } else {
            Some(Matrix::zeros())
        }
    }
}
