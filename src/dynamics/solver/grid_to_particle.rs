use super::MlsSolver;
use crate::dynamics::solver::{DamageModel, RigidWorld};
use crate::dynamics::{GridNode, ParticleModelSet, ParticleSet};
use crate::geometry::SpGrid;
use crate::math::{Kernel, Matrix, Real, Vector};
use rayon::prelude::*;
use std::sync::atomic::Ordering;

impl MlsSolver {
    pub(crate) fn grid_to_particle(
        damage_model: DamageModel,
        enable_boundary_particle_projection: bool,
        dt: Real,
        rigid_world: &RigidWorld,
        grid: &SpGrid<GridNode>,
        particles: &mut ParticleSet,
        models: &ParticleModelSet,
    ) {
        let cell_width = grid.cell_width();
        let inv_d = Kernel::inv_d(cell_width);

        let particles_ptr = std::sync::atomic::AtomicPtr::new(particles as *mut _);

        particles.particle_bins.par_iter().for_each(|particle_bin| {
            let particles: &mut ParticleSet =
                unsafe { std::mem::transmute(particles_ptr.load(Ordering::Relaxed)) };

            for ii in 0..4 {
                if particle_bin[ii] == usize::MAX {
                    break;
                }

                let particle = &mut particles.particles[particle_bin[ii]];
                let model = &models[particle.model];
                let is_fluid = model.constitutive_model.is_fluid();
                let ref_elt_pos_minus_particle_pos =
                    particle.dir_to_associated_grid_node(cell_width);

                // APIC grid-to-particle transfer.
                let mut velocity = Vector::zeros();
                let mut velocity_gradient = Matrix::zeros();
                let mut velocity_gradient_det = 0.0;
                let mut psi_pos_momentum = 0.0;

                let w = Kernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);

                grid.for_each_neighbor_packed(particle.grid_index, |_cell_id, shift, cell| {
                    let dpt = ref_elt_pos_minus_particle_pos + shift.cast::<Real>() * cell_width;
                    #[cfg(feature = "dim2")]
                    let weight = w[0][shift.x] * w[1][shift.y];
                    #[cfg(feature = "dim3")]
                    let weight = w[0][shift.x] * w[1][shift.y] * w[2][shift.z];
                    let cell_velocity = cell.velocity;

                    velocity += weight * cell_velocity;
                    velocity_gradient += (weight * inv_d) * cell_velocity * dpt.transpose();
                    psi_pos_momentum +=
                        weight * cell.psi_momentum * crate::utils::inv_exact(cell.psi_mass);
                    velocity_gradient_det += weight * cell_velocity.dot(&dpt) * inv_d;
                });

                particle.velocity = velocity;
                particle.velocity_gradient = velocity_gradient;

                /*
                 * Modified Eigenerosion.
                 */
                if damage_model == DamageModel::ModifiedEigenerosion
                    && particle.crack_propagation_factor != 0.0
                    && particle.phase > 0.0
                {
                    let crack_energy =
                        particle.crack_propagation_factor * cell_width * psi_pos_momentum;
                    if crack_energy > particle.crack_threshold {
                        particle.phase = 0.0;
                    }
                }

                /*
                 * Advection.
                 */
                if let Some(kinematic_vel) = particle.kinematic_vel {
                    particle.velocity = kinematic_vel;
                }

                particle.position += particle.velocity * dt;

                /*
                 * Deformation gradient update.
                 */
                if !is_fluid {
                    particle.deformation_gradient +=
                        (particle.velocity_gradient * dt) * particle.deformation_gradient;
                } else {
                    particle.deformation_gradient[(0, 0)] +=
                        (velocity_gradient_det * dt) * particle.deformation_gradient[(0, 0)];
                    model
                        .constitutive_model
                        .update_internal_energy_and_pressure(particle, dt, cell_width);
                }

                if let Some(plasticity) = &model.plastic_model {
                    plasticity.update_particle(particle);
                }

                if particle.is_static {
                    particle.velocity.fill(0.0);
                    particle.velocity_gradient.fill(0.0);
                }

                if particle.density_def_grad() == 0.0
                    || particle.failed
                    // Isolated particles tend to accumulate a huge amount of numerical
                    // error, leading to completely broken deformation gradients.
                    // Don’t let them destroy the whole simulation by setting them as failed.
                    || (!is_fluid && particle.deformation_gradient[(0, 0)].abs() > 1.0e4)
                {
                    particle.failed = true;
                    particle.deformation_gradient = Matrix::identity();
                    particle.velocity_gradient = Matrix::zeros();
                }

                /*
                 * Update Pos energy.
                 * FIXME: refactor to its own function.
                 */
                {
                    let energy = model.constitutive_model.pos_energy(particle);
                    particle.psi_pos = particle.psi_pos.max(energy);
                    particle.parameter1 = particle.psi_pos * particle.mass;
                    particle.parameter2 = particle.mass;
                }

                /*
                 * Apply failure model.
                 * FIXME: refactor to its own function.
                 */
                {
                    if let Some(failure_model) = &model.failure_model {
                        if failure_model.particle_failed(particle, model) {
                            particle.phase = 0.0;
                        }
                    }
                }

                /*
                 * Particle projection.
                 * FIXME: refactor to its own function.
                 */
                if enable_boundary_particle_projection {
                    for (_, collider) in rigid_world.colliders.iter() {
                        let proj = collider.shape().project_point(
                            collider.position(),
                            &particle.position,
                            false,
                        );

                        if proj.is_inside {
                            particle.velocity += (proj.point - particle.position) / dt;
                            particle.position = proj.point;
                        }
                    }
                }
            }
        });
    }
}
