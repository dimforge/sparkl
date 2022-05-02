use crate::cuda::AtomicAdd;
use crate::gpu_grid::GpuGrid;
use crate::{GpuParticleModel, ParticleModelIndex};
use cuda_std::thread;
use cuda_std::*;
use sparkl_core::dynamics::{ParticleData, ParticlePosVelPhase};
use sparkl_core::math::{Kernel, DIM};

#[kernel]
pub unsafe fn recompute_fluids_volume_p2g(
    particles: *const ParticleData,
    particles_pos_vel: *const ParticlePosVelPhase,
    num_particles: usize,
    mut grid: GpuGrid,
) {
    let i = thread::index();

    if i < num_particles as u32 {
        let particle_i = &*particles.add(i as usize);
        let particle_pos_vel_i = &*particles_pos_vel.add(i as usize);

        scatter_particle(particle_i, particle_pos_vel_i, &mut grid);
    }
}

fn scatter_particle(
    particle: &ParticleData,
    particle_pos_vel: &ParticlePosVelPhase,
    grid: &mut GpuGrid,
) {
    /*
     * Scatter-style P2G.
     */
    let cell_width = grid.cell_width();
    let ref_elt_pos_minus_particle_pos = particle_pos_vel.dir_to_associated_grid_node(cell_width);
    let w = Kernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);
    let grid_index = grid.cell_associated_to_point(&particle_pos_vel.position);

    unsafe {
        grid.for_each_neighbor_packed_next_mut(grid_index, |_cell_id, shift, cell| {
            #[cfg(feature = "dim2")]
            let weight = w[0][shift.x] * w[1][shift.y];
            #[cfg(feature = "dim3")]
            let weight = w[0][shift.x] * w[1][shift.y] * w[2][shift.z];

            cell.mass.global_red_add(weight * particle.mass);
        });
    }
}

#[kernel]
pub unsafe fn recompute_fluids_volume_g2p(
    particles: *mut ParticleData,
    particles_pos_vel: *mut ParticlePosVelPhase,
    particles_model_id: *mut ParticleModelIndex,
    num_particles: usize,
    models: *mut GpuParticleModel,
    mut grid: GpuGrid,
) {
    let i = thread::index();

    if i < num_particles as u32 {
        let particle_i = &mut *particles.add(i as usize);
        let particle_pos_vel_i = &mut *particles_pos_vel.add(i as usize);
        let particle_model_id = &*particles_model_id.add(i as usize);
        let model_i = &*models.add(particle_model_id.0);

        gather_particle(particle_i, particle_pos_vel_i, model_i, &mut grid);
    }
}

fn gather_particle(
    particle: &mut ParticleData,
    particle_pos_vel: &mut ParticlePosVelPhase,
    model: &GpuParticleModel,
    grid: &mut GpuGrid,
) {
    if !model.constitutive_model.is_fluid() {
        return;
    }

    let cell_width = grid.cell_width();
    let ref_elt_pos_minus_particle_pos = particle_pos_vel.dir_to_associated_grid_node(cell_width);

    // APIC grid-to-particle transfer.
    let mut new_mass = 0.0;

    let w = Kernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);
    let grid_index = grid.cell_associated_to_point(&particle_pos_vel.position);

    unsafe {
        grid.for_each_neighbor_packed_curr(grid_index, |_cell_id, shift, cell| {
            #[cfg(feature = "dim2")]
            let weight = w[0][shift.x] * w[1][shift.y];
            #[cfg(feature = "dim3")]
            let weight = w[0][shift.x] * w[1][shift.y] * w[2][shift.z];

            new_mass += weight * cell.mass;
        });
    }

    let new_density = new_mass / cell_width.powi(DIM as i32);
    let new_volume = particle.mass / new_density;
    particle.deformation_gradient[(0, 0)] = new_volume / particle.volume0;
}
