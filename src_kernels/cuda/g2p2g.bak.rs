use crate::cuda::atomic::{AtomicAdd, AtomicInt};
use crate::gpu_grid::GpuGrid;
use crate::{BlockVirtualId, GpuParticleModel, NBH_SHIFTS, NBH_SHIFTS_SHARED, NUM_CELL_PER_BLOCK};
use cuda_std::thread;
use cuda_std::*;
use nalgebra::vector;
use parry::math::DIM;
use sparkl_core::math::{Kernel, Matrix, Real, Vector};
use sparkl_core::prelude::{
    DamageModel, ParticleFracture, ParticlePhase, ParticlePosition, ParticleStatus,
    ParticleVelocity, ParticleVolume,
};

#[cfg(feature = "dim2")]
const NUM_SHARED_CELLS: usize = (4 * 4) * (2 * 2);
#[cfg(feature = "dim3")]
const NUM_SHARED_CELLS: usize = (4 * 4 * 4) * (2 * 2 * 2);
const FREE: u32 = u32::MAX;

struct GridGatherData {
    mass: Real,
    momentum: Vector<Real>,
    velocity: Vector<Real>,
    psi_mass: Real,
    psi_momentum: Real,
    psi_velocity: Real,
    // NOTE: right now we are using a manually implemented lock, based on
    // integer atomics exchange, to avoid float atomics on shared memory
    // which are super slow.
    // If we are to remove this lock at some point in the future, it may be
    // worth just renaming it to `_padding: u32` instead of just removing it.
    // This extra padding seems to improve performance (we should investigate
    // why, perhaps a bank conflict thing).
    lock: u32,
}

// This MUST be called with a block size equal to G2P2G_THREADS
#[kernel]
pub unsafe fn g2p2g(
    dt: Real,
    particles_status: *mut ParticleStatus,
    particles_pos: *mut ParticlePosition,
    particles_vel: *mut ParticleVelocity,
    particles_volume: *mut ParticleVolume,
    particles_phase: *mut ParticlePhase,
    sorted_particle_ids: *const u32,
    models: *mut GpuParticleModel,
    curr_grid: GpuGrid,
    mut next_grid: GpuGrid,
    damage_model: DamageModel,
    halo: bool,
) {
    let shared_nodes = shared_array![GridGatherData; NUM_SHARED_CELLS];

    let bid = thread::block_idx_x();
    let tid = thread::thread_idx_x();

    let dispatch2active = if halo {
        next_grid.dispatch_halo_block_to_active_block
    } else {
        next_grid.dispatch_block_to_active_block
    };

    let dispatch_block_to_active_block = *dispatch2active.as_ptr().add(bid as usize);
    let active_block =
        *next_grid.active_block_unchecked(dispatch_block_to_active_block.active_block_id);

    transfer_global_blocks_to_shared_memory(shared_nodes, &curr_grid, active_block.virtual_id);

    // Sync after shared memory initialization.
    thread::sync_threads();

    if dispatch_block_to_active_block.first_particle + tid
        < active_block.first_particle + active_block.num_particles
    {
        let particle_id = *sorted_particle_ids
            .add((dispatch_block_to_active_block.first_particle + tid) as usize);
        let mut particle_status_i = *particles_status.add(particle_id as usize);
        let mut particle_pos_i = *particles_pos.add(particle_id as usize);
        let mut particle_vel_i = *particles_vel.add(particle_id as usize);
        let mut particle_volume_i = *particles_volume.add(particle_id as usize);
        let mut particle_phase_i = *particles_phase.add(particle_id as usize);
        let model_i = *models.add(particle_status_i.model_index);

        particle_g2p2g(
            dt,
            particle_id,
            &mut particle_status_i,
            &mut particle_pos_i,
            &mut particle_vel_i,
            &mut particle_volume_i,
            &mut particle_phase_i,
            &model_i,
            shared_nodes,
            next_grid.cell_width(),
            damage_model,
        );

        *particles_status.add(particle_id as usize) = particle_status_i;
        *particles_pos.add(particle_id as usize) = particle_pos_i;
        *particles_vel.add(particle_id as usize) = particle_vel_i;
        *particles_volume.add(particle_id as usize) = particle_volume_i;
        *particles_phase.add(particle_id as usize) = particle_phase_i;
    }

    // Sync before writeback.
    thread::sync_threads();
    transfer_shared_blocks_to_grid(shared_nodes, &mut next_grid, active_block.virtual_id);
}

unsafe fn particle_g2p2g(
    dt: Real,
    particle_id: u32,
    particle_status: &mut ParticleStatus,
    particle_pos: &mut ParticlePosition,
    particle_vel: &mut ParticleVelocity,
    particle_volume: &mut ParticleVolume,
    particle_phase: &mut ParticlePhase,
    model: &GpuParticleModel,
    shared_nodes: *mut GridGatherData,
    cell_width: Real,
    damage_model: DamageModel,
) {
    let tid = thread::thread_idx_x();
    let inv_d = Kernel::inv_d(cell_width);

    let is_fluid = model.constitutive_model.is_fluid();
    let ref_elt_pos_minus_particle_pos = particle_pos.dir_to_associated_grid_node(cell_width);

    // APIC grid-to-particle transfer.
    let mut velocity = Vector::zeros();
    let mut velocity_gradient = Matrix::zeros();
    let mut velocity_gradient_det = 0.0;
    let mut psi_pos_momentum = 0.0;

    let w = Kernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);

    let assoc_cell_before_integration = particle_pos.position.map(|e| (e / cell_width).round());
    let assoc_cell_index_in_block =
        particle_pos.associated_cell_index_in_block_off_by_two(cell_width);
    #[cfg(feature = "dim2")]
    let packed_cell_index_in_block =
        (assoc_cell_index_in_block.x + 1) + (assoc_cell_index_in_block.y + 1) * 8;
    #[cfg(feature = "dim3")]
    let packed_cell_index_in_block = (assoc_cell_index_in_block.x + 1)
        + (assoc_cell_index_in_block.y + 1) * 8
        + (assoc_cell_index_in_block.z + 1) * 8 * 8;

    for (shift, packed_shift) in NBH_SHIFTS.iter().zip(NBH_SHIFTS_SHARED.iter()) {
        let dpt = ref_elt_pos_minus_particle_pos + shift.cast::<Real>() * cell_width;
        #[cfg(feature = "dim2")]
        let weight = w[0][shift.x] * w[1][shift.y];
        #[cfg(feature = "dim3")]
        let weight = w[0][shift.x] * w[1][shift.y] * w[2][shift.z];

        let cell = &*shared_nodes.add(packed_cell_index_in_block as usize + packed_shift);
        velocity += weight * cell.velocity;
        velocity_gradient += (weight * inv_d) * cell.velocity * dpt.transpose();
        psi_pos_momentum += weight * cell.psi_velocity;
        velocity_gradient_det += weight * cell.velocity.dot(&dpt) * inv_d;
    }

    particle_vel.velocity = velocity;

    /*
     * Modified Eigenerosion.
     */
    if damage_model == DamageModel::ModifiedEigenerosion
        && particle.crack_propagation_factor != 0.0
        && particle_status.phase > 0.0
    {
        let crack_energy = particle.crack_propagation_factor * cell_width * psi_pos_momentum;
        if crack_energy > particle.crack_threshold {
            particle_status.phase = 0.0;
        }
    }

    /*
     * Advection.
     */
    if particle_status.kinematic_vel_enabled {
        particle_vel.velocity = particle_status.kinematic_vel;
    }

    if particle_vel
        .velocity
        .iter()
        .any(|x| x.abs() * dt >= cell_width)
    {
        particle_vel
            .velocity
            .apply(|x| *x = x.signum() * cell_width / dt);
    }

    particle_pos.position += particle_vel.velocity * dt;

    /*
     * Deformation gradient update.
     */
    if !is_fluid {
        particle_volume.deformation_gradient +=
            (velocity_gradient * dt) * particle_volume.deformation_gradient;
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

    /*
     * Evolve the particle volume increase.
     */
    if particle_status.phase == 0.0 {
        let swell_def_velocity_factor =
            (-(0.5 * (velocity_gradient + velocity_gradient.transpose())).determinant()
                * particle.volume_increase_deformation_multiplier)
                .clamp(0.0, 1.0);
        let swell_spin_velocity_factor = ((0.5
            * (velocity_gradient - velocity_gradient.transpose()))
        .determinant()
        .abs()
            * particle.volume_increase_spin_multiplier)
            .clamp(0.0, 1.0);
        let swell_mov_velocity_factor = (particle_vel.velocity.xy().norm()
            * particle.volume_increase_movement_multiplier)
            .clamp(0.0, 1.0);
        let swell_velocity_factor =
            swell_def_velocity_factor + swell_spin_velocity_factor + swell_mov_velocity_factor;
        let new_volume_increase = (particle.volume_increase
            + particle.volume_increase_velocity * swell_velocity_factor * dt)
            .clamp(
                particle.volume_increase_bounds[0],
                particle.volume_increase_bounds[1],
            );

        let delta_volume = new_volume_increase - particle.volume_increase;
        let j_factor = particle_volume.volume0 / (particle_volume.volume0 + delta_volume);
        particle_volume.volume0 += delta_volume;
        particle_volume.deformation_gradient *= j_factor.powf(1.0 / (DIM as Real));
        particle.volume_increase = new_volume_increase;

        // let volume_increase_range =
        //     (particle.volume_increase_bounds[1] - particle.volume_increase_bounds[0]) / 2.0;
        // particle_pos_vel.debug_val = new_volume_increase / volume_increase_range * 10.0;
    }

    if let Some(plasticity) = &model.plastic_model {
        plasticity.update_particle(particle_id, particle_volume, particle_status.phase);
    }

    if particle_status.is_static {
        particle_vel.velocity.fill(0.0);
        velocity_gradient.fill(0.0);
    }

    if particle.internal_energy.is_nan()
        || particle_volume.density_def_grad() == 0.0
        || particle_status.failed
        // Isolated particles tend to accumulate a huge amount of numerical
        // error, leading to completely broken deformation gradients.
        // Don’t let them destroy the whole simulation by setting them as failed.
        || (!is_fluid && particle_volume.deformation_gradient[(0, 0)].abs() > 1.0e4)
    {
        particle_status.failed = true;
        particle.internal_energy = 0.0;
        particle.pressure = 0.0;
        particle_volume.deformation_gradient = Matrix::identity();
        return;
    }

    /*
     * Update Pos energy.
     * FIXME: refactor to its own function.
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

    /*
     * Scatter-style P2G.
     */
    let inv_d = Kernel::inv_d(cell_width);
    let ref_elt_pos_minus_particle_pos = particle_pos.dir_to_associated_grid_node(cell_width);
    let w = Kernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);

    // MPM-MLS: the APIC affine matrix and the velocity gradient are the same.
    let stress = if !particle_status.failed {
        model.constitutive_model.kirchhoff_stress(
            particle_id,
            particle_volume,
            particle_status.phase,
            &velocity_gradient,
        )
    } else {
        Matrix::zeros()
    };

    let affine =
        particle_volume.mass * velocity_gradient - (particle_volume.volume0 * inv_d * dt) * stress;
    let momentum = particle_volume.mass * particle_vel.velocity;

    let psi_mass = if particle_status.phase > 0.0
        // TODO: this test was removed from the original g2p2g kernel.
        //       Not having this check means that the mass of non-breakable
        //       particles don’t affect the breakability of surrounding particles.
        && particle.crack_propagation_factor != 0.0
        && !particle_status.failed
    {
        particle_volume.mass
    } else {
        0.0
    };

    let psi_pos_momentum = psi_mass * particle_phase.psi_pos;

    let assoc_cell_after_integration = particle_pos.position.map(|e| (e / cell_width).round());
    let particle_cell_movement =
        (assoc_cell_after_integration - assoc_cell_before_integration).map(|e| e as i64);

    #[cfg(feature = "dim2")]
    let packed_cell_index_in_block = (packed_cell_index_in_block as i64
        + (particle_cell_movement.x)
        + (particle_cell_movement.y) * 8) as u32;
    #[cfg(feature = "dim3")]
    let packed_cell_index_in_block = (packed_cell_index_in_block as i64
        + (particle_cell_movement.x)
        + (particle_cell_movement.y) * 8
        + (particle_cell_movement.z) * 8 * 8) as u32;

    for (shift, packed_shift) in NBH_SHIFTS.iter().zip(NBH_SHIFTS_SHARED.iter()) {
        let dpt = ref_elt_pos_minus_particle_pos + shift.cast::<Real>() * cell_width;
        #[cfg(feature = "dim2")]
        let weight = w[0][shift.x] * w[1][shift.y];
        #[cfg(feature = "dim3")]
        let weight = w[0][shift.x] * w[1][shift.y] * w[2][shift.z];

        let added_mass = weight * particle_volume.mass;
        let added_momentum = weight * (affine * dpt + momentum);
        let added_psi_momentum = weight * psi_pos_momentum;
        let added_psi_mass = weight * psi_mass;

        // NOTE: float atomics on shared memory are super slow because they are not
        // hardware accelerated yet. So we implement a manual lock instead, which seems
        // much faster in practice (because atomic exchange on integers are hardware-accelerated
        // on shared memory).
        let cell = &mut *shared_nodes.add(packed_cell_index_in_block as usize + packed_shift);

        loop {
            let old = cell.lock.shared_atomic_exch_acq(tid);
            if old == FREE {
                cell.mass += added_mass;
                cell.momentum += added_momentum;
                cell.psi_momentum += added_psi_momentum;
                cell.psi_mass += added_psi_mass;
                cell.lock.shared_atomic_exch_rel(FREE);
                break;
            }
        }

        /*
           // TODO: shared float atomics are much slower than global float atomics.
           cell.mass.shared_red_add(weight * particle.mass);
           cell.momentum
               .shared_red_add(weight * (affine * dpt + momentum));
           cell.psi_momentum.shared_red_add(weight * psi_pos_momentum);
           cell.psi_mass.shared_red_add(weight * psi_mass);
        */
    }
}

unsafe fn transfer_global_blocks_to_shared_memory(
    shared_nodes: *mut GridGatherData,
    curr_grid: &GpuGrid,
    active_block_vid: BlockVirtualId,
) {
    let tid = thread::thread_idx_x();
    let blk_sz = thread::block_dim_x();

    // There are less threads than the size of the allocated shared memory.
    // So a single thread has to initialize/writeback several cells between
    // shared and global memory.
    let num_cell_per_thread = NUM_SHARED_CELLS / blk_sz as usize;
    // The contiguous index, in the (4x4x4) * (2x2x2) shared memory blocks, of the first cell
    // managed by this thread for the shared <-> global memory transfers. From the point of view
    // of thread indexing, we give cells within the same (4x4x4) sub-block contiguous indices.
    let first_transfer_cell_id = tid as u64 * num_cell_per_thread as u64;
    // Identify block/octant that will be touched by this thread during the shared <-> global memory transfers.
    // Note that thanks to the kernel launch dimensions have to be arranged in such a way that all
    // the cells addressed by this thread and contiguous in memory and belong to the same octant.
    let octant = first_transfer_cell_id / NUM_CELL_PER_BLOCK;
    assert!(octant < 8);

    #[cfg(feature = "dim2")]
    let octant = vector![(octant & 0b0001) >> 0, (octant & 0b0010) >> 1];
    #[cfg(feature = "dim3")]
    let octant = vector![
        (octant & 0b0001) >> 0,
        (octant & 0b0010) >> 1,
        (octant & 0b0100) >> 2
    ];

    let base_block_pos_int = active_block_vid.unpack();
    let octant_hid = curr_grid.get_header_block_id(BlockVirtualId::pack(
        base_block_pos_int + octant.cast::<usize>(),
    ));

    // It’s interesting to observe that replacing blk_sz by G2P2G_THREADS actually
    // makes the simulation slower.
    // TODO: investigate why. Maybe because some unwanted automatic loop unrolling are happening?
    let num_cell_per_thread = NUM_SHARED_CELLS / blk_sz as usize;
    // This is the index of the first cell managed by this thread, inside of the (4x4x4) block
    // identified by the octant.
    let first_cell_in_octant = (tid as u64 * num_cell_per_thread as u64) % NUM_CELL_PER_BLOCK;

    if let Some(octant_hid) = octant_hid {
        for id in first_cell_in_octant..first_cell_in_octant + num_cell_per_thread as u64 {
            // Compute the (x,y,z) integer coordinates of the cell addressed by this thread,
            // within the (4x4x4) block.
            #[cfg(feature = "dim2")]
            let shift_in_octant = vector![id & 0b0011, (id >> 2) & 0b0011];
            #[cfg(feature = "dim3")]
            let shift_in_octant = vector![id & 0b0011, (id >> 2) & 0b0011, id >> 4];

            // Compute the corresponding cell in the shared memory (4x4x4) * (2x2x2) buffer.
            let shift_in_shared = octant * 4 + shift_in_octant;
            // Flatten the shared memory index.
            #[cfg(feature = "dim2")]
            let id_in_shared = shift_in_shared.x + shift_in_shared.y * 8;
            #[cfg(feature = "dim3")]
            let id_in_shared =
                shift_in_shared.x + shift_in_shared.y * 8 + shift_in_shared.z * 8 * 8;
            // Flatten the shared global_memory_index.
            let id_in_global = octant_hid
                .to_physical()
                .node_id_unchecked(shift_in_octant.cast::<usize>());

            let shared_node = &mut *shared_nodes.add(id_in_shared as usize);

            if let Some(global_node) = curr_grid.get_node(id_in_global) {
                shared_node.velocity = global_node.momentum_velocity;
                shared_node.psi_velocity = global_node.psi_momentum_velocity;
            } else {
                shared_node.velocity = na::zero();
                shared_node.psi_velocity = na::zero();
            }

            shared_node.psi_momentum = 0.0;
            shared_node.psi_mass = 0.0;
            shared_node.momentum.fill(0.0);
            shared_node.mass = 0.0;
            shared_node.lock = FREE;
        }
    } else {
        // This octant didn’t exist during the last simulation step.
        // TODO: what about gravity?
        for id in first_cell_in_octant..first_cell_in_octant + num_cell_per_thread as u64 {
            // Compute the (x,y,z) integer coordinates of the cell addressed by this thread,
            // within the (4x4x4) block.
            #[cfg(feature = "dim2")]
            let shift_in_octant = vector![id & 0b0011, (id >> 2) & 0b0011];
            #[cfg(feature = "dim3")]
            let shift_in_octant = vector![id & 0b0011, (id >> 2) & 0b0011, id >> 4];
            // Compute the corresponding cell in the shared memory (4x4x4) * (2x2x2) buffer.
            let shift_in_shared = octant * 4 + shift_in_octant;
            // Flatten the shared memory index.
            #[cfg(feature = "dim2")]
            let id_in_shared = shift_in_shared.x + shift_in_shared.y * 8;
            #[cfg(feature = "dim3")]
            let id_in_shared =
                shift_in_shared.x + shift_in_shared.y * 8 + shift_in_shared.z * 8 * 8;

            let shared_node = &mut *shared_nodes.add(id_in_shared as usize);

            shared_node.velocity = na::zero();
            shared_node.psi_velocity = na::zero();
            shared_node.psi_momentum = 0.0;
            shared_node.psi_mass = 0.0;
            shared_node.momentum.fill(0.0);
            shared_node.mass = 0.0;
            shared_node.lock = FREE;
        }
    }
}

unsafe fn transfer_shared_blocks_to_grid(
    shared_nodes: *const GridGatherData,
    next_grid: &mut GpuGrid,
    active_block_vid: BlockVirtualId,
) {
    let tid = thread::thread_idx_x();
    let blk_sz = thread::block_dim_x();

    // There are less threads than the size of the allocated shared memory.
    // So a single thread has to initialize/writeback several cells between
    // shared and global memory.
    let num_cell_per_thread = NUM_SHARED_CELLS / blk_sz as usize;
    // The contiguous index, in the (4x4x4) * (2x2x2) shared memory blocks, of the first cell
    // managed by this thread for the shared <-> global memory transfers. From the point of view
    // of thread indexing, we give cells within the same (4x4x4) sub-block contiguous indices.
    let first_transfer_cell_id = tid as u64 * num_cell_per_thread as u64;
    // Identify block/octant that will be touched by this thread during the shared <-> global memory transfers.
    // Note that thanks to the kernel launch dimensions have to be arranged in such a way that all
    // the cells addressed by this thread and contiguous in memory and belong to the same octant.
    let octant = first_transfer_cell_id / NUM_CELL_PER_BLOCK;
    assert!(octant < 8);
    #[cfg(feature = "dim2")]
    let octant = vector![(octant & 0b0001) >> 0, (octant & 0b0010) >> 1];
    #[cfg(feature = "dim3")]
    let octant = vector![
        (octant & 0b0001) >> 0,
        (octant & 0b0010) >> 1,
        (octant & 0b0100) >> 2
    ];

    let base_block_pos_int = active_block_vid.unpack();
    let octant_hid = next_grid
        .get_header_block_id(BlockVirtualId::pack(
            base_block_pos_int + octant.cast::<usize>(),
        ))
        .unwrap();

    // It’s interesting to observe that replacing blk_sz by G2P2G_THREADS actually
    // makes the simulation slower.
    // TODO: investigate why. Maybe because some unwanted automatic loop unrolling are happening?
    let num_cell_per_thread = NUM_SHARED_CELLS / blk_sz as usize;
    // This is the index of the first cell managed by this thread, inside of the (4x4x4) block
    // identified by the octant.
    let first_cell_in_octant = (tid as u64 * num_cell_per_thread as u64) % NUM_CELL_PER_BLOCK;

    for id in first_cell_in_octant..first_cell_in_octant + num_cell_per_thread as u64 {
        // Compute the (x,y,z) integer coordinates of the cell addressed by this thread,
        // within the (4x4x4) block.
        #[cfg(feature = "dim2")]
        let shift_in_octant = vector![id & 0b0011, (id >> 2) & 0b0011];
        #[cfg(feature = "dim3")]
        let shift_in_octant = vector![id & 0b0011, (id >> 2) & 0b0011, id >> 4];
        // Compute the corresponding cell in the shared memory (4x4x4) * (2x2x2) buffer.
        let shift_in_shared = octant * 4 + shift_in_octant;
        // Flatten the shared memory index.
        #[cfg(feature = "dim2")]
        let id_in_shared = shift_in_shared.x + shift_in_shared.y * 8;
        #[cfg(feature = "dim3")]
        let id_in_shared = shift_in_shared.x + shift_in_shared.y * 8 + shift_in_shared.z * 8 * 8;
        // Flatten the shared global_memory_index.
        let id_in_global = octant_hid
            .to_physical()
            .node_id_unchecked(shift_in_octant.cast::<usize>());

        let shared_node = &*shared_nodes.add(id_in_shared as usize);

        if let Some(global_node) = next_grid.get_node_mut(id_in_global) {
            global_node.mass.global_red_add(shared_node.mass);
            global_node
                .momentum_velocity
                .global_red_add(shared_node.momentum);
            global_node
                .psi_momentum_velocity
                .global_red_add(shared_node.psi_momentum);
            global_node.psi_mass.global_red_add(shared_node.psi_mass);
        }
    }
}
