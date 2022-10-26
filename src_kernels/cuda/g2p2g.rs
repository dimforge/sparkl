use crate::cuda::atomic::{AtomicAdd, AtomicInt};
use crate::cuda::{DefaultParticleUpdater, ParticleUpdater};
use crate::gpu_grid::{GpuGrid, GpuGridProjectionStatus};
use crate::{
    BlockVirtualId, GpuCollider, GpuColliderSet, GpuParticleModel, NBH_SHIFTS, NBH_SHIFTS_SHARED,
    NUM_CELL_PER_BLOCK,
};
use cuda_std::thread;
use cuda_std::*;
use nalgebra::vector;
use sparkl_core::math::{Kernel, Matrix, Real, Vector};
use sparkl_core::prelude::{
    DamageModel, ParticlePhase, ParticlePosition, ParticleStatus, ParticleVelocity, ParticleVolume,
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
    projection_scaled_dir: Vector<Real>,
    projection_status: GpuGridProjectionStatus,
    // NOTE: right now we are using a manually implemented lock, based on
    // integer atomics exchange, to avoid float atomics on shared memory
    // which are super slow.
    // If we are to remove this lock at some point in the future, it may be
    // worth just renaming it to `_padding: u32` instead of just removing it.
    // This extra padding seems to improve performance (we should investigate
    // why, perhaps a bank conflict thing).
    lock: u32,
}

pub struct InterpolatedParticleData {
    pub velocity: Vector<Real>,
    pub velocity_gradient: Matrix<Real>,
    pub psi_pos_momentum: Real,
    pub velocity_gradient_det: Real,
    pub projection_scaled_dir: Vector<Real>,
    pub projection_status: GpuGridProjectionStatus,
}

impl Default for InterpolatedParticleData {
    fn default() -> Self {
        Self {
            velocity: na::zero(),
            velocity_gradient: na::zero(),
            psi_pos_momentum: na::zero(),
            velocity_gradient_det: na::zero(),
            projection_scaled_dir: na::zero(),
            projection_status: GpuGridProjectionStatus::NotComputed,
        }
    }
}

#[kernel]
pub unsafe fn g2p2g(
    dt: Real,
    colliders_ptr: *const GpuCollider,
    num_colliders: usize,
    particles_status: *mut ParticleStatus,
    particles_pos: *mut ParticlePosition,
    particles_vel: *mut ParticleVelocity,
    particles_volume: *mut ParticleVolume,
    particles_phase: *mut ParticlePhase,
    sorted_particle_ids: *const u32,
    models: *mut GpuParticleModel,
    curr_grid: GpuGrid,
    next_grid: GpuGrid,
    damage_model: DamageModel,
    halo: bool,
) {
    g2p2g_generic(
        dt,
        colliders_ptr,
        num_colliders,
        particles_status,
        particles_pos,
        particles_vel,
        particles_volume,
        particles_phase,
        sorted_particle_ids,
        curr_grid,
        next_grid,
        damage_model,
        halo,
        DefaultParticleUpdater { models },
    )
}

// This MUST be called with a block size equal to G2P2G_THREADS
pub unsafe fn g2p2g_generic(
    dt: Real,
    colliders_ptr: *const GpuCollider,
    num_colliders: usize,
    particles_status: *mut ParticleStatus,
    particles_pos: *mut ParticlePosition,
    particles_vel: *mut ParticleVelocity,
    particles_volume: *mut ParticleVolume,
    particles_phase: *mut ParticlePhase,
    sorted_particle_ids: *const u32,
    curr_grid: GpuGrid,
    mut next_grid: GpuGrid,
    damage_model: DamageModel,
    halo: bool,
    particle_updater: impl ParticleUpdater,
) {
    let shared_nodes = shared_array![GridGatherData; NUM_SHARED_CELLS];

    let bid = thread::block_idx_x();
    let tid = thread::thread_idx_x();

    let dispatch2active = if halo {
        next_grid.dispatch_halo_block_to_active_block
    } else {
        next_grid.dispatch_block_to_active_block
    };

    let collider_set = GpuColliderSet {
        ptr: colliders_ptr,
        len: num_colliders,
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

        particle_g2p2g(
            dt,
            particle_id,
            &collider_set,
            &mut particle_status_i,
            &mut particle_pos_i,
            &mut particle_vel_i,
            &mut particle_volume_i,
            &mut particle_phase_i,
            shared_nodes,
            next_grid.cell_width(),
            damage_model,
            particle_updater,
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
    colliders: &GpuColliderSet,
    particle_status: &mut ParticleStatus,
    particle_pos: &mut ParticlePosition,
    particle_vel: &mut ParticleVelocity,
    particle_volume: &mut ParticleVolume,
    particle_phase: &mut ParticlePhase,
    shared_nodes: *mut GridGatherData,
    cell_width: Real,
    _damage_model: DamageModel,
    particle_updater: impl ParticleUpdater,
) {
    let tid = thread::thread_idx_x();
    let inv_d = Kernel::inv_d(cell_width);

    let ref_elt_pos_minus_particle_pos = particle_pos.dir_to_associated_grid_node(cell_width);

    // APIC grid-to-particle transfer.
    let mut interpolated_data = InterpolatedParticleData::default();

    let w = Kernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);

    let assoc_cell_before_integration = particle_pos.point.map(|e| (e / cell_width).round());
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
        interpolated_data.velocity += weight * cell.velocity;
        interpolated_data.velocity_gradient += (weight * inv_d) * cell.velocity * dpt.transpose();
        interpolated_data.psi_pos_momentum += weight * cell.psi_velocity;
        interpolated_data.velocity_gradient_det += weight * cell.velocity.dot(&dpt) * inv_d;
    }

    {
        let shift = NBH_SHIFTS[NBH_SHIFTS.len() - 1];
        let packed_shift = NBH_SHIFTS_SHARED[NBH_SHIFTS_SHARED.len() - 1];
        let dpt = ref_elt_pos_minus_particle_pos + shift.cast::<Real>() * cell_width;
        let cell = &*shared_nodes.add(packed_cell_index_in_block as usize + packed_shift);

        let proj_norm = cell.projection_scaled_dir.norm();

        if proj_norm > 1.0e-5 {
            let normal = cell.projection_scaled_dir / proj_norm;
            interpolated_data.projection_scaled_dir =
                cell.projection_scaled_dir - normal * dpt.dot(&normal);

            if interpolated_data.projection_scaled_dir.dot(&normal) < 0.0 {
                interpolated_data.projection_status = cell.projection_status.flip();
            } else {
                interpolated_data.projection_status = cell.projection_status;
            }
        }
    }

    if let Some((stress, force)) = particle_updater.update_particle_and_compute_kirchhoff_stress(
        dt,
        cell_width,
        colliders,
        particle_id,
        particle_status,
        particle_pos,
        particle_vel,
        particle_volume,
        particle_phase,
        &mut interpolated_data,
    ) {
        let inv_d = Kernel::inv_d(cell_width);
        let ref_elt_pos_minus_particle_pos = particle_pos.dir_to_associated_grid_node(cell_width);
        let w = Kernel::precompute_weights(ref_elt_pos_minus_particle_pos, cell_width);

        let affine = particle_volume.mass * interpolated_data.velocity_gradient
            - (particle_volume.volume0 * inv_d * dt) * stress;
        let momentum = particle_volume.mass * particle_vel.vector + force * dt;

        let psi_mass = if particle_phase.phase > 0.0 && !particle_status.failed {
            particle_volume.mass
        } else {
            0.0
        };

        let psi_pos_momentum = psi_mass * particle_phase.psi_pos;

        let assoc_cell_after_integration = particle_pos.point.map(|e| (e / cell_width).round());
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
                shared_node.projection_scaled_dir = global_node.projection_scaled_dir;
                shared_node.projection_status = global_node.projection_status;
            } else {
                shared_node.velocity = na::zero();
                shared_node.psi_velocity = na::zero();
                shared_node.projection_scaled_dir = na::zero();
                shared_node.projection_status = GpuGridProjectionStatus::NotComputed;
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
            shared_node.projection_scaled_dir = na::zero();
            shared_node.projection_status = GpuGridProjectionStatus::NotComputed;
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
