use crate::cuda::atomic::AtomicAdd;
use crate::cuda::AtomicInt;
use crate::gpu_grid::GpuGrid;
use crate::{
    BlockHeaderId, BlockVirtualId, DispatchBlock2ActiveBlock, HaloBlockData, NUM_CELL_PER_BLOCK,
};
use cuda_std::thread;
use sparkl_core::dynamics::ParticlePosition;

// NOTE: this is similar to what we could have gotten with bitflags,
// except that we have direct access to the bits to run atomic operations.
#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy, bytemuck::Zeroable))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct HaloState(pub u32);
impl HaloState {
    pub const EMPTY: Self = HaloState(0);
    pub const IS_HALO: Self = HaloState(1);
    pub const HAS_HALO_NEIGHBOR: Self = HaloState(2);

    pub fn contains(self, rhs: Self) -> bool {
        (self.0 & rhs.0) != 0
    }
    pub fn needs_halo_treatment(self) -> bool {
        self.contains(Self::IS_HALO) || self.contains(Self::HAS_HALO_NEIGHBOR)
    }
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy, bytemuck::Zeroable))]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct ActiveBlockHeader {
    // Needed to compute the world-space position of the block.
    pub virtual_id: BlockVirtualId,
    pub first_particle: u32,
    pub num_particles: u32,
    pub halo_state: HaloState,
}

impl ActiveBlockHeader {
    pub fn multiplicity(&self, max_threads_per_block: u32) -> u32 {
        self.num_particles / max_threads_per_block
            + (self.num_particles % max_threads_per_block > 0) as u32
    }
}

impl Default for ActiveBlockHeader {
    fn default() -> Self {
        Self {
            virtual_id: BlockVirtualId(0),
            first_particle: 0,
            num_particles: 0,
            halo_state: HaloState::EMPTY,
        }
    }
}

#[cuda_std::kernel]
pub unsafe fn touch_particle_blocks(
    particles: *mut ParticlePosition,
    particles_len: u32,
    mut grid: GpuGrid,
) {
    let id = thread::index();
    if id < particles_len {
        let p = &*particles.add(id as usize);

        for block_id in grid.blocks_associated_to_point(&p.point) {
            grid.mark_block_as_active(block_id);
        }
    }
}

#[cuda_std::kernel]
pub unsafe fn tag_halo_blocks(
    mut grid: GpuGrid,
    remote_active_blocks: *const ActiveBlockHeader,
    num_remote_active_blocks: u32,
    num_halo_blocks: *mut u32,
) {
    let id = thread::index();
    if id < num_remote_active_blocks {
        let block_vid = (*remote_active_blocks.add(id as usize)).virtual_id;
        if let Some(block_hid) = grid.get_header_block_id(block_vid) {
            if grid
                .active_block_unchecked_mut(block_hid)
                .halo_state
                .0
                .global_atomic_exch(HaloState::IS_HALO.0)
                == 0
            {
                (*num_halo_blocks).global_atomic_add(1);
            }
        }
    }
}

#[cuda_std::kernel]
pub unsafe fn tag_halo_neighbors(mut grid: GpuGrid, num_active_blocks: u32) {
    let id = thread::index();
    if id < num_active_blocks {
        let active_block = grid.active_block_unchecked(BlockHeaderId(id));
        if active_block.halo_state.contains(HaloState::IS_HALO) {
            let assoc_blocks =
                GpuGrid::blocks_transferring_into_block(active_block.virtual_id.unpack());
            for assoc_block in &assoc_blocks[1..] {
                if let Some(active_assoc_block_id) = grid.get_header_block_id(*assoc_block) {
                    let active_assoc_block = grid.active_block_unchecked_mut(active_assoc_block_id);
                    active_assoc_block.halo_state.0 |= HaloState::HAS_HALO_NEIGHBOR.0;
                }
            }
        }
    }
}

#[cuda_std::kernel]
pub unsafe fn copy_halo_to_staging(
    grid: GpuGrid,
    staging_buffer: *mut HaloBlockData,
    num_halo_blocks: *mut u32,
) {
    let id = thread::index();
    if id < grid.num_active_blocks() {
        let block_id = BlockHeaderId(id);
        let block = grid.active_block_unchecked(block_id);

        if block.halo_state.contains(HaloState::IS_HALO) {
            let index = (&mut *num_halo_blocks).global_atomic_dec() - 1;

            let out = &mut *staging_buffer.add(index as usize);
            out.virtual_id = block.virtual_id;
            let first_cell_id = block_id.to_physical();

            for k in 0..NUM_CELL_PER_BLOCK {
                let curr_cell_id = first_cell_id.node(k);
                out.cells[k as usize] = *grid.get_node_unchecked(curr_cell_id);
            }
        }
    }
}

#[cuda_std::kernel]
pub unsafe fn merge_halo_blocks(mut grid: GpuGrid, remote_halo_blocks: *const HaloBlockData) {
    let bid = thread::block_idx_x();
    let tid = thread::thread_idx_x();

    let halo_block = &*remote_halo_blocks.add(bid as usize);
    if let Some(target_block_id) = grid.get_header_block_id(halo_block.virtual_id) {
        let node_id = target_block_id.to_physical().node(tid as u64);
        let target_node = &mut *grid.get_node_mut(node_id).unwrap();
        target_node
            .mass
            .global_red_add(halo_block.cells[tid as usize].mass);
        target_node
            .momentum_velocity
            .global_red_add(halo_block.cells[tid as usize].momentum_velocity);
        target_node
            .psi_mass
            .global_red_add(halo_block.cells[tid as usize].psi_mass);
        target_node
            .psi_momentum_velocity
            .global_red_add(halo_block.cells[tid as usize].psi_momentum_velocity);
    }
}

#[cuda_std::kernel]
pub unsafe fn update_block_particle_count(
    particles: *mut ParticlePosition,
    particles_len: u32,
    mut grid: GpuGrid,
) {
    let id = thread::index();
    if id < particles_len {
        let p = &*particles.add(id as usize);
        let block_id = grid.block_associated_to_point(&p.point);

        if let Some(active_block) = grid.get_packed_block_mut(block_id) {
            active_block.num_particles.global_red_add(1)
        }
    }
}

#[cuda_std::kernel]
pub unsafe fn copy_particles_len_to_scan_value(grid: GpuGrid, scan_values: *mut u32) {
    let id = thread::index();
    if id < grid.num_active_blocks() {
        *scan_values.add(id as usize) =
            grid.active_block_unchecked(BlockHeaderId(id)).num_particles;
    }
}

#[cuda_std::kernel]
pub unsafe fn copy_scan_values_to_first_particles(mut grid: GpuGrid, scan_values: *const u32) {
    let id = thread::index();
    if id < grid.num_active_blocks() {
        grid.active_block_unchecked_mut(BlockHeaderId(id))
            .first_particle = *scan_values.add(id as usize);
    }
}

#[cuda_std::kernel]
pub unsafe fn finalize_particles_sort(
    particles: *mut ParticlePosition,
    particles_len: u32,
    grid: GpuGrid,
    scan_values: *mut u32,
    sorted_particle_ids: *mut u32,
) {
    let id = thread::index();
    if id < particles_len {
        let p = &*particles.add(id as usize);
        let block_id = grid.block_associated_to_point(&p.point);
        // Place the particles to their rightful place.
        // TODO: store the block id inside of the particle instead?
        if let Some(active_block_id) = grid.get_header_block_id(block_id) {
            let scan_value = &mut *scan_values.add(active_block_id.0 as usize);
            let target_index = scan_value.global_atomic_add(1);
            *sorted_particle_ids.add(target_index as usize) = id;
        }
    }
}

/*
 * Kernel for handling block multiplicity for mapping between grid blocks and GPU dispatch blocks.
 */
#[cuda_std::kernel]
pub unsafe fn write_blocks_multiplicity_to_scan_value(
    grid: GpuGrid,
    scan_values: *mut u32,
    halo_scan_values: *mut u32,
    max_threads_per_block: u32,
) {
    let id = thread::index();
    if id < grid.num_active_blocks() {
        let active_block = grid.active_block_unchecked(BlockHeaderId(id));
        let multiplicity = active_block.multiplicity(max_threads_per_block);

        if active_block.halo_state.needs_halo_treatment() {
            *scan_values.add(id as usize) = 0;
            *halo_scan_values.add(id as usize) = multiplicity;
        } else {
            *scan_values.add(id as usize) = multiplicity;
            *halo_scan_values.add(id as usize) = 0;
        }
    }
}

#[cuda_std::kernel]
pub unsafe fn init_gpu_dispatch_blocks_mapping(
    grid: GpuGrid,
    not_halo_scan_values: *mut u32,
    halo_scan_values: *mut u32,
    max_threads_per_gpu_block: u32,
) {
    let mut tid = thread::thread_idx_x();
    let bid = BlockHeaderId(thread::block_idx_x());
    let bsize = thread::block_dim_x();

    let active_block = grid.active_block_unchecked(bid);

    let (dispatch2active, scan_values) = if active_block.halo_state.needs_halo_treatment() {
        (grid.dispatch_halo_block_to_active_block, halo_scan_values)
    } else {
        (grid.dispatch_block_to_active_block, not_halo_scan_values)
    };

    let multiplicity = active_block.multiplicity(max_threads_per_gpu_block);
    let first_particle = active_block.first_particle;
    let base_dispatch_block_id = *scan_values.add(bid.0 as usize);

    while tid < multiplicity {
        *dispatch2active
            .as_mut_ptr()
            .add((base_dispatch_block_id + tid) as usize) = DispatchBlock2ActiveBlock {
            active_block_id: bid,
            first_particle: first_particle + tid * max_threads_per_gpu_block,
        };
        tid += bsize;
    }
}
