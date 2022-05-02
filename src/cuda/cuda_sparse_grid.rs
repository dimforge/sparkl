use crate::core::math::Point;
use crate::cuda::{CudaParticleSet, SingleGpuMpmContext};
use crate::kernels::cuda::{ActiveBlockHeader, GridHashMap, GridHashMapEntry};
use crate::kernels::{
    BlockHeaderId, BlockVirtualId, DispatchBlock2ActiveBlock, GpuGrid, GpuGridNode, HaloBlockData,
    NUM_CELL_PER_BLOCK,
};
use crate::math::{Real, DIM};
use crate::utils::PrefixSumWorkspace;
use cust::memory::GpuBuffer;
use cust::{
    error::CudaResult,
    launch,
    memory::{CopyDestination, DeviceBox, DeviceBuffer},
    prelude::Stream,
};
use std::collections::{HashMap, HashSet};

struct CudaSparseGridData {
    buffer: DeviceBuffer<GpuGridNode>,
    active_blocks: DeviceBuffer<ActiveBlockHeader>,
    packed_key_to_active_block: DeviceBuffer<GridHashMapEntry>,
    block_capacity: u32,
}

pub struct CudaSparseGrid {
    cell_width: Real,

    curr_data: CudaSparseGridData,
    next_data: CudaSparseGridData,

    dispatch_block_to_active_block: DeviceBuffer<DispatchBlock2ActiveBlock>,
    dispatch_halo_block_to_active_block: DeviceBuffer<DispatchBlock2ActiveBlock>,
    num_active_blocks: DeviceBox<u32>,
    pub(crate) num_halo_blocks: DeviceBox<u32>,
    num_remote_active_blocks: u32,
    // Active blocks read from other GPUs.
    remote_active_blocks: DeviceBuffer<ActiveBlockHeader>,
    pub(crate) remote_halo_blocks: DeviceBuffer<HaloBlockData>,
    pub(crate) halo_blocks_staging: DeviceBuffer<HaloBlockData>,
    scan_values: DeviceBuffer<u32>,
    halo_scan_values: DeviceBuffer<u32>,
    scan_workspace: PrefixSumWorkspace,
    halo_scan_workspace: PrefixSumWorkspace,
    zero: DeviceBox<u32>,
}

impl CudaSparseGrid {
    pub fn new(cell_width: Real) -> CudaResult<Self> {
        Self::with_capacity(cell_width, 1024)
    }

    pub fn with_capacity(cell_width: Real, block_capacity: u32) -> CudaResult<Self> {
        let block_capacity = block_capacity.next_power_of_two().max(1024); // Make sure it’s a power of two.

        // SAFETY: Ok because 0 is a valid bit pattern for the GpuGridNode.
        let hashmap_init = vec![GridHashMapEntry::free(); block_capacity as usize];
        let cell_capacity = block_capacity as usize * NUM_CELL_PER_BLOCK as usize;

        Ok(Self {
            cell_width,
            curr_data: CudaSparseGridData {
                buffer: DeviceBuffer::zeroed(cell_capacity)?,
                active_blocks: DeviceBuffer::zeroed(block_capacity as usize)?,
                packed_key_to_active_block: DeviceBuffer::from_slice(&hashmap_init)?,
                block_capacity,
            },
            next_data: CudaSparseGridData {
                buffer: DeviceBuffer::zeroed(cell_capacity)?,
                active_blocks: DeviceBuffer::zeroed(block_capacity as usize)?,
                packed_key_to_active_block: DeviceBuffer::from_slice(&hashmap_init)?,
                block_capacity,
            },
            dispatch_block_to_active_block: DeviceBuffer::zeroed(block_capacity as usize)?,
            dispatch_halo_block_to_active_block: DeviceBuffer::zeroed(block_capacity as usize)?,
            num_active_blocks: DeviceBox::zeroed()?,
            num_halo_blocks: DeviceBox::zeroed()?,
            num_remote_active_blocks: 0,
            remote_active_blocks: DeviceBuffer::zeroed(block_capacity as usize)?,
            scan_values: DeviceBuffer::zeroed(block_capacity as usize)?,
            halo_scan_values: DeviceBuffer::zeroed(block_capacity as usize)?,
            scan_workspace: PrefixSumWorkspace::with_capacity(block_capacity as u32)?,
            halo_scan_workspace: PrefixSumWorkspace::with_capacity(block_capacity as u32)?,
            zero: DeviceBox::zeroed()?,
            halo_blocks_staging: DeviceBuffer::zeroed(block_capacity as usize)?,
            remote_halo_blocks: DeviceBuffer::zeroed(block_capacity as usize)?,
        })
    }

    pub fn cell_width(&self) -> Real {
        self.cell_width
    }

    pub fn block_associated_to_point(&self, pt: &Point<Real>) -> BlockVirtualId {
        const OFF_BY_TWO: i64 = 2;
        let coord = (pt / self.cell_width).map(|e| e.round() as i64);
        BlockVirtualId::pack(
            coord
                .coords
                .map(|x| (x - OFF_BY_TWO + BlockVirtualId::PACK_ORIGIN as i64 * 4) as usize / 4),
        )
    }

    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.curr_data, &mut self.next_data);
    }

    pub fn curr_device_elements(&mut self) -> GpuGrid {
        unsafe {
            GpuGrid::new(
                self.cell_width,
                self.curr_data.block_capacity as u64 * NUM_CELL_PER_BLOCK,
                self.curr_data.buffer.as_device_ptr(),
                self.curr_data.active_blocks.as_device_ptr(),
                self.num_active_blocks.as_device_ptr(),
                self.dispatch_block_to_active_block.as_device_ptr(),
                self.dispatch_halo_block_to_active_block.as_device_ptr(),
                GridHashMap::from_raw_parts(
                    self.curr_data.packed_key_to_active_block.as_device_ptr(),
                    self.curr_data.block_capacity,
                ),
            )
        }
    }

    pub fn next_device_elements(&mut self) -> GpuGrid {
        unsafe {
            GpuGrid::new(
                self.cell_width,
                self.next_data.block_capacity as u64 * NUM_CELL_PER_BLOCK,
                self.next_data.buffer.as_device_ptr(),
                self.next_data.active_blocks.as_device_ptr(),
                self.num_active_blocks.as_device_ptr(),
                self.dispatch_block_to_active_block.as_device_ptr(),
                self.dispatch_halo_block_to_active_block.as_device_ptr(),
                GridHashMap::from_raw_parts(
                    self.next_data.packed_key_to_active_block.as_device_ptr(),
                    self.next_data.block_capacity,
                ),
            )
        }
    }

    // Returns the pair (number of active blocks, number of GPU dispatch blocks needed to cover all the particles).
    pub unsafe fn launch_sort(
        contexts: &mut [SingleGpuMpmContext],
        max_threads_per_gpu_block: u32,
    ) -> CudaResult<()> {
        use cust::memory::AsyncCopyDestination;

        let multigpu = contexts.len() > 1;
        let nthreads = 512;

        for context in &mut *contexts {
            context.sparse_grid_has_the_correct_size = false;
            context.grid.num_remote_active_blocks = 0;
        }

        // Retry until we allocated enough room on the sparse grid for all the blocks.
        loop {
            for context in &mut *contexts {
                if !context.sparse_grid_has_the_correct_size {
                    context.make_current()?;
                    let stream = &context.stream;
                    let module = &context.module;

                    let next_grid = context.grid.next_device_elements();
                    context
                        .grid
                        .num_active_blocks
                        .async_copy_from(&context.grid.zero, stream)?;

                    if multigpu {
                        context
                            .grid
                            .num_halo_blocks
                            .async_copy_from(&context.grid.zero, stream)?;
                    }

                    {
                        let hashmap = GridHashMap::from_raw_parts(
                            context
                                .grid
                                .next_data
                                .packed_key_to_active_block
                                .as_device_ptr(),
                            context.grid.next_data.block_capacity,
                        );
                        let nthreads = 1024;
                        let nblocks = context.grid.next_data.block_capacity as u32 / nthreads + 1;
                        launch!(module.reset_hashmap<<<nblocks, nthreads, 0, stream>>>(hashmap))?;
                    }

                    let particles_ptr = context.particles.particle_pos.as_device_ptr();
                    let particles_len = context.particles.len() as u32; // NOTE: must be u32 for the kernels.
                    let ngroups = particles_len / nthreads + 1;

                    launch!(module.touch_particle_blocks<<<ngroups, nthreads, 0, stream>>>(
                        particles_ptr, particles_len, next_grid
                    ))?;
                }
            }

            for context in &mut *contexts {
                if !context.sparse_grid_has_the_correct_size {
                    context.make_current()?;
                    let stream = &context.stream;

                    stream.synchronize()?;
                    context
                        .grid
                        .num_active_blocks
                        .copy_to(&mut context.num_active_blocks)?;

                    // The higher the load factor, the less memory will be used by the sparse grid,
                    // but that may reduce performances.
                    const MAX_HASHMAP_LOAD_FACTOR: u32 = 50;
                    if context.num_active_blocks
                        > context.grid.next_data.block_capacity * MAX_HASHMAP_LOAD_FACTOR / 100
                    {
                        // Estimate the number of blocks, assuming 2x2x2 particles per cell, and 4x4x4
                        // cells per block.
                        let estimated_capacity = ((context.particles.len() * 100)
                            / (4usize.pow(DIM as u32)
                                * 2usize.pow(DIM as u32)
                                * MAX_HASHMAP_LOAD_FACTOR as usize))
                            .next_power_of_two()
                            .max(context.grid.curr_data.block_capacity as usize);

                        // We reached an occupancy that is too high for the sparse grid.
                        // Resize the grid to make more room.
                        // NOTE: must remain a power of two.
                        let new_capacity = estimated_capacity
                            .max(context.grid.next_data.block_capacity as usize * 2);

                        context.grid.next_data.active_blocks = DeviceBuffer::zeroed(new_capacity)?;
                        context.grid.next_data.packed_key_to_active_block =
                            DeviceBuffer::zeroed(new_capacity)?;
                        context.grid.next_data.block_capacity = new_capacity as u32;
                        context.grid.scan_values = DeviceBuffer::zeroed(new_capacity)?;
                        context.grid.halo_scan_values = DeviceBuffer::zeroed(new_capacity)?;
                    } else {
                        context.sparse_grid_has_the_correct_size = true;
                    }
                }
            }

            if contexts.iter().all(|c| c.sparse_grid_has_the_correct_size) {
                break;
            }
        }

        // Hallo tagging (only if we are running on multiple GPUs).
        if multigpu {
            // Download the active block headers to all the GPUs.
            for i in 0..contexts.len() {
                for j in 0..contexts.len() {
                    if i != j {
                        contexts[i].grid.num_remote_active_blocks += contexts[j].num_active_blocks;
                    }
                }

                let mut context_i = &mut contexts[i];
                context_i.make_current()?;

                // First, make sure we have enough room for the remote active blocks.
                if context_i.grid.remote_active_blocks.len()
                    < context_i.grid.num_remote_active_blocks as usize
                {
                    context_i.grid.remote_active_blocks =
                        DeviceBuffer::zeroed(context_i.grid.num_remote_active_blocks as usize)?;
                }

                // Second, transfer the remote active blocks to this buffer.
                let mut shift = 0;
                for j in 0..contexts.len() {
                    if i != j {
                        let num_active_blocks_j = contexts[j].num_active_blocks;
                        let mut target_i = contexts[i]
                            .grid
                            .remote_active_blocks
                            .index(shift..shift + num_active_blocks_j as usize);
                        let source_j = contexts[j]
                            .grid
                            .next_data
                            .active_blocks
                            .index(..num_active_blocks_j as usize);

                        target_i.async_copy_from(&source_j, &contexts[i].stream)?;
                        shift += num_active_blocks_j as usize;
                    }
                }
            }

            // Launch the halo tagging kernel.
            for context in &mut *contexts {
                context.make_current()?;
                let module = &context.module;
                let stream = &context.stream;
                let next_grid = context.grid.next_device_elements();
                let nthreads = 1024;
                let ngroups = context.grid.num_remote_active_blocks / nthreads + 1;
                launch!(
                    module.tag_halo_blocks<<<ngroups, nthreads, 0, stream>>>(
                        next_grid,
                        context.grid.remote_active_blocks.as_device_ptr(),
                        context.grid.num_remote_active_blocks,
                        context.grid.num_halo_blocks.as_device_ptr(),
                    )
                )?;

                let ngroups = context.num_active_blocks / nthreads + 1;
                launch!(
                    module.tag_halo_neighbors<<<ngroups, nthreads, 0, stream>>>(
                        next_grid,
                        context.num_active_blocks,
                    )
                )?;
            }
        }

        for context in &mut *contexts {
            context.make_current()?;
            let module = &context.module;
            let stream = &context.stream;
            let num_active_blocks = context.num_active_blocks;
            let next_grid = context.grid.next_device_elements();
            let scan_values = context.grid.scan_values.as_device_ptr();
            let halo_scan_values = context.grid.halo_scan_values.as_device_ptr();
            let particles_len = context.particles.len() as u32; // NOTE: must be u32 for the kernels.
            let ngroups = particles_len / nthreads + 1;
            let sorted_particle_ids = context.particles.sorted_particle_ids.as_device_ptr();

            let particles_ptr = context.particles.particle_pos.as_device_ptr();
            launch!(module.update_block_particle_count<<<ngroups, nthreads, 0, stream>>>(
                particles_ptr, particles_len, next_grid,
            ))?;

            let n_block_groups = num_active_blocks / nthreads + 1;
            launch!(module.copy_particles_len_to_scan_value<<<n_block_groups, nthreads, 0, stream>>>(
                next_grid, scan_values,
            ))?;

            context.grid.scan_workspace.launch(
                module,
                stream,
                &mut context.grid.scan_values,
                num_active_blocks,
            )?;

            launch!(module.copy_scan_values_to_first_particles<<<n_block_groups, nthreads, 0, stream>>>(
                next_grid, scan_values
            ))?;

            launch!(module.finalize_particles_sort<<<ngroups, nthreads, 0, stream>>>(
                particles_ptr, particles_len, next_grid, scan_values, sorted_particle_ids
            ))?;

            /*
             *
             * Compute blocks multiplicity to setup the grid-block <-> GPU dispatch block mapping
             *
             */
            launch!(module.write_blocks_multiplicity_to_scan_value<<<n_block_groups, nthreads, 0, stream>>>(
                next_grid, scan_values, halo_scan_values, max_threads_per_gpu_block
            ))?;
            context.grid.scan_workspace.launch(
                module,
                stream,
                &mut context.grid.scan_values,
                num_active_blocks,
            )?;

            if multigpu {
                context.grid.halo_scan_workspace.launch(
                    module,
                    stream,
                    &mut context.grid.halo_scan_values,
                    num_active_blocks,
                )?;
            }
        }

        for context in &mut *contexts {
            context.make_current()?;
            let stream = &context.stream;
            let module = &context.module;
            let num_active_blocks = context.num_active_blocks;

            stream.synchronize()?; // Needed to read the total number of GPU dispatch blocks needed.

            context.num_dispatch_blocks = context.grid.scan_workspace.read_max_scan_value()?;

            if multigpu {
                context.num_dispatch_halo_blocks =
                    context.grid.halo_scan_workspace.read_max_scan_value()?;
            }
            info!(
                "num dispatch blocks: {}, num halo dispatch blocks: {}",
                context.num_dispatch_blocks, context.num_dispatch_halo_blocks
            );

            if context.grid.dispatch_block_to_active_block.len()
                < context.num_dispatch_blocks as usize
            {
                context.grid.dispatch_block_to_active_block =
                    DeviceBuffer::zeroed(context.num_dispatch_blocks.next_power_of_two() as usize)?;
            }

            if multigpu
                && context.grid.dispatch_halo_block_to_active_block.len()
                    < context.num_dispatch_halo_blocks as usize
            {
                context.grid.dispatch_halo_block_to_active_block = DeviceBuffer::zeroed(
                    context.num_dispatch_halo_blocks.next_power_of_two() as usize,
                )?;
            }

            let next_grid = context.grid.next_device_elements(); // Needed to grab the new dispatch block ptr if it was resized.
            let scan_values = context.grid.scan_values.as_device_ptr();
            let halo_scan_values = context.grid.halo_scan_values.as_device_ptr();
            launch!(module.init_gpu_dispatch_blocks_mapping<<<num_active_blocks, 32, 0, stream>>>(
                next_grid, scan_values, halo_scan_values, max_threads_per_gpu_block
            ))?;
        }

        if multigpu {
            // At this point we called synchronize on all the streams, and this sync happens
            // after the halo tagging was launched, so we can:
            // - Read the number of halo blocks.
            // - Resize the blocks data buffer.
            // - Resize the halo blocks data buffer.
            for context in &mut *contexts {
                context.make_current()?;
                context.num_halo_blocks = context.grid.num_halo_blocks.as_host_value()?;
                info!(
                    "num halo blocks: {}, num active blocks: {}",
                    context.num_halo_blocks, context.num_active_blocks
                );
            }

            for i in 0..contexts.len() {
                contexts[i].num_remote_halo_blocks = 0;

                for j in 0..contexts.len() {
                    if i != j {
                        contexts[i].num_remote_halo_blocks += contexts[j].num_halo_blocks;
                    }
                }
            }
        }

        for context in &mut *contexts {
            context.make_current()?;
            let desired_buffer_len = context.num_active_blocks * NUM_CELL_PER_BLOCK as u32;
            if context.grid.next_data.buffer.len() < desired_buffer_len as usize {
                context.grid.next_data.buffer = DeviceBuffer::zeroed(desired_buffer_len as usize)?;
            }

            if multigpu
                && context.grid.remote_halo_blocks.len() < context.num_remote_halo_blocks as usize
            {
                context.grid.remote_halo_blocks =
                    DeviceBuffer::zeroed(context.num_remote_halo_blocks as usize)?;
            }

            if multigpu && context.grid.halo_blocks_staging.len() < context.num_halo_blocks as usize
            {
                context.grid.halo_blocks_staging =
                    DeviceBuffer::zeroed(context.num_halo_blocks as usize)?;
            }
        }

        // NOTE: uncomment this to test the validity of the sorted blocks.
        // context.grid.check_active_blocks(
        //     stream,
        //     num_active_blocks,
        //     num_dispatch_blocks,
        //     particles,
        //     max_threads_per_gpu_block,
        // );

        Ok(())
    }

    #[allow(dead_code)] // For testing only.
    fn check_active_blocks(
        &self,
        stream: &Stream,
        num_active_blocks: u32,
        num_dispatch_blocks: u32,
        particles: &CudaParticleSet,
        max_threads_per_gpu_block: u32,
    ) -> CudaResult<()> {
        stream.synchronize().unwrap();
        let mut active_blocks = vec![ActiveBlockHeader::default(); num_active_blocks as usize];
        let mut dispatch_blocks =
            vec![DispatchBlock2ActiveBlock::default(); num_dispatch_blocks as usize];
        let mut hashmap_entries =
            vec![GridHashMapEntry::free(); self.next_data.block_capacity as usize];

        let particles_pos = particles.read_positions().unwrap();
        let sorted_particle_ids = particles.read_sorted_particle_ids().unwrap();

        self.next_data
            .active_blocks
            .index(..num_active_blocks as usize)
            .copy_to(&mut active_blocks)?;
        self.dispatch_block_to_active_block
            .index(..num_dispatch_blocks as usize)
            .copy_to(&mut dispatch_blocks)?;
        self.next_data
            .packed_key_to_active_block
            .copy_to(&mut hashmap_entries)?;

        // Check that each sorted particle id is valid and unique.
        let mut ids_found = HashSet::new();
        for sorted_id in &sorted_particle_ids {
            if !ids_found.insert(*sorted_id) {
                panic!("Non-unique sorted particle id found.");
            }

            if *sorted_id as usize > particles.len() {
                panic!("Out-of-bounds particle id found.");
            }
        }

        // Check that all the active blocks are contained by the hashmap.
        let hashmap: HashMap<_, _> = hashmap_entries
            .into_iter()
            .map(|e| (e.key, e.value))
            .collect();
        for (i, block) in active_blocks.iter().enumerate() {
            assert_eq!(
                hashmap.get(&block.virtual_id).copied(),
                Some(BlockHeaderId(i as u32))
            );
        }
        assert_eq!(hashmap.len() - 1, active_blocks.len());

        // Check that the active blocks contains coherent data:
        // - Each active block must have a unique virtual_id
        // - The number of particles must match what we find with a naive implementation.
        // - Each particle read from the sorted particle array must actually belong to this block.
        let mut blocks_found = HashSet::new();
        for block in &active_blocks {
            if !blocks_found.insert(block.virtual_id) {
                panic!("Non-unique block found.")
            }

            for i in block.first_particle..block.first_particle + block.num_particles {
                let pid = sorted_particle_ids[i as usize];
                let particle_pos = particles_pos[pid as usize].point;
                let packed_id = self.block_associated_to_point(&particle_pos);
                assert_eq!(packed_id, block.virtual_id);
            }
        }

        // Check that if a block is active with a non-zero number of particles,
        // then it’s 7 neighbors are also active.
        for block in &active_blocks {
            if block.num_particles > 0 {
                for assoc_block in GpuGrid::blocks_associated_to_block(block.virtual_id.unpack()) {
                    if !blocks_found.contains(&assoc_block) {
                        info!("The block {:?} contains {} particles, but it’s adj block: {:?} wasn’t found.",
                        block.virtual_id.unpack(), block.num_particles, assoc_block.unpack());
                    }
                    assert!(blocks_found.contains(&assoc_block));
                }
            }
        }

        // Check that the dispatch block 2 active block mapping is valid:
        // - The number of dispatch block associated to each active block must be correct.
        // - The particles addressed by each dispatch block must cover all the particles
        //   associated to the active block, without duplicates.
        let mut block_assocs = HashMap::new();
        for dispatch_block in &dispatch_blocks {
            block_assocs
                .entry(dispatch_block.active_block_id)
                .or_insert(vec![])
                .push(*dispatch_block);

            let active_block = &active_blocks[dispatch_block.active_block_id.0 as usize];
            assert!(dispatch_block.first_particle >= active_block.first_particle);
            assert!(
                dispatch_block.first_particle
                    < active_block.first_particle + active_block.num_particles
            );
        }

        // Check that all active block with non-zero multiplicity has at least one dispatch block.
        for (i, active_block) in active_blocks.iter().enumerate() {
            let multiplicity = active_block.multiplicity(max_threads_per_gpu_block);
            if multiplicity != 0 {
                assert_eq!(
                    block_assocs.get(&BlockHeaderId(i as u32)).map(|b| b.len()),
                    Some(multiplicity as usize)
                );
            }
        }

        info!("Num block assocs: {}", block_assocs.len());
        for (active_id, disp_blocks) in block_assocs.iter_mut() {
            let active_block = &active_blocks[active_id.0 as usize];
            assert_eq!(
                active_block.multiplicity(max_threads_per_gpu_block),
                disp_blocks.len() as u32
            );
            disp_blocks.sort_by_key(|b| b.first_particle);

            assert_eq!(disp_blocks[0].first_particle, active_block.first_particle);
            for window in disp_blocks.windows(2) {
                assert_eq!(
                    window[0].first_particle + max_threads_per_gpu_block,
                    window[1].first_particle
                );
            }
        }

        /*
         * Check that each particle is traversed exactly once when traversing the dispatch blocks.
         * This is very similar to the particle addressing done by P2G.
         */
        let mut seen = vec![false; particles.len()];
        for bid in 0..num_dispatch_blocks {
            for tid in 0..max_threads_per_gpu_block {
                let dispatch_block_to_active_block = dispatch_blocks[bid as usize];
                let active_block =
                    active_blocks[dispatch_block_to_active_block.active_block_id.0 as usize];

                if dispatch_block_to_active_block.first_particle + tid
                    < active_block.first_particle + active_block.num_particles
                {
                    let particle_id = sorted_particle_ids
                        [(dispatch_block_to_active_block.first_particle + tid) as usize];
                    assert!(!seen[particle_id as usize], "Saw a particle twice.");
                    seen[particle_id as usize] = true;
                }
            }
        }
        let num_seen = seen.iter().filter(|b| **b).count();
        info!("Num seen: {} vs. {}", num_seen, particles.len());
        assert!(seen.iter().all(|b| *b));

        Ok(())
    }
}
