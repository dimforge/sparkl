use crate::cuda::{ActiveBlockHeader, AtomicAdd, AtomicInt, GridHashMap, HaloState};
use crate::DevicePointer;
use na::vector;
#[cfg(not(feature = "std"))]
use na::ComplexField;
use sparkl_core::math::{Point, Real, Vector};

#[cfg(feature = "dim2")]
pub const NUM_CELL_PER_BLOCK: u64 = 4 * 4;
#[cfg(feature = "dim3")]
pub const NUM_CELL_PER_BLOCK: u64 = 4 * 4 * 4;
#[cfg(feature = "dim2")]
pub const NUM_ASSOC_BLOCKS: usize = 4;
#[cfg(feature = "dim3")]
pub const NUM_ASSOC_BLOCKS: usize = 8;

#[cfg(feature = "dim2")]
pub const NBH_SHIFTS: [Vector<usize>; 9] = [
    vector![2, 2],
    vector![2, 0],
    vector![2, 1],
    vector![0, 2],
    vector![0, 0],
    vector![0, 1],
    vector![1, 2],
    vector![1, 0],
    vector![1, 1],
];

#[cfg(feature = "dim3")]
pub const NBH_SHIFTS: [Vector<usize>; 27] = [
    vector![2, 2, 2],
    vector![2, 0, 2],
    vector![2, 1, 2],
    vector![0, 2, 2],
    vector![0, 0, 2],
    vector![0, 1, 2],
    vector![1, 2, 2],
    vector![1, 0, 2],
    vector![1, 1, 2],
    vector![2, 2, 0],
    vector![2, 0, 0],
    vector![2, 1, 0],
    vector![0, 2, 0],
    vector![0, 0, 0],
    vector![0, 1, 0],
    vector![1, 2, 0],
    vector![1, 0, 0],
    vector![1, 1, 0],
    vector![2, 2, 1],
    vector![2, 0, 1],
    vector![2, 1, 1],
    vector![0, 2, 1],
    vector![0, 0, 1],
    vector![0, 1, 1],
    vector![1, 2, 1],
    vector![1, 0, 1],
    vector![1, 1, 1],
];

#[cfg(feature = "dim3")]
pub const NBH_SHIFTS_SHARED: [usize; 27] = [
    146, 130, 138, 144, 128, 136, 145, 129, 137, 18, 2, 10, 16, 0, 8, 17, 1, 9, 82, 66, 74, 80, 64,
    72, 81, 65, 73,
];
#[cfg(feature = "dim2")]
pub const NBH_SHIFTS_SHARED: [usize; 9] = [18, 2, 10, 16, 0, 8, 17, 1, 9];

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, bytemuck::Zeroable, Default)]
#[repr(transparent)]
pub struct BlockVirtualId(pub u64);

impl BlockVirtualId {
    #[cfg(feature = "dim2")]
    pub const PACK_ORIGIN: u64 = 0x0000_8000_0000;
    #[cfg(feature = "dim3")]
    pub const PACK_ORIGIN: u64 = 0x0010_0000;

    #[cfg(feature = "dim2")]
    const MASK: u64 = 0x0000_0000_ffff_ffff;
    #[cfg(feature = "dim3")]
    const MASK: u64 = 0b0001_1111_1111_1111_1111_1111;

    pub fn unpack_pos_on_signed_grid(self) -> Vector<i64> {
        self.unpack().cast::<i64>() - Vector::repeat(Self::PACK_ORIGIN as i64)
    }

    pub fn pack_pos_on_signed_grid(pos: Vector<i64>) -> BlockVirtualId {
        Self::pack((pos + Vector::repeat(Self::PACK_ORIGIN as i64)).map(|e| e as usize))
    }

    #[cfg(feature = "dim2")]
    pub fn pack(idx: Vector<usize>) -> BlockVirtualId {
        // In 2D, we can take the 32 first bits into account (using a total of 64 bits).
        BlockVirtualId((idx.x as u64 & Self::MASK) | ((idx.y as u64 & Self::MASK) << 32))
    }

    #[cfg(feature = "dim3")]
    pub fn pack(idx: Vector<usize>) -> BlockVirtualId {
        // In 3D, we can only take the 21 first bits into account (using a total of 63 bits).
        BlockVirtualId(
            (idx.x as u64 & Self::MASK)
                | ((idx.y as u64 & Self::MASK) << 21)
                | ((idx.z as u64 & Self::MASK) << 42),
        )
    }

    #[cfg(feature = "dim2")]
    pub fn unpack(self) -> Vector<usize> {
        vector![self.0 & Self::MASK, (self.0 >> 32) & Self::MASK].cast::<usize>()
    }

    #[cfg(feature = "dim3")]
    pub fn unpack(self) -> Vector<usize> {
        vector![
            self.0 & Self::MASK,
            (self.0 >> 21) & Self::MASK,
            (self.0 >> 42) & Self::MASK
        ]
        .cast::<usize>()
    }
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, bytemuck::Zeroable, Default)]
#[repr(transparent)]
pub struct BlockHeaderId(pub u32);

impl BlockHeaderId {
    pub fn to_physical(self) -> BlockPhysicalId {
        BlockPhysicalId(self.0 as u64 * NUM_CELL_PER_BLOCK)
    }
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct BlockPhysicalId(pub u64);

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct NodePhysicalId(pub u64);

impl BlockPhysicalId {
    // All the components of the shift must be between 0 and 4.
    #[cfg(feature = "dim2")]
    pub fn node_id_unchecked(self, shift: Vector<usize>) -> NodePhysicalId {
        NodePhysicalId(self.0 + shift.x as u64 + shift.y as u64 * 4)
    }

    // All the components of the shift must be between 0 and 4.
    #[cfg(feature = "dim3")]
    pub fn node_id_unchecked(self, shift_in_block: Vector<usize>) -> NodePhysicalId {
        NodePhysicalId(
            self.0
                + shift_in_block.x as u64
                + shift_in_block.y as u64 * 4
                + shift_in_block.z as u64 * 4 * 4,
        )
    }

    pub fn node(self, i: u64) -> NodePhysicalId {
        NodePhysicalId(self.0 + i)
    }
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Clone, Copy, bytemuck::Zeroable)]
#[repr(C)]
pub struct DispatchBlock2ActiveBlock {
    pub active_block_id: BlockHeaderId,
    pub first_particle: u32,
}

impl Default for DispatchBlock2ActiveBlock {
    fn default() -> Self {
        Self {
            active_block_id: BlockHeaderId(0),
            first_particle: 0,
        }
    }
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct GpuGrid {
    cell_width: Real,
    // Double buffering: the curr buffer is read-only, the next buffer is read/write.
    buffer: DevicePointer<GpuGridNode>,
    active_blocks: DevicePointer<ActiveBlockHeader>,
    num_active_blocks: DevicePointer<u32>,
    pub packed_key_to_active_block: GridHashMap,
    pub dispatch_block_to_active_block: DevicePointer<DispatchBlock2ActiveBlock>,
    pub dispatch_halo_block_to_active_block: DevicePointer<DispatchBlock2ActiveBlock>,
    capacity: u64,
}

impl GpuGrid {
    /// Creates a new GPU grid.
    ///
    /// The input buffer must be a valid pointer to a buffer able to hold exactly `capacity` grid nodes.
    pub unsafe fn new(
        cell_width: Real,
        capacity: u64,
        buffer: DevicePointer<GpuGridNode>,
        active_blocks: DevicePointer<ActiveBlockHeader>,
        num_active_blocks: DevicePointer<u32>,
        dispatch_block_to_active_block: DevicePointer<DispatchBlock2ActiveBlock>,
        dispatch_halo_block_to_active_block: DevicePointer<DispatchBlock2ActiveBlock>,
        packed_key_to_active_block: GridHashMap,
    ) -> Self {
        Self {
            cell_width,
            capacity,
            buffer,
            active_blocks,
            num_active_blocks,
            dispatch_block_to_active_block,
            dispatch_halo_block_to_active_block,
            packed_key_to_active_block,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }

    pub fn cell_width(&self) -> Real {
        self.cell_width
    }

    pub unsafe fn dispatch_block_to_active_block_unchecked(
        &self,
        dispatch_block_id: usize,
    ) -> DispatchBlock2ActiveBlock {
        *self
            .dispatch_block_to_active_block
            .as_ptr()
            .add(dispatch_block_id)
    }

    pub unsafe fn dispatch_block_to_active_block_unchecked_mut(
        &mut self,
        dispatch_block_id: usize,
    ) -> &mut DispatchBlock2ActiveBlock {
        &mut (*self
            .dispatch_block_to_active_block
            .as_mut_ptr()
            .add(dispatch_block_id))
    }

    pub unsafe fn active_block_unchecked(&self, block_id: BlockHeaderId) -> &ActiveBlockHeader {
        &*self.active_blocks.as_ptr().add(block_id.0 as usize)
    }

    pub unsafe fn active_block_unchecked_mut(
        &mut self,
        block_id: BlockHeaderId,
    ) -> &mut ActiveBlockHeader {
        &mut *self.active_blocks.as_mut_ptr().add(block_id.0 as usize)
    }

    pub fn num_active_blocks(&self) -> u32 {
        unsafe { *self.num_active_blocks.as_ptr() }
    }

    pub fn num_active_blocks_mut(&mut self) -> &mut u32 {
        unsafe { &mut *self.num_active_blocks.as_mut_ptr() }
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

    pub fn blocks_associated_to_point(
        &self,
        pt: &Point<Real>,
    ) -> [BlockVirtualId; NUM_ASSOC_BLOCKS] {
        const OFF_BY_TWO: i64 = 2;
        let coord = (pt / self.cell_width).map(|e| e.round() as i64);
        let main_block = coord
            .coords
            .map(|x| (x - OFF_BY_TWO + BlockVirtualId::PACK_ORIGIN as i64 * 4) as usize / 4);
        Self::blocks_associated_to_block(main_block)
    }

    #[cfg(feature = "dim2")]
    pub fn blocks_associated_to_block(block: Vector<usize>) -> [BlockVirtualId; 4] {
        // TODO: we could improve this by:
        // - Using pre-computed packed shifts, instead of packing the shifted key.
        // - Selecting only the neighbor blocks actually affected by the particle.
        [
            BlockVirtualId::pack(block + vector![0, 0]),
            BlockVirtualId::pack(block + vector![0, 1]),
            BlockVirtualId::pack(block + vector![1, 0]),
            BlockVirtualId::pack(block + vector![1, 1]),
        ]
    }

    #[cfg(feature = "dim3")]
    pub fn blocks_associated_to_block(block: Vector<usize>) -> [BlockVirtualId; 8] {
        // TODO: we could improve this by:
        // - Using pre-computed packed shifts, instead of packing the shifted key.
        // - Selecting only the neighbor blocks actually affected by the particle.
        [
            BlockVirtualId::pack(block + vector![0, 0, 0]),
            BlockVirtualId::pack(block + vector![0, 0, 1]),
            BlockVirtualId::pack(block + vector![0, 1, 0]),
            BlockVirtualId::pack(block + vector![0, 1, 1]),
            BlockVirtualId::pack(block + vector![1, 0, 0]),
            BlockVirtualId::pack(block + vector![1, 0, 1]),
            BlockVirtualId::pack(block + vector![1, 1, 0]),
            BlockVirtualId::pack(block + vector![1, 1, 1]),
        ]
    }

    #[cfg(feature = "dim2")]
    pub fn blocks_transferring_into_block(block: Vector<usize>) -> [BlockVirtualId; 4] {
        // TODO: we could improve this by:
        // - Using pre-computed packed shifts, instead of packing the shifted key.
        // - Selecting only the neighbor blocks actually affected by the particle.
        [
            BlockVirtualId::pack(block - vector![0, 0]),
            BlockVirtualId::pack(block - vector![0, 1]),
            BlockVirtualId::pack(block - vector![1, 0]),
            BlockVirtualId::pack(block - vector![1, 1]),
        ]
    }

    #[cfg(feature = "dim3")]
    pub fn blocks_transferring_into_block(block: Vector<usize>) -> [BlockVirtualId; 8] {
        // TODO: we could improve this by:
        // - Using pre-computed packed shifts, instead of packing the shifted key.
        // - Selecting only the neighbor blocks actually affected by the particle.
        [
            BlockVirtualId::pack(block - vector![0, 0, 0]),
            BlockVirtualId::pack(block - vector![0, 0, 1]),
            BlockVirtualId::pack(block - vector![0, 1, 0]),
            BlockVirtualId::pack(block - vector![0, 1, 1]),
            BlockVirtualId::pack(block - vector![1, 0, 0]),
            BlockVirtualId::pack(block - vector![1, 0, 1]),
            BlockVirtualId::pack(block - vector![1, 1, 0]),
            BlockVirtualId::pack(block - vector![1, 1, 1]),
        ]
    }

    pub fn mark_block_as_active(&mut self, virtual_id: BlockVirtualId) {
        let mut hashmap = self.packed_key_to_active_block;
        hashmap.insert_nonexistant_with(virtual_id, || {
            // This is the first time we see this block.
            let block_header_id = unsafe { self.num_active_blocks_mut().global_atomic_add(1) };
            let active_block =
                unsafe { self.active_block_unchecked_mut(BlockHeaderId(block_header_id)) };
            *active_block = ActiveBlockHeader {
                virtual_id,
                first_particle: 0,
                num_particles: 0,
                halo_state: HaloState::EMPTY,
            };
            BlockHeaderId(block_header_id)
        })
    }

    pub unsafe fn get_packed_block_mut(
        &mut self,
        id: BlockVirtualId,
    ) -> Option<&mut ActiveBlockHeader> {
        let active_block_id = self.packed_key_to_active_block.get(id)?;
        Some(
            &mut *self
                .active_blocks
                .as_mut_ptr()
                .add(active_block_id.0 as usize),
        )
    }

    pub unsafe fn get_header_block_id(&self, id: BlockVirtualId) -> Option<BlockHeaderId> {
        self.packed_key_to_active_block.get(id)
    }

    pub unsafe fn get_node_unchecked(&self, id: NodePhysicalId) -> &GpuGridNode {
        core::mem::transmute(self.buffer.as_ptr().add(id.0 as usize) as *const _)
    }

    pub fn get_node(&self, id: NodePhysicalId) -> Option<&GpuGridNode> {
        if id.0 >= self.capacity {
            None
        } else {
            unsafe {
                Some(core::mem::transmute(
                    self.buffer.as_ptr().add(id.0 as usize) as *const _,
                ))
            }
        }
    }

    pub fn get_node_mut(&mut self, id: NodePhysicalId) -> Option<&mut GpuGridNode> {
        if id.0 >= self.capacity {
            None
        } else {
            unsafe {
                Some(core::mem::transmute(
                    self.buffer.as_mut_ptr().add(id.0 as usize),
                ))
            }
        }
    }

    pub fn get_node_id_at_coord(&self, node_coord: Point<i64>) -> Option<NodePhysicalId> {
        let block_coord = node_coord.coords.map(|e| (e as Real / 4.0).floor() as i64);
        let shift = (node_coord.coords - block_coord * 4).map(|e| e as usize);

        let block_virtual = BlockVirtualId::pack_pos_on_signed_grid(block_coord);
        let block_header = unsafe { self.get_header_block_id(block_virtual)? };
        let block_physical = block_header.to_physical();
        let node_physical = block_physical.node_id_unchecked(shift);

        Some(node_physical)
    }
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GpuGridProjectionStatus {
    NotComputed,
    Inside(usize),
    Outside(usize),
    TooFar,
}

impl GpuGridProjectionStatus {
    pub fn is_inside(self) -> bool {
        matches!(self, Self::Inside(_))
    }

    pub fn is_outside(self) -> bool {
        matches!(self, Self::Outside(_))
    }

    pub fn flip(self) -> Self {
        match self {
            Self::Inside(i) => Self::Outside(i),
            Self::Outside(i) => Self::Inside(i),
            _ => self,
        }
    }
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct CdfData {
    // The unsigned distance to the closest collider.
    min_unsigned_distance: u32,
    // The affinity and tag (inside/ outside) information stored for up to 16 colliders.
    color: u32,
    pub active: u32,
    // pub real_pos: Point<Real>, // the actual position of the node
    // pub cdf_pos: Point<Real>,  // the falsely assumed position of the node in the cdf kernel
    // pub particle_pos: Point<Real>,
}

impl Default for CdfData {
    fn default() -> Self {
        Self {
            min_unsigned_distance: u32::MAX,
            color: 0,
            active: 0,
            // real_pos: Point::default(),
            // cdf_pos: Point::default(),
            // particle_pos: Point::default(),
        }
    }
}

impl CdfData {
    pub const FACTOR: f32 = 1_000_000.0;

    pub fn update(&mut self, signed_distance: f32, collider_index: u32) {
        let unsigned_distance = signed_distance.abs();
        let affinity = 1;
        let tag = if signed_distance >= 0.0 { 1 } else { 0 };
        let color = ((affinity << 1) | tag) << (collider_index << 1);

        let integer_unsigned_distance = (unsigned_distance * Self::FACTOR) as u32;

        unsafe {
            self.min_unsigned_distance
                .global_red_min(integer_unsigned_distance);
            self.color.global_red_or(color);
            self.active.global_red_add(1);
        }
    }

    pub fn unsigned_distance(&self) -> f32 {
        self.min_unsigned_distance as f32 / Self::FACTOR
    }

    pub fn color(&self, collider_index: u32) -> (u32, u32) {
        let collider_color = self.color >> (collider_index << 1);
        let affinity = (collider_color >> 1) & 1;
        let tag = collider_color & 1;

        (affinity, tag)
    }
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct GpuGridNode {
    pub mass: Real,
    // That’s where the particles transfer their momentum.
    // This is then replaced by the velocity during the grid update.
    pub momentum_velocity: Vector<Real>,
    // That’s where the particles transfer their momentum.
    // This is then replaced by the velocity during the grid update.
    pub psi_momentum_velocity: Real,
    pub psi_mass: Real,
    pub prev_mass: Real,
    pub projection_status: GpuGridProjectionStatus,
    pub projection_scaled_dir: Vector<Real>,
    pub cdf_data: CdfData,
}

impl Default for GpuGridNode {
    fn default() -> Self {
        Self {
            mass: 0.0,
            momentum_velocity: Vector::zeros(),
            psi_momentum_velocity: 0.0,
            psi_mass: 0.0,
            prev_mass: 0.0,
            projection_status: GpuGridProjectionStatus::NotComputed,
            projection_scaled_dir: Vector::zeros(),
            cdf_data: CdfData::default(),
        }
    }
}

#[cfg_attr(not(target_os = "cuda"), derive(cust::DeviceCopy))]
#[derive(Copy, Clone)]
#[repr(C)]
pub struct HaloBlockData {
    pub virtual_id: BlockVirtualId,
    pub cells: [GpuGridNode; NUM_CELL_PER_BLOCK as usize],
}

impl Default for HaloBlockData {
    fn default() -> Self {
        Self {
            virtual_id: BlockVirtualId(0),
            cells: [GpuGridNode::default(); NUM_CELL_PER_BLOCK as usize],
        }
    }
}
