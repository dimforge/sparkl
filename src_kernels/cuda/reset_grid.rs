use crate::gpu_grid::{GpuGrid, GpuGridNode};
use crate::BlockHeaderId;
use cuda_std::thread;
use cuda_std::*;
use na::vector;

// NOTE: the number of threads must be 4x4x4 (3D) or 4x4 (2D)
#[kernel]
pub unsafe fn reset_grid(mut next_grid: GpuGrid) {
    let bid = BlockHeaderId(thread::block_idx_x());
    #[cfg(feature = "dim2")]
    let shift = vector![
        thread::thread_idx_x() as usize,
        thread::thread_idx_y() as usize
    ];
    #[cfg(feature = "dim3")]
    let shift = vector![
        thread::thread_idx_x() as usize,
        thread::thread_idx_y() as usize,
        thread::thread_idx_z() as usize
    ];

    let node_id = bid.to_physical().node_id_unchecked(shift);
    if let Some(cell) = next_grid.get_node_mut(node_id) {
        *cell = GpuGridNode::default();
    }
}

// NOTE: the number of threads must be 4x4x4 (3D) or 4x4 (2D)
#[kernel]
pub unsafe fn copy_grid_projection_data(prev_grid: GpuGrid, mut next_grid: GpuGrid) {
    let next_bid = BlockHeaderId(thread::block_idx_x());
    #[cfg(feature = "dim2")]
    let shift = vector![
        thread::thread_idx_x() as usize,
        thread::thread_idx_y() as usize
    ];
    #[cfg(feature = "dim3")]
    let shift = vector![
        thread::thread_idx_x() as usize,
        thread::thread_idx_y() as usize,
        thread::thread_idx_z() as usize
    ];

    let next_block = next_grid.active_block_unchecked(next_bid);
    if let Some(prev_bid) = prev_grid.get_header_block_id(next_block.virtual_id) {
        let next_node_id = next_bid.to_physical().node_id_unchecked(shift);
        let prev_node_id = prev_bid.to_physical().node_id_unchecked(shift);

        if let (Some(prev_cell), Some(next_cell)) = (
            prev_grid.get_node(prev_node_id),
            next_grid.get_node_mut(next_node_id),
        ) {
            next_cell.prev_mass = prev_cell.mass;
            next_cell.projection_status = prev_cell.projection_status;
            next_cell.projection_scaled_dir = prev_cell.projection_scaled_dir;
        }
    }
}
