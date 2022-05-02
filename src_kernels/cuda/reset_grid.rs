use crate::gpu_grid::GpuGrid;
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
        cell.mass = 0.0;
        cell.momentum_velocity.fill(0.0);
        cell.psi_momentum_velocity = 0.0;
        cell.psi_mass = 0.0;
    }
}
