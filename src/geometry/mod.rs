pub use self::sp_grid::SpGrid;

mod sp_grid;

use crate::math::Vector;
use na::vector;

#[cfg(feature = "dim2")]
pub(self) const NBH_SHIFTS: [Vector<usize>; 9] = [
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
pub(self) const NBH_SHIFTS: [Vector<usize>; 27] = [
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
