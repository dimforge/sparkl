pub use self::atomic::{AtomicAdd, AtomicInt};
// pub use self::fluids_volume::{recompute_fluids_volume_g2p, recompute_fluids_volume_p2g};
pub use self::g2p2g::{g2p2g, g2p2g_generic, InterpolatedParticleData};
pub use self::grid_update::grid_update;
pub use self::hashmap::{GridHashMap, GridHashMapEntry};
pub use self::particle_updater::{DefaultParticleUpdater, ParticleUpdater};
pub use self::prefix_sum::prefix_sum_512;
pub use self::reset_grid::reset_grid;
pub use self::sort::{ActiveBlockHeader, HaloState};
pub use self::timestep::{estimate_timestep_length, estimate_timestep_length_generic};
pub use self::update_cdf::update_cdf;

#[cfg(feature = "dim2")]
pub const G2P2G_THREADS: usize = 64;
#[cfg(feature = "dim3")]
pub const G2P2G_THREADS: usize = 128;

mod atomic;
// mod fluids_volume;
mod g2p2g;
mod grid_update;
mod hashmap;
mod particle_updater;
mod prefix_sum;
mod reset_grid;
mod sort;
mod timestep;
mod update_cdf;
