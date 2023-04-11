#![no_std]
#![cfg_attr(
    target_os = "cuda",
    feature(register_attr, bench_black_box, asm_experimental_arch),
    register_attr(nvvm_internal)
)]
#![cfg_attr(target_os = "cuda", feature(core_intrinsics))]
#![cfg_attr(target_os = "cuda", feature(asm))]
#![cfg_attr(target_os = "cuda", feature(const_float_bits_conv))]
#![cfg_attr(target_os = "cuda", feature(int_log))]

#[cfg(all(feature = "dim2"))]
extern crate parry2d as parry;
#[cfg(all(feature = "dim3"))]
extern crate parry3d as parry;

#[cfg(all(feature = "dim2"))]
extern crate sparkl2d_core as sparkl_core;
#[cfg(all(feature = "dim3"))]
extern crate sparkl3d_core as sparkl_core;

extern crate nalgebra as na;

pub use self::gpu_cdf::*;
pub use self::gpu_constitutive_model::*;
pub use self::gpu_grid::*;
pub use self::gpu_particle_model::*;
pub use self::gpu_plastic_model::*;
pub use self::gpu_rigid_world::*;
pub use self::gpu_timestep::*;

pub mod cuda;
mod gpu_cdf;
mod gpu_constitutive_model;
mod gpu_grid;
mod gpu_particle_model;
mod gpu_plastic_model;
mod gpu_rigid_world;
mod gpu_timestep;

pub use parry::utils::DevicePointer;
