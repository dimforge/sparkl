#[cfg(feature = "build-ptx")]
fn main() {
    use cuda_builder::{CudaBuilder, NvvmArch};

    println!("cargo:rerun-if-changed={}", "../src_kernels");

    let mut builder = CudaBuilder::new("../sparkl2d-kernels")
        .copy_to("../../resources/sparkl2d-kernels.ptx")
        .emit_llvm_ir(true);
    builder.arch = NvvmArch::Compute70;
    builder.build().unwrap();
}

#[cfg(not(feature = "build-ptx"))]
fn main() {}
