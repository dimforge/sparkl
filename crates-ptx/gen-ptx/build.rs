fn main() {
    use cuda_builder::{CudaBuilder, NvvmArch};
    println!("cargo:rerun-if-changed={}", "build.rs");
    let mut builder = CudaBuilder::new("../sparkl3d-kernels-ptx")
        .copy_to("../../resources/sparkl3d-kernels.ptx")
        .emit_llvm_ir(true);
    builder.arch = NvvmArch::Compute70;

    builder.build().unwrap();

    let mut builder = CudaBuilder::new("../sparkl2d-kernels-ptx")
        .copy_to("../../resources/sparkl2d-kernels.ptx")
        .emit_llvm_ir(true);
    builder.arch = NvvmArch::Compute70;
    builder.build().unwrap();
}
