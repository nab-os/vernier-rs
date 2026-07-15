fn main() {
    let cuda_root = std::env::var("CUDA_ROOT")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:rustc-link-search=native={}/lib64", cuda_root);
    println!("cargo:rustc-link-search=native={}/lib/x64", cuda_root);
    println!("cargo:rustc-link-lib=dylib=cufft");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
