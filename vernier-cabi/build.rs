fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    std::fs::create_dir_all(format!("{crate_dir}/include")).ok();
    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(
            cbindgen::Config::from_file(format!("{crate_dir}/cbindgen.toml"))
                .expect("cbindgen.toml missing"),
        )
        .generate()
        .expect("cbindgen failed")
        .write_to_file(format!("{crate_dir}/include/vernier.h"));
}
