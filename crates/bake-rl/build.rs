use std::{env, path::Path};

fn main() {
    let lib = env::var("DEP_TCH_LIBTORCH_LIB").unwrap_or_else(|_| {
        panic!(
            "DEP_TCH_LIBTORCH_LIB is not set. \
             Add `torch-sys` as a direct dependency of this package \
             (only direct dependents of a `links` package receive DEP_* variables)."
        )
    });

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");

    if Path::new(&lib).join("libtorch_cuda.so").exists() {
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-arg=-ltorch_cuda");
    }
}