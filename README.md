# `bake`: My small lab

# Warning
When using a CUDA device with `tch`, binaries may fail with
"Could not run '...' with arguments from the 'CUDA' backend",
because the linker drops `libtorch_cuda.so` (`--as-needed`).

To fix it, in the package that builds your binaries/examples/tests:

1. Add `torch-sys` as a **direct** dependency (`DEP_TCH_LIBTORCH_LIB` is only
   passed to direct dependents of `torch-sys`).
2. Add the following `build.rs` (copy of `crates/bake-rl/build.rs`).

```rust
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
```

`cargo:rustc-link-arg` only applies to the package's own binaries, examples,
tests, benches, and cdylibs, so each such package needs this `build.rs`.