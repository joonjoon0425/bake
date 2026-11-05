fn main() {
    let lib = std::env::var("DEP_TCH_LIBTORCH_LIB").unwrap();
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-rpath,{lib}");
    println!("cargo:rustc-link-arg=-ltorch_cuda");
}