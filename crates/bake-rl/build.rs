fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "gym")]
    pyo3_build_config::add_libpython_rpath_link_args();
    #[cfg(feature = "gym")]
    pyo3_build_config::add_python_framework_link_args();
}