fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    pyo3_build_config::add_libpython_rpath_link_args();
    pyo3_build_config::add_python_framework_link_args();
}