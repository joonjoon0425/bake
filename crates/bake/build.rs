fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_FEATURE_DEEP").is_some() {
        pyo3_build_config::add_libpython_rpath_link_args();
        pyo3_build_config::add_python_framework_link_args();
    }
}