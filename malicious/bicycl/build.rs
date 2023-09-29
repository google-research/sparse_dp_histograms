// Copyright 2023 Google LLC
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

use std::path::PathBuf;

fn main() {
    // Locate headers and library of the GMP library provided by the gmp-mpfr-sys
    // crate.
    let gmp_include_dir = std::env::var("DEP_GMP_INCLUDE_DIR").unwrap();
    let gmp_lib_dir = std::env::var("DEP_GMP_LIB_DIR").unwrap();
    // Path to the directory where Cargo.toml is located.
    let cargo_manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    // Compute path to the bicycl submodule.
    let bicycl_path = cargo_manifest_dir.join("../../third_party/bicycl/src");

    // Compile the cxx bridge code interfacing with C++.
    cxx_build::bridges(["src/mpz.rs"])
        // Constant taken from the BICYCL code.
        .define("BICYCL_GMP_PRIMALITY_TESTS_ITERATION", "30")
        // We need access to the GMP headers.
        .include(gmp_include_dir)
        // We need access to the BICYCL headers.
        .include(bicycl_path)
        // Compile this crate.
        .compile("bicycl");

    // We need to link against the GMP library.
    println!("cargo:rustc-link-search={}", gmp_lib_dir);

    // We want to recompile whenever the C++ helper code is changed.
    println!("cargo:rerun-if-changed=cxx/helpers.h");
    println!("cargo:rerun-if-changed=cxx/trycatch.h");
}
