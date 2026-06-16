extern crate cmake;

use cmake::Config;
use std::{env, path::PathBuf};

/// Path to the forked mlx-c C wrapper.  Set MLX_C_DIR env var to override.
/// The fork points at Tribunus-dev/mlx.git (tag tribunus-v0.31.2) for
/// the C++ core, which adds output buffer hint support for IOSurface
/// materialization.
fn mlx_c_dir() -> String {
    std::env::var("MLX_C_DIR")
        .unwrap_or_else(|_| "/Users/user/Developer/GitHub/mlx-c-fork".to_string())
}

fn build_and_link_mlx_c() {
    let mut config = Config::new(mlx_c_dir());
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    #[cfg(debug_assertions)]
    {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");

    #[cfg(feature = "metal")]
    {
        config.define("MLX_BUILD_METAL", "ON");
    }

    #[cfg(feature = "accelerate")]
    {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // build the mlx-c project
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}

fn main() {
    build_and_link_mlx_c();

    let dir = mlx_c_dir();

    // generate bindings
    let bindings = bindgen::Builder::default()
        .rust_target("1.73.0".parse().expect("rust-version"))
        .header(format!("{}/mlx/c/mlx.h", dir))
        .header(format!("{}/mlx/c/linalg.h", dir))
        .header(format!("{}/mlx/c/error.h", dir))
        .header(format!("{}/mlx/c/transforms_impl.h", dir))
        .clang_arg(format!("-I{}", dir))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // Emit build-generated version constants
    let mlx_c_version = std::fs::read_to_string(format!("{}/VERSION", dir))
        .unwrap_or_else(|_| "0.6.0".to_string())
        .trim()
        .to_string();
    println!("cargo:rustc-env=MLX_C_VERSION={}", mlx_c_version);
    println!("cargo:rustc-env=MLX_CORE_TARGET=v0.31.2");
    println!("cargo:rustc-env=MLX_SYS_VERSION=0.6.0-tribunus.1");
    println!("cargo:rustc-env=MLX_RS_BASE_COMMIT=93ed8db");
}
