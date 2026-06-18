use std::{env, path::PathBuf, process::Command};

fn build_and_link_mlx_c() {
    // build the mlx-c project
    // Step 1: cmake configure (fetches sources, creates build tree)
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mlx_c_src = manifest_dir.join("src/mlx-c");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_dir = out_dir.join("build");
    let install_prefix = out_dir.join("install");

    let mut cmake_args = vec![
        "-S".to_string(), mlx_c_src.to_str().unwrap().to_string(),
        "-B".to_string(), build_dir.to_str().unwrap().to_string(),
        format!("-DCMAKE_INSTALL_PREFIX={}", install_prefix.to_str().unwrap()),
        "-DMLX_BUILD_METAL=OFF".to_string(),
        "-DMLX_BUILD_ACCELERATE=OFF".to_string(),
    ];

    #[cfg(debug_assertions)]
    { cmake_args.push("-DCMAKE_BUILD_TYPE=Debug".to_string()); }
    #[cfg(not(debug_assertions))]
    { cmake_args.push("-DCMAKE_BUILD_TYPE=Release".to_string()); }
    #[cfg(feature = "metal")]
    { cmake_args.push("-DMLX_BUILD_METAL=ON".to_string()); }
    #[cfg(feature = "accelerate")]
    { cmake_args.push("-DMLX_BUILD_ACCELERATE=ON".to_string()); }

    let status = Command::new("cmake")
        .args(&cmake_args)
        .status()
        .expect("failed to run cmake configure");
    if !status.success() {
        panic!("cmake configure failed");
    }

    // Patch bf16_math.h: guard half-typed instantiations on macOS 26+
    // where bfloat16_t falls back to `half` and Metal already provides
    // native half math functions.
    let bf16_math_path = build_dir.join("_deps/mlx-src/mlx/backend/metal/kernels/bf16_math.h");
    if bf16_math_path.exists() {
        let content = std::fs::read_to_string(&bf16_math_path).unwrap_or_default();
        let guarded = content.replace(
            "// Xcode 26.5+ Metal SDK provides bfloat16 math natively — skip.\n#if __METAL_VERSION__ < 310000",
            "// Xcode 26.5+ Metal SDK provides bfloat16 math natively — skip.\n// Also skip when bfloat16_t == half (macOS 26 fallback).\n#if __has_extension(metal_bfloat) && __METAL_VERSION__ < 310000",
        );
        if content != guarded {
            std::fs::write(&bf16_math_path, &guarded).unwrap();
            eprintln!("Patched bf16_math.h for macOS 26+ compatibility (half guard)");
        } else {
            eprintln!("NOT patched: {} does not exist", bf16_math_path.display());
        }
    } else {
        eprintln!("NOT patched: {} does not exist", bf16_math_path.display());
    }

    // Step 2: build (make)
    let status = Command::new("cmake")
        .args(["--build", build_dir.to_str().unwrap()])
        .args(["-j", "8"])
        .args(["--target", "install"])
        .status()
        .expect("failed to run cmake --build");
    if !status.success() {
        panic!("cmake --build failed");
    }

    println!("cargo:rustc-link-search=native={}/build/lib", build_dir.display());
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");

    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=openblas");
        println!("cargo:rustc-link-lib=lapack");
        println!("cargo:rustc-link-lib=lapacke");
    } else {
        println!("cargo:rustc-link-lib=c++");
    }
    println!("cargo:rustc-link-lib=dylib=objc");
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" || std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "ios" {
        println!("cargo:rustc-link-lib=framework=Foundation");
    }

    #[cfg(feature = "metal")]
    {
        if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" || std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "ios" {
            println!("cargo:rustc-link-lib=framework=Metal");
        }
    }

    #[cfg(feature = "accelerate")]
    {
        if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" || std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "ios" {
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
    }
}

fn main() {
    #[cfg(not(feature = "stub"))]
    {
        build_and_link_mlx_c();

        // generate bindings
        let bindings = bindgen::Builder::default()
            .rust_target("1.73.0".parse().expect("rust-version"))
            .header("src/mlx-c/mlx/c/mlx.h")
            .header("src/mlx-c/mlx/c/linalg.h")
            .header("src/mlx-c/mlx/c/error.h")
            .header("src/mlx-c/mlx/c/transforms_impl.h")
            .clang_arg("-Isrc/mlx-c")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate bindings");

        // Write the bindings to the $OUT_DIR/bindings.rs file.
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings!");
    }

    #[cfg(feature = "stub")]
    {
        // Write a dummy bindings file so the crate compiles
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        let dummy_bindings = r#"
pub type mlx_array = *mut std::ffi::c_void;
pub type mlx_stream = *mut std::ffi::c_void;

#[repr(C)]
pub struct mlx_optional_int_ {
    pub value: i32,
    pub has_value: bool,
}

#[repr(C)]
pub struct mlx_optional_dtype_ {
    pub value: i32,
    pub has_value: bool,
}

#[no_mangle]
pub unsafe extern "C" fn mlx_get_active_memory(_res: *mut usize) {}
#[no_mangle]
pub unsafe extern "C" fn mlx_get_cache_memory(_res: *mut usize) {}
#[no_mangle]
pub unsafe extern "C" fn mlx_get_peak_memory(_res: *mut usize) {}
#[no_mangle]
pub unsafe extern "C" fn mlx_clear_cache() {}
#[no_mangle]
pub unsafe extern "C" fn mlx_set_cache_limit(_prev: *mut usize, _limit: usize) {}
#[no_mangle]
pub unsafe extern "C" fn mlx_get_memory_limit(_res: *mut usize) {}
#[no_mangle]
pub unsafe extern "C" fn mlx_set_memory_limit(_prev: *mut usize, _limit: usize) {}
#[no_mangle]
pub unsafe extern "C" fn mlx_metal_is_available(_res: *mut bool) -> i32 { 0 }
#[no_mangle]
pub unsafe extern "C" fn mlx_reshape_ffi(_x: mlx_array, _shape_ar: *const i32, _ndim: i32) -> mlx_array { std::ptr::null_mut() }
#[no_mangle]
pub unsafe extern "C" fn mlx_transpose_ffi(_x: mlx_array, _axes: *const i32, _n_axes: i32) -> mlx_array { std::ptr::null_mut() }
#[no_mangle]
pub unsafe extern "C" fn mlx_slice_ffi(_x: mlx_array, _start: *const i32, _stop: *const i32, _stride: *const i32, _n_axes: i32) -> mlx_array { std::ptr::null_mut() }
#[no_mangle]
pub unsafe extern "C" fn mlx_concatenate_ffi(_arrays: *const mlx_array, _n_arrays: i32, _axis: i32) -> mlx_array { std::ptr::null_mut() }
#[no_mangle]
pub unsafe extern "C" fn mlx_pad_ffi(_x: mlx_array, _pad_widths: *const i32, _n_pads: i32) -> mlx_array { std::ptr::null_mut() }
#[no_mangle]
pub unsafe extern "C" fn mlx_array_new_data_managed_payload(
    _data: *const std::ffi::c_void, 
    _shape: *const i32, 
    _dim: i32, 
    _dtype: u32, 
    _payload: *mut std::ffi::c_void, 
    _dtor: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>
) -> mlx_array { std::ptr::null_mut() }
#[no_mangle]
pub unsafe extern "C" fn mlx_fast_scaled_dot_product_attention(
    _res: *mut mlx_array,
    _q: mlx_array,
    _k: mlx_array,
    _v: mlx_array,
    _scale: f32,
    _mask_mode: *const std::ffi::c_char,
    _mask_arr: mlx_array,
    _sinks: mlx_array,
    _stream: mlx_stream,
) -> i32 {
    0
}
#[no_mangle]
pub unsafe extern "C" fn mlx_array_new() -> mlx_array { std::ptr::null_mut() }
"#;
        std::fs::write(out_path.join("bindings.rs"), dummy_bindings).expect("dummy bindings");
    }

    // Emit build-generated version constants
    let mlx_c_version = std::fs::read_to_string("src/mlx-c/VERSION")
        .unwrap_or_else(|_| "0.6.0".to_string())
        .trim()
        .to_string();
    println!("cargo:rustc-env=MLX_C_VERSION={}", mlx_c_version);
    println!("cargo:rustc-env=MLX_CORE_TARGET=v0.31.2");
    println!("cargo:rustc-env=MLX_SYS_VERSION=0.6.0-tribunus.1");
    println!("cargo:rustc-env=MLX_RS_BASE_COMMIT=93ed8db");
}
