// build.rs — mlx-sys build script
//
// Orchestrates:
//   1. Lightweight cmake configure to trigger FetchContent (downloads mlx source)
//   2. Post-fetch source patches for macOS 26+ Metal shader compatibility
//   3. Full build via the `cmake` crate (preserving its SDK/toolchain/env magic)
//   4. Bindgen C→Rust FFI binding generation
//   5. Build-time version constant emission

extern crate cmake;

use cmake::Config;
use std::{env, fs, path::{Path, PathBuf}, process::Command};

/// Path to the forked mlx-c C wrapper.  Set MLX_C_DIR env var to override.
/// The fork points at Tribunus-dev/mlx.git (tag tribunus-v0.31.2) for
/// the C++ core, which adds output buffer hint support for IOSurface
/// materialization.
fn mlx_c_dir() -> String {
    std::env::var("MLX_C_DIR")
        .unwrap_or_else(|_| "/Users/user/Developer/GitHub/mlx-c-fork".to_string())
}

// ─── Corruption detection ───────────────────────────────────────────

/// Check if make_compiled_preamble.sh was corrupted by a previous patcher.
///
/// The old Python/Rust patcher could drop `depth_this`/`depth_next` and
/// `done` lines from the for-loop body, leaving a script that references
/// `HDRS_LIST` but can't be parsed by bash.  We detect this by checking
/// that the script has *either* the original word-split pattern (patchable)
/// *or* our own `SPACE_SAFE_PATCH` marker (already fixed).
fn is_preamble_corrupted(mlx_src: &Path) -> bool {
    let script = mlx_src.join("mlx/backend/metal/make_compiled_preamble.sh");
    if !script.exists() {
        return false;
    }

    let content = fs::read_to_string(&script).unwrap_or_default();

    // Already patched by us — not corrupted
    if content.contains("# SPACE_SAFE_PATCH") {
        return false;
    }
    // Has the original pattern we know how to patch — not corrupted
    if content.contains("declare -a HDRS_LIST=($HDRS)") {
        return false;
    }
    // Doesn't reference HDRS_LIST at all (different MLX version) — not our problem
    if !content.contains("HDRS_LIST") {
        return false;
    }
    // Has HDRS_LIST but in an unrecognisable form — corrupted
    true
}

// ─── Metal shader source patches (macOS 26+ compatibility) ──────────

/// Replace bf16.h with our version that gates `bfloat` behind
/// `__has_extension(metal_bfloat)` and falls back to `half`.
fn patch_bf16_header(mlx_src: &Path) {
    let target = mlx_src.join("mlx/backend/metal/kernels/bf16.h");
    if !target.exists() {
        return;
    }

    let content = fs::read_to_string(&target).unwrap_or_default();
    if content.contains("__has_extension(metal_bfloat)") {
        return; // idempotent
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let patch_file = manifest_dir.join("patches/bf16_patched.h");
    if patch_file.exists() {
        fs::copy(&patch_file, &target).expect("Failed to copy bf16_patched.h");
        println!("cargo:warning=Patched bf16.h for macOS 26+ compatibility");
    }
}

/// Wrap the entire bf16_math.h in
/// `#if __has_extension(metal_bfloat) ... #endif`
/// so the math overloads only compile when Metal actually provides `bfloat`.
fn patch_bf16_math_header(mlx_src: &Path) {
    let target = mlx_src.join("mlx/backend/metal/kernels/bf16_math.h");
    if !target.exists() {
        return;
    }

    let content = fs::read_to_string(&target).unwrap_or_default();
    if content.contains("__has_extension(metal_bfloat)") {
        return; // idempotent
    }

    let wrapped = format!(
        "#if __has_extension(metal_bfloat)\n\
         {}\n\
         #endif // __has_extension(metal_bfloat)\n",
        content
    );
    fs::write(&target, wrapped).expect("Failed to patch bf16_math.h");
    println!("cargo:warning=Patched bf16_math.h (conditional bfloat guard)");
}

/// Patch utils.h:
///   1. Add `using metal::vec;` after the logging.h include
///   2. Make `instantiate_float_limit(bfloat16_t)` conditional on bfloat support
fn patch_utils_header(mlx_src: &Path) {
    let target = mlx_src.join("mlx/backend/metal/kernels/utils.h");
    if !target.exists() {
        return;
    }

    let mut content = fs::read_to_string(&target).unwrap_or_default();
    let mut changed = false;

    // 1. Add `using metal::vec;`
    if !content.contains("using metal::vec") {
        content = content.replace(
            "#include \"mlx/backend/metal/kernels/logging.h\"",
            "#include \"mlx/backend/metal/kernels/logging.h\"\n\nusing metal::vec;",
        );
        changed = true;
    }

    // 2. Conditional bfloat16_t float limit
    if content.contains("instantiate_float_limit(bfloat16_t);")
        && !content.contains("#if __has_extension(metal_bfloat)")
    {
        content = content.replace(
            "instantiate_float_limit(bfloat16_t);",
            "#if __has_extension(metal_bfloat)\n\
             \x20   instantiate_float_limit(bfloat16_t);\n\
             #endif",
        );
        changed = true;
    }

    if changed {
        fs::write(&target, content).expect("Failed to patch utils.h");
        println!("cargo:warning=Patched utils.h for macOS 26+ compatibility");
    }
}

/// Fix the word-splitting bug in `make_compiled_preamble.sh` that breaks
/// when the build path contains spaces (e.g. "Tribunus Compute").
///
/// The original script uses:
///     declare -a HDRS_LIST=($HDRS)
/// which splits the Metal compiler's `-H` output on **all** whitespace.
/// Paths with spaces corrupt the array — depth/path pairs get misaligned
/// and the whole JIT generation fails.
///
/// The fix replaces that single line with a `while read` loop that splits
/// on newlines, then extracts depth (`${line%% *}`) and path (`${line#* }`)
/// from each line.  This preserves the exact same array layout (alternating
/// depth/path pairs) that the rest of the script expects.
fn patch_preamble_script(mlx_src: &Path) {
    let script = mlx_src.join("mlx/backend/metal/make_compiled_preamble.sh");
    if !script.exists() {
        return;
    }

    let content = fs::read_to_string(&script).unwrap_or_default();

    // Idempotent: already patched
    if content.contains("# SPACE_SAFE_PATCH") {
        return;
    }
    // Nothing to patch (different script version, or already handled)
    if !content.contains("declare -a HDRS_LIST=($HDRS)") {
        return;
    }

    let patched = content.replace(
        "declare -a HDRS_LIST=($HDRS)",
        "# SPACE_SAFE_PATCH: read line-by-line to handle spaces in paths\n\
         declare -a HDRS_LIST=()\n\
         while IFS= read -r _line; do\n\
         \x20 [[ -z \"$_line\" ]] && continue\n\
         \x20 HDRS_LIST+=(\"${_line%% *}\" \"${_line#* }\")\n\
         done <<< \"$HDRS\"",
    );

    fs::write(&script, &patched).expect("Failed to patch make_compiled_preamble.sh");
    println!("cargo:warning=Patched make_compiled_preamble.sh for space-safe paths");
}

/// Strip `bfloat` instantiations from .metal files and `#if` guard `operator bfloat16_t` in .h files.
/// Fixes duplicate instantiation errors on macOS 26+ where `bfloat16_t` == `half`.
fn patch_bfloat_instantiations(mlx_src: &Path) {
    let kernels_dir = mlx_src.join("mlx/backend/metal/kernels");
    if !kernels_dir.exists() {
        return;
    }

    // 1. Fix fp4.h
    let fp4 = kernels_dir.join("fp4.h");
    if fp4.exists() {
        let content = fs::read_to_string(&fp4).unwrap_or_default();
        if content.contains("operator bfloat16_t()") && !content.contains("#if __has_extension(metal_bfloat)") {
            let patched = content.replace(
                "  operator bfloat16_t() {\n    return static_cast<bfloat16_t>(this->operator float16_t());\n  }",
                "#if __has_extension(metal_bfloat)\n  operator bfloat16_t() {\n    return static_cast<bfloat16_t>(this->operator float16_t());\n  }\n#endif"
            );
            fs::write(&fp4, patched).expect("Failed to patch fp4.h");
            println!("cargo:warning=Patched fp4.h for bfloat16_t conflict");
        }
    }

    // 2. Fix fp8.h
    let fp8 = kernels_dir.join("fp8.h");
    if fp8.exists() {
        let content = fs::read_to_string(&fp8).unwrap_or_default();
        if content.contains("operator bfloat16_t()") && !content.contains("#if __has_extension(metal_bfloat)") {
            let mut patched = content.replace(
                "  operator bfloat16_t() {\n    return static_cast<bfloat16_t>(this->operator float16_t());\n  }",
                "#if __has_extension(metal_bfloat)\n  operator bfloat16_t() {\n    return static_cast<bfloat16_t>(this->operator float16_t());\n  }\n#endif"
            );
            patched = patched.replace(
                "  operator bfloat16_t() {\n    uint16_t out = (bits == 0 ? 0x40 : (static_cast<uint16_t>(bits) << 7));\n    return as_type<bfloat16_t>(out);\n  }",
                "#if __has_extension(metal_bfloat)\n  operator bfloat16_t() {\n    uint16_t out = (bits == 0 ? 0x40 : (static_cast<uint16_t>(bits) << 7));\n    return as_type<bfloat16_t>(out);\n  }\n#endif"
            );
            fs::write(&fp8, patched).expect("Failed to patch fp8.h");
            println!("cargo:warning=Patched fp8.h for bfloat16_t conflict");
        }
    }

    // 3. Fix fp_quantized_nax.h Wtype
    let fp_nax = kernels_dir.join("fp_quantized_nax.h");
    if fp_nax.exists() {
        let content = fs::read_to_string(&fp_nax).unwrap_or_default();
        if content.contains("Wtype = bfloat>") {
            let patched = content.replace("Wtype = bfloat>", "Wtype = bfloat16_t>");
            fs::write(&fp_nax, patched).expect("Failed to patch fp_quantized_nax.h");
            println!("cargo:warning=Patched fp_quantized_nax.h for bfloat16_t conflict");
        }
    }

    // 4. Fix binary_ops.h FloorDivide bfloat overload
    let bin_ops = kernels_dir.join("binary_ops.h");
    if bin_ops.exists() {
        let content = fs::read_to_string(&bin_ops).unwrap_or_default();
        if content.contains("bfloat16_t operator()(bfloat16_t x, bfloat16_t y)") && !content.contains("#if __has_extension(metal_bfloat)") {
            let patched = content.replace(
                "  template <>\n  bfloat16_t operator()(bfloat16_t x, bfloat16_t y) {\n    return trunc(x / y);\n  }",
                "#if __has_extension(metal_bfloat)\n  template <>\n  bfloat16_t operator()(bfloat16_t x, bfloat16_t y) {\n    return trunc(x / y);\n  }\n#endif"
            );
            fs::write(&bin_ops, patched).expect("Failed to patch binary_ops.h");
            println!("cargo:warning=Patched binary_ops.h for bfloat16_t conflict");
        }
    }

    // 5. Fix device.cpp dictionary syntax error
    let device_cpp = mlx_src.join("mlx/backend/metal/device.cpp");
    if device_cpp.exists() {
        let content = fs::read_to_string(&device_cpp).unwrap_or_default();
        if content.contains("NS::Dictionary::dictionary(macro_key, macro_val, nullptr)") {
            let patched = content.replace(
                "NS::Dictionary::dictionary(macro_key, macro_val, nullptr)",
                "NS::Dictionary::dictionary(macro_val, macro_key)"
            );
            fs::write(&device_cpp, patched).expect("Failed to patch device.cpp");
            println!("cargo:warning=Patched device.cpp for dictionary syntax");
        }
    }

    // 6. Strip bfloat instantiations from all .metal files
    fn visit_metal_files(dir: &Path, cb: &mut dyn FnMut(&Path)) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    visit_metal_files(&path, cb);
                } else if path.extension().map_or(false, |e| e == "metal") {
                    cb(&path);
                }
            }
        }
    }

    visit_metal_files(&kernels_dir, &mut |path: &Path| {
        let original_content = fs::read_to_string(path).unwrap_or_default();
        if !original_content.contains("bfloat") {
            return;
        }

        // Handle the specific multi-line instantiation in steel_gemm_segmented.metal
        let content = original_content.replace(
            "instantiate_segmented_mm_shapes_helper(\n    bfloat16,\n    bfloat16_t,\n    bfloat16,\n    bfloat16_t);",
            ""
        );

        let mut new_lines: Vec<String> = Vec::new();
        for line in content.lines() {
            if line.contains("bfloat") {
                let trimmed = line.trim_end();
                if !trimmed.ends_with('\\') {
                    // This bfloat line is the end of a macro or not a macro.
                    // Strip '\\' from the previous line so it doesn't continue.
                    if let Some(last) = new_lines.last_mut() {
                        let last_trimmed = last.trim_end();
                        if last_trimmed.ends_with('\\') {
                            *last = last_trimmed[..last_trimmed.len() - 1].trim_end().to_string();
                        }
                    }
                }
                continue; // Drop the bfloat line
            }
            new_lines.push(line.to_string());
        }

        let new_content = new_lines.join("\n") + "\n";
        if new_content != original_content {
            fs::write(path, new_content).expect("Failed to patch .metal file");
            if let Some(name) = path.file_name() {
                println!("cargo:warning=Stripped bfloat instantiations from {:?}", name);
            }
        }
    });
}

// ─── Patch orchestrator ─────────────────────────────────────────────

/// Ensure MLX source is fetched and patched **before** the cmake crate builds.
///
/// Strategy:
///   1. If the source tree is corrupted (by a previous bad patcher), nuke
///      `_deps/` and the cmake cache so FetchContent re-downloads cleanly.
///   2. If `_deps/mlx-src` doesn't exist yet, run a lightweight
///      `cmake -S ... -B ...` to trigger FetchContent.  This may fail on
///      the full C++ configuration (missing flags the cmake crate would set)
///      but FetchContent runs early enough in configure to succeed.
///   3. Apply all idempotent patches to the fetched source.
///   4. Let the cmake crate handle the *real* build with all its
///      cross-platform SDK/toolchain/CFLAGS magic intact.
fn ensure_patched_mlx_source(mlx_c_dir_str: &str, out_dir: &Path) {
    let build_dir = out_dir.join("build");
    let mlx_src = build_dir.join("_deps").join("mlx-src");

    // ── Corruption recovery ──
    if is_preamble_corrupted(&mlx_src) {
        println!(
            "cargo:warning=Detected corrupted make_compiled_preamble.sh \
             from a previous patcher; forcing clean re-fetch"
        );
        // Remove the entire _deps tree and cmake cache to force FetchContent
        // to re-download a pristine source tree.
        fs::remove_dir_all(build_dir.join("_deps")).ok();
        fs::remove_file(build_dir.join("CMakeCache.txt")).ok();
        fs::remove_dir_all(build_dir.join("CMakeFiles")).ok();
    }

    // ── Phase 1: Trigger FetchContent if source tree absent ──
    if !mlx_src.exists() {
        println!("cargo:warning=Triggering cmake configure to fetch MLX source...");
        let _ = Command::new("cmake")
            .arg("-S")
            .arg(mlx_c_dir_str)
            .arg("-B")
            .arg(&build_dir)
            .status();
        // We intentionally ignore the exit status: the configure may fail on
        // C++ compiler setup (which the cmake crate handles properly), but
        // FetchContent runs early enough that the source tree will exist.
    }

    // ── Phase 2: Apply idempotent patches ──
    if mlx_src.exists() {
        patch_bf16_header(&mlx_src);
        patch_bf16_math_header(&mlx_src);
        patch_utils_header(&mlx_src);
        patch_preamble_script(&mlx_src);
        patch_bfloat_instantiations(&mlx_src);
        // CPU preamble already quotes its variables; no patch needed.
    } else {
        println!(
            "cargo:warning=_deps/mlx-src not found after configure; \
             the cmake crate will fetch it on the full build"
        );
    }
}

// ─── CMake build (via cmake crate) ──────────────────────────────────

fn build_and_link_mlx_c() {
    let dir = mlx_c_dir();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // ── Pre-build: fetch source and apply patches ──
    ensure_patched_mlx_source(&dir, &out_dir);

    // ── Build via cmake crate ──
    // The cmake crate handles Apple SDK detection, CFLAGS, target arch,
    // MSVC flags, and other cross-platform magic that raw Command would lose.
    let mut config = Config::new(&dir);
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
