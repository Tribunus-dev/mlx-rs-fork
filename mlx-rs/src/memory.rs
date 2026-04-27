//! GPU memory management.
//!
//! These functions wrap the global memory-management APIs from `mlx-c`. They
//! affect the active device, which on Apple Silicon is the unified-memory
//! Metal allocator.
//!
//! # Example
//!
//! ```rust,no_run
//! use mlx_rs::memory;
//!
//! // Cap MLX's GPU allocator at 8 GiB. Returns the previous limit.
//! let prev = memory::set_memory_limit(8 * 1024 * 1024 * 1024).unwrap();
//! println!("previous limit: {} bytes", prev);
//!
//! // Read the current limit back.
//! let cur = memory::get_memory_limit().unwrap();
//! println!("current limit:  {} bytes", cur);
//! ```

use std::panic::Location;

use crate::error::{Exception, Result};

/// Get the current memory limit, in bytes.
///
/// This is the cap that MLX's allocator will keep its working set under.
/// Allocations beyond the limit will block until scheduled tasks free memory.
/// The default limit is 1.5× the device's recommended max working-set size.
#[track_caller]
pub fn get_memory_limit() -> Result<usize> {
    init_error_handler();
    let mut current: usize = 0;
    let status = unsafe { mlx_sys::mlx_get_memory_limit(&mut current as *mut usize) };
    if status != 0 {
        return Err(last_error_or_unknown(
            "mlx_get_memory_limit failed without setting an error",
        ));
    }
    Ok(current)
}

/// Set the memory limit, in bytes. Returns the previous limit.
///
/// Calls to allocate will block on scheduled tasks if the limit is exceeded.
/// The limit defaults to 1.5× the device's recommended max working-set size,
/// so you usually only need this to lower the cap (e.g. when sharing a
/// machine with other GPU workloads).
#[track_caller]
pub fn set_memory_limit(limit: usize) -> Result<usize> {
    init_error_handler();
    let mut previous: usize = 0;
    let status = unsafe { mlx_sys::mlx_set_memory_limit(&mut previous as *mut usize, limit) };
    if status != 0 {
        return Err(last_error_or_unknown(
            "mlx_set_memory_limit failed without setting an error",
        ));
    }
    Ok(previous)
}

#[inline]
fn init_error_handler() {
    crate::error::INIT_ERR_HANDLER
        .with(|init| init.call_once(crate::error::setup_mlx_error_handler));
}

#[track_caller]
fn last_error_or_unknown(fallback_msg: &'static str) -> Exception {
    match crate::error::get_and_clear_last_mlx_error() {
        Some(raw) => Exception::from(raw),
        None => Exception {
            what: fallback_msg.to_string(),
            location: Location::caller(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Memory limit is global state on the active device, so we exercise
    // get/set + the "returns previous" contract in a single test to avoid
    // races with parallel test runners that could interleave writes.
    #[test]
    fn test_get_and_set_memory_limit() {
        let original = get_memory_limit().unwrap();

        // 1 GiB and 2 GiB are well under the default cap of 1.5x the
        // device's recommended working set, so they should fit on every
        // machine that can run MLX at all.
        let a = 1usize << 30;
        let b = 2usize << 30;

        let prev_a = set_memory_limit(a).unwrap();
        assert_eq!(prev_a, original, "first set should return the prior limit");
        assert_eq!(
            get_memory_limit().unwrap(),
            a,
            "get should reflect the most recent set"
        );

        let prev_b = set_memory_limit(b).unwrap();
        assert_eq!(
            prev_b, a,
            "consecutive set should return the immediately previous limit"
        );
        assert_eq!(get_memory_limit().unwrap(), b);

        // Restore.
        set_memory_limit(original).unwrap();
        assert_eq!(get_memory_limit().unwrap(), original);
    }
}
