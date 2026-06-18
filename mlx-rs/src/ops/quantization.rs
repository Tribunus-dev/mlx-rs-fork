use std::ffi::CStr;

use mlx_internal_macros::{default_device, generate_macro};

use crate::{
    error::Result,
    utils::{guard::Guarded, VectorArray},
    Array, Stream,
};

/// Quantize the matrix `w` using `bits` bits per element.
///
/// Note, every `group_size` elements in a row of `w` are quantized together. Hence, number of
/// columns of `w` should be divisible by `group_size`. In particular, the rows of `w` are divided
/// into groups of size `group_size` which are quantized together.
///
/// > `quantized` currently only supports 2D inputs with dimensions which are multiples of 32
///
/// Default group size for quantization (64 elements per group).
const DEFAULT_GROUP_SIZE: i32 = 64;
/// Default number of bits per quantized element (4 bits).
const DEFAULT_BITS: i32 = 4;

/// Helper: unwrap an optional integer to a default value.
fn optional_int(v: impl Into<Option<i32>>, default: i32) -> i32 {
    v.into().unwrap_or(default)
}

/// Quantize the matrix `w` using `bits` bits per element.
///
/// Note, every `group_size` elements in a row of `w` are quantized together. Hence, number of
/// columns of `w` should be divisible by `group_size`. In particular, the rows of `w` are divided
/// into groups of size `group_size` which are quantized together.
///
/// > `quantized` currently only supports 2D inputs with dimensions which are multiples of 32
///
/// For details, please see [this
/// documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
#[generate_macro]
#[default_device]
pub fn quantize_device(
    w: impl AsRef<Array>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<(Array, Array, Array)> {
    let group_size = optional_int(group_size.into(), DEFAULT_GROUP_SIZE);
    let bits = optional_int(bits.into(), DEFAULT_BITS);

    let group_size = mlx_sys::mlx_optional_int_ {
        value: group_size,
        has_value: true,
    };
    let bits = mlx_sys::mlx_optional_int_ {
        value: bits,
        has_value: true,
    };

    let v = VectorArray::try_from_op(|res| unsafe {
        mlx_sys::mlx_quantize(
            res,
            w.as_ref().as_ptr(),
            group_size,
            bits,
            std::ffi::CStr::from_bytes_with_nul(b"affine\0")
                .unwrap()
                .as_ptr(),
            mlx_sys::mlx_array_new(),
            stream.as_ref().as_ptr(),
        )
    })?;

    let vals: Vec<Array> = v.try_into_values()?;
    let mut iter = vals.into_iter();
    let qw = iter.next().unwrap();
    let qs = iter.next().unwrap();
    let qb = iter.next().unwrap();

    Ok((qw, qs, qb))
}

/// Perform the matrix multiplication with the quantized matrix `w`. The quantization uses one
/// floating point scale and bias per `group_size` of elements. Each element in `w` takes `bits`
/// bits and is packed in an unsigned 32 bit integer.
#[allow(clippy::too_many_arguments)]
#[generate_macro]
#[default_device]
pub fn quantized_matmul_device<'a>(
    x: impl AsRef<Array>,
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
        #[optional] biases: impl Into<Option<&'a Array>>,
    #[optional] transpose: impl Into<Option<bool>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let transpose = transpose.into().unwrap_or(false);
    let group_size = optional_int(group_size.into(), DEFAULT_GROUP_SIZE);
    let bits = optional_int(bits.into(), DEFAULT_BITS);

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_quantized_matmul(
            res,
            x.as_ref().as_ptr(),
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            transpose,
            mlx_sys::mlx_optional_int_ {
                value: group_size,
                has_value: true,
            },
            mlx_sys::mlx_optional_int_ {
                value: bits,
                has_value: true,
            },
            std::ffi::CStr::from_bytes_with_nul(b"affine\0")
                .unwrap()
                .as_ptr(),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Dequantize the matrix `w` using the provided `scales` and `biases` and the `group_size` and
/// `bits` configuration.
/// For details, please see [this
/// documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html)
#[generate_macro]
#[default_device]
pub fn dequantize_device<'a>(
    w: impl AsRef<Array>,
    scales: impl AsRef<Array>,
        #[optional] biases: impl Into<Option<&'a Array>>,
    #[optional] group_size: impl Into<Option<i32>>,
    #[optional] bits: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let group_size = optional_int(group_size.into(), DEFAULT_GROUP_SIZE);
    let bits = optional_int(bits.into(), DEFAULT_BITS);

    <Array as Guarded>::try_from_op(|res| unsafe {
        mlx_sys::mlx_dequantize(
            res,
            w.as_ref().as_ptr(),
            scales.as_ref().as_ptr(),
            biases
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            mlx_sys::mlx_optional_int_ {
                value: group_size,
                has_value: true,
            },
            mlx_sys::mlx_optional_int_ {
                value: bits,
                has_value: true,
            },
            std::ffi::CStr::from_bytes_with_nul(b"affine\0")
                .unwrap()
                .as_ptr(),
            mlx_sys::mlx_array_new(),
            mlx_sys::mlx_optional_dtype_ {
                value: mlx_sys::mlx_dtype__MLX_FLOAT32,
                has_value: false,
            },
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        ops::{dequantize, expand_dims, quantize, quantized_matmul},
        random, Array,
    };

    #[test]
    fn test_quantize_dequantize() {
        let x1 = Array::ones::<f32>(&[128, 1]).unwrap();
        let x2 = expand_dims(Array::arange::<_, f32>(0, 512, None).unwrap(), 0).unwrap();
        let x = x1 * x2;

        for i in [2, 4, 8].iter() {
            let el_per_int = 32 / i;
            let (x_q, scales, biases) = quantize(&x, 128, *i).unwrap();
            assert_eq!(x_q.shape(), [128, 512 / el_per_int]);
            assert_eq!(scales.shape(), [128, 4]);
            assert_eq!(biases.shape(), [128, 4]);

            let x_hat = dequantize(&x_q, &scales, &biases, 128, *i).unwrap();
            let max_diff = ((&x - &x_hat).abs().unwrap().max(None).unwrap()).item::<f32>();
            assert!(max_diff <= 127.0 / (1 << i) as f32);
        }
    }

    // Test adapted from Python test `test_quantized.py/test_qmm`
    #[test]
    fn test_quantized_matmul() {
        random::seed(0).unwrap();

        let group_size = 64;
        let bits = 4;
        let m = 32;
        let n = 128;
        let k = 128;

        let scale = 1.0 / (k as f32).sqrt();
        let x = random::normal::<f32>(&[m, k], None, None, None).unwrap() * scale;
        let w = random::normal::<f32>(&[k, n], None, None, None).unwrap() * scale;

        let (w_q, scales, biases) = quantize(&w, group_size, bits).unwrap();
        let w_hat = dequantize(&w_q, &scales, &biases, group_size, bits).unwrap();

        // Test with biases
        let y_q = quantized_matmul(&x, &w_q, &scales, &biases, false, group_size, bits).unwrap();
        let y_hat = x.matmul(&w_hat).unwrap();

        assert_eq!(y_q.shape(), y_hat.shape());
        let max_diff = ((&y_q - &y_hat).abs().unwrap().max(None).unwrap()).item::<f32>();
        assert!(max_diff < 1e-3, "max_diff: {}", max_diff);
    }

    // Test adapted from Python test `test_quantized.py/test_gather_qmm`
    #[test]
    fn test_gather_qmm() {
        use crate::ops::{gather_mm, gather_qmm, swap_axes};

        random::seed(0).unwrap();

        let group_size = 64;
        let bits = 4;

        // Helper to quantize with transpose option
        fn quantize_with_transpose(
            w: &Array,
            transpose: bool,
            group_size: i32,
            bits: i32,
        ) -> (Array, Array, Array, Array) {
            let (w_q, scales, biases) = quantize(w, group_size, bits).unwrap();
            let mut w_hat = dequantize(&w_q, &scales, &biases, group_size, bits).unwrap();
            if transpose {
                w_hat = swap_axes(&w_hat, -1, -2).unwrap();
            }
            (w_hat, w_q, scales, biases)
        }

        // Test case 1: batch_A=(1,), lhs_indices=(0,), batch_B=(3,), rhs_indices=(2, 1)
        let m = 32;
        let n = 64;
        let k = 64;

        let x = random::normal::<f32>(&[1, m, k], None, None, None).unwrap();
        let w = random::normal::<f32>(&[3, n, k], None, None, None).unwrap(); // transpose=true shape
        let (w_hat, w_q, scales, biases) = quantize_with_transpose(&w, true, group_size, bits);

        let lhs_indices = Array::from_slice(&[0u32], &[1]);
        let rhs_indices = Array::from_slice(&[2u32, 1], &[2]);

        // Compare gather_mm on dequantized weights vs gather_qmm
        let c1 = gather_mm(&x, &w_hat, &lhs_indices, &rhs_indices, None).unwrap();
        let c2 = gather_qmm(
            &x,
            &w_q,
            &scales,
            &biases,
            &lhs_indices,
            &rhs_indices,
            true,
            group_size,
            bits,
            None,
        )
        .unwrap();
        assert!(
            c1.all_close(&c2, 1e-4, 1e-4, None).unwrap().item::<bool>(),
            "gather_qmm test case 1 failed"
        );

        // Test case 2: batch_A=(5,), lhs_indices=(0, 2), batch_B=(3,), rhs_indices=(2, 1)
        let x = random::normal::<f32>(&[5, m, k], None, None, None).unwrap();
        let lhs_indices = Array::from_slice(&[0u32, 2], &[2]);

        let c1 = gather_mm(&x, &w_hat, &lhs_indices, &rhs_indices, None).unwrap();
        let c2 = gather_qmm(
            &x,
            &w_q,
            &scales,
            &biases,
            &lhs_indices,
            &rhs_indices,
            true,
            group_size,
            bits,
            None,
        )
        .unwrap();
        assert!(
            c1.all_close(&c2, 1e-4, 1e-4, None).unwrap().item::<bool>(),
            "gather_qmm test case 2 failed"
        );
    }
}
