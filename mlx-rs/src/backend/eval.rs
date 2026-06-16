use super::error::{MlxError, MlxResult};
use crate::Array;

pub fn eval_array(array: &Array) -> MlxResult<()> {
    array.eval().map_err(|e| MlxError::EvaluationFailed(e.what))
}

pub fn eval_arrays(arrays: &[&Array]) -> MlxResult<()> {
    crate::transforms::eval(arrays.iter().map(|a| *a)).map_err(|e| MlxError::EvaluationFailed(e.what))
}

pub fn readback_f32(array: &Array) -> MlxResult<Vec<f32>> {
    eval_array(array)?;
    if array.dtype() != crate::Dtype::Float32 {
        return Err(MlxError::UnsupportedDType);
    }

    // as_slice() forces evaluation if not already evaluated.
    let slice: &[f32] = array.as_slice();
    Ok(slice.to_vec())
}
