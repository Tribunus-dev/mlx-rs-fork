pub mod error;
pub mod dtype;
pub mod tensor;
pub mod capabilities;
pub mod evidence;
pub mod eval;
pub mod ops;
pub mod reference;

pub use error::{MlxError, MlxResult};
pub use dtype::DType;
pub use tensor::{TensorSpec, TensorLayout, DevicePreference, TensorRole};
pub use capabilities::{MlxBackendCapabilities, SupportStatus, ImplementationKind};
pub use evidence::{ConformanceEvidence, NumericalComparison};
pub use eval::{eval_array, eval_arrays, readback_f32};
pub use ops::BackendConformanceRunner;
