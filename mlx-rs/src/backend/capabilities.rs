use super::dtype::DType;

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "evidence", derive(serde::Serialize, serde::Deserialize))]
pub enum SupportStatus {
    Supported,
    PartiallySupported,
    Unsupported,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "evidence", derive(serde::Serialize, serde::Deserialize))]
pub enum ImplementationKind {
    NativeMlx,
    ComposedMlx,
    RustReference,
    MetadataOnly,
    Unsupported,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "evidence", derive(serde::Serialize, serde::Deserialize))]
pub struct OperationEntry {
    pub name: String,
    pub support_status: SupportStatus,
    pub implementation_kind: ImplementationKind,
    pub supported_dtypes: Vec<DType>,
    pub shape_notes: Option<String>,
    pub limitations: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "evidence", derive(serde::Serialize, serde::Deserialize))]
pub struct PlatformSummary {
    pub os: String,
    pub architecture: String,
    pub is_apple: bool,
    pub is_apple_silicon: bool,
    pub metal_available: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "evidence", derive(serde::Serialize, serde::Deserialize))]
pub struct MlxBackendCapabilities {
    pub schema_version: String,
    pub crate_version: String,
    pub git_commit_hash: Option<String>,
    pub enabled_features: Vec<String>,
    pub platform: PlatformSummary,
    pub mlx_runtime_version: Option<String>,
    pub supported_dtypes: Vec<DType>,
    pub supported_devices: Vec<String>,
    pub supported_operations: Vec<OperationEntry>,
    pub known_limitations: Vec<String>,
    // TODO: Add `capability_hash` once canonical serialization rules are defined.
}
