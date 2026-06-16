use mlx_rs::backend::{
    MlxBackendCapabilities, DType,
    capabilities::PlatformSummary
};

fn main() {
    let caps = MlxBackendCapabilities {
        schema_version: "tribunus.mlx.backend_capabilities.v0".into(),
        crate_version: env!("CARGO_PKG_VERSION").into(),
        git_commit_hash: None,
        enabled_features: vec![
            "evidence".into()
        ],
        platform: PlatformSummary {
            os: std::env::consts::OS.into(),
            architecture: std::env::consts::ARCH.into(),
            is_apple: std::env::consts::OS == "macos",
            is_apple_silicon: std::env::consts::OS == "macos" && std::env::consts::ARCH == "aarch64",
            metal_available: None,
        },
        mlx_runtime_version: None,
        supported_dtypes: vec![DType::F32],
        supported_devices: vec!["CPU".into()],
        supported_operations: vec![],
        known_limitations: vec!["Canonical hash computation deferred.".into()],
    };

    println!("{}", serde_json::to_string_pretty(&caps).unwrap());
}
