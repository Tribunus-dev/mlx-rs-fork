use mlx_rs::backend::{
    MlxBackendCapabilities, BackendConformanceRunner,
    capabilities::PlatformSummary, DType
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

    let runner = BackendConformanceRunner::default().with_capabilities(caps);
    let evidence = runner.run_core_ops().unwrap();

    let mut failed = false;
    for record in evidence {
        if record.error.is_some() || record.comparison.as_ref().map(|c| !c.passed).unwrap_or(false) {
            failed = true;
        }
        println!("{}", serde_json::to_string(&record).unwrap());
    }

    if failed {
        std::process::exit(1);
    }
}
