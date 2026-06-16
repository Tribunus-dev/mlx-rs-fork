use mlx_rs::backend::{
    MlxBackendCapabilities, BackendConformanceRunner,
    DevicePreference, DType, TensorSpec, SupportStatus
};

#[test]
fn test_conformance_runner_can_init() {
    let caps = MlxBackendCapabilities {
        schema_version: "tribunus.mlx.backend_capabilities.v0".into(),
        crate_version: "0.1.0".into(),
        git_commit_hash: None,
        enabled_features: vec![],
        platform: mlx_rs::backend::capabilities::PlatformSummary {
            os: "test".into(),
            architecture: "test".into(),
            is_apple: false,
            is_apple_silicon: false,
            metal_available: None,
        },
        mlx_runtime_version: None,
        supported_dtypes: vec![DType::F32],
        supported_devices: vec!["CPU".into()],
        supported_operations: vec![],
        known_limitations: vec![],
    };

    let runner = BackendConformanceRunner::default()
        .with_capabilities(caps);

    // We expect this to run and produce evidence records without aborting,
    // though the numerical results might fail or pass depending on the actual MLX backend.
    let evidence = runner.run_core_ops().unwrap();
    assert_eq!(evidence.len(), 10, "Expected exactly 10 operations evaluated");

    for record in evidence {
        assert_eq!(record.schema_version, "tribunus.mlx.conformance_evidence.v0");
        // We do not strictly assert it passes here, because CI might lack MLX completely or lack GPUs.
        // We only assert that we captured the evidence properly as requested by the plan.
        assert_ne!(record.support_status, SupportStatus::Unknown);
    }
}

#[test]
fn test_tensor_spec_validation() {
    let valid_spec = TensorSpec::dense(DType::F32, vec![2, 3], DevicePreference::Default);
    assert!(valid_spec.validate().is_ok());

    let invalid_spec = TensorSpec::dense(DType::F32, vec![2, 0, 3], DevicePreference::Default);
    assert!(invalid_spec.validate().is_err());
}

#[test]
fn test_negative_invalid_shape_add() {
    let a = mlx_rs::Array::from_slice(&[1.0_f32, 2.0], &[2]);
    let b = mlx_rs::Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3]);
    let res = mlx_rs::ops::add(&a, &b);
    assert!(res.is_err());
}

#[test]
fn test_negative_invalid_shape_matmul() {
    let a = mlx_rs::Array::from_slice(&[1.0_f32, 2.0], &[2, 1]);
    let b = mlx_rs::Array::from_slice(&[1.0_f32, 2.0, 3.0], &[3, 1]);
    let res = mlx_rs::ops::matmul(&a, &b);
    assert!(res.is_err());
}

#[test]
fn test_negative_reshape_element_count() {
    let a = mlx_rs::Array::from_slice(&[1.0_f32, 2.0], &[2]);
    let res = mlx_rs::ops::reshape(&a, &[3]);
    assert!(res.is_err());
}
