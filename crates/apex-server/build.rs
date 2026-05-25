fn main() -> Result<(), Box<dyn std::error::Error>> {
    // KServe v2 protocol must be vendored into /proto/inference.proto.
    // Source: https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/grpc_predict_v2.proto
    let proto = "../../proto/inference.proto";
    if std::path::Path::new(proto).exists() {
        tonic_build::compile_protos(proto)?;
        println!("cargo:rerun-if-changed={proto}");
    } else {
        println!("cargo:warning=KServe v2 proto not yet vendored at {proto}; skipping codegen");
    }
    Ok(())
}
