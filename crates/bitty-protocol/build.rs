fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use the system protoc when there is one, otherwise the vendored binary, so a plain
    // `cargo build` works on machines without protobuf installed.
    if std::env::var_os("PROTOC").is_none() {
        std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);
    }
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["../../proto/bitty/v1/cluster.proto"], &["../../proto"])?;
    println!("cargo:rerun-if-changed=../../proto/bitty/v1/cluster.proto");
    Ok(())
}
