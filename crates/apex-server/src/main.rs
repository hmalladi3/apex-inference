use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    apex_server::run(apex_server::Args::parse()).await
}
