#[derive(Debug, Clone, Copy)]
pub enum BridgeRuntime {
    /// Dispatch `Session::run` via `tokio::task::spawn_blocking`.
    Blocking,
    /// Dispatch via a dedicated OS thread per session. Not implemented in v1
    /// — declared so the variant exists in the type; see ort-bridge LLD.
    DedicatedThread,
}
