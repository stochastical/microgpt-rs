/// Entrypoint for native builds
#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use microgpt_rs::*;
    train(fetch_names()?);
    Ok(())
}
