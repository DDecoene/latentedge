use layatrade_rs::config::Config;
use layatrade_rs::executor::{evaluate_trade, SafetyGuardState};
use layatrade_rs::laya_client::LayaClient;
use layatrade_rs::streamer::{run_streamer, FetchLadder};
use layatrade_rs::tui::run_tui;
use layatrade_rs::types::{BotState, TradeEvent};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing_subscriber::EnvFilter;

/// Placeholder fetcher wiring real Solana WS RPC + phoenix-sdk decoding;
/// see Task 4's note on replacing this with a live subscription before
/// enabling real trading.
struct RpcPollFetcher {
    rpc_url: String,
}

#[async_trait::async_trait]
impl FetchLadder for RpcPollFetcher {
    async fn fetch(&self) -> anyhow::Result<(u64, String)> {
        anyhow::bail!(
            "RpcPollFetcher against {} not yet implemented — see Task 4 note",
            self.rpc_url
        )
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let file_appender = tracing_appender::rolling::daily("./logs", "layatrade.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let config = Config::from_env()?;
    tracing::info!(mode = ?config.execution_mode, "starting layatrade-rs");

    let laya = LayaClient::new(config.laya_server_url.clone(), config.laya_confidence_threshold);
    laya.health_check().await.map_err(|e| {
        anyhow::anyhow!("Laya server not reachable at {}: {e}", config.laya_server_url)
    })?;

    let state = Arc::new(RwLock::new(BotState::new()));
    let (shutdown_tx, _) = broadcast::channel::<()>(1);
    let (snapshot_tx, mut snapshot_rx) = mpsc::channel(16);

    let fetcher = Arc::new(RpcPollFetcher { rpc_url: config.solana_rpc_url.clone() });

    let streamer_state = state.clone();
    let streamer_config = config.clone();
    let streamer_shutdown = shutdown_tx.subscribe();
    let streamer_handle = tokio::spawn(async move {
        let _ = run_streamer(
            fetcher,
            streamer_config.phoenix_market_address,
            streamer_state,
            snapshot_tx,
            streamer_shutdown,
        )
        .await;
    });

    let signal_state = state.clone();
    let signal_config = config.clone();
    let mut signal_shutdown = shutdown_tx.subscribe();
    let signal_handle = tokio::spawn(async move {
        let mut guard = SafetyGuardState::new();
        loop {
            tokio::select! {
                _ = signal_shutdown.recv() => break,
                Some(snapshot) = snapshot_rx.recv() => {
                    let sig = laya.get_signal(&snapshot).await;
                    signal_state.write().await.last_signal = Some(sig);
                    signal_state.write().await.last_snapshot = Some(snapshot);

                    if sig.should_trade {
                        let event: TradeEvent = evaluate_trade(
                            &signal_config,
                            &mut guard,
                            "https://quote-api.jup.ag/v6",
                            &signal_state,
                        ).await;
                        signal_state.write().await.push_trade(event);
                    }
                }
            }
        }
    });

    let tui_state = state.clone();
    let tui_shutdown = shutdown_tx.subscribe();
    let tui_handle = tokio::spawn(async move {
        let _ = run_tui(tui_state, tui_shutdown).await;
    });

    tokio::signal::ctrl_c().await?;
    let _ = shutdown_tx.send(());

    let _ = tokio::join!(streamer_handle, signal_handle, tui_handle);
    Ok(())
}
