use crate::config::Config;
use crate::phoenix_decode::decode_from_ladder_json;
use crate::types::{BotState, OrderBookSnapshot, StreamHealth};
use chrono::Utc;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};

#[async_trait::async_trait]
pub trait FetchLadder: Send + Sync {
    /// Returns (slot, raw_ladder_json) or an error representing a dropped
    /// connection / failed fetch.
    async fn fetch(&self) -> anyhow::Result<(u64, String)>;
}

/// Attempts one fetch; on failure, retries once immediately (real backoff
/// timing is applied by the caller loop in `run_streamer`). On success,
/// decodes, sends the snapshot, and records reconnects in `state`.
pub async fn poll_once_with_retry<F: FetchLadder>(
    fetcher: Arc<F>,
    market: &str,
    state: &Arc<RwLock<BotState>>,
    snapshot_tx: &mpsc::Sender<OrderBookSnapshot>,
    mut reconnects_so_far: u32,
) -> anyhow::Result<()> {
    let (slot, raw_json) = match fetcher.fetch().await {
        Ok(result) => result,
        Err(_first_err) => {
            reconnects_so_far += 1;
            fetcher.fetch().await?
        }
    };

    let snapshot = decode_from_ladder_json(market, slot, &raw_json)?;
    snapshot_tx.send(snapshot).await?;

    let mut guard = state.write().await;
    guard.stream_health = Some(StreamHealth {
        last_update: Utc::now(),
        reconnect_count: reconnects_so_far,
    });
    Ok(())
}

pub async fn run_streamer<F: FetchLadder + 'static>(
    fetcher: Arc<F>,
    market: String,
    state: Arc<RwLock<BotState>>,
    snapshot_tx: mpsc::Sender<OrderBookSnapshot>,
    mut shutdown_rx: broadcast::Receiver<()>,
) -> anyhow::Result<()> {
    let mut reconnects = 0u32;
    let mut backoff_ms: u64 = 200;
    loop {
        tokio::select! {
            _ = shutdown_rx.recv() => return Ok(()),
            result = poll_once_with_retry(fetcher.clone(), &market, &state, &snapshot_tx, reconnects) => {
                match result {
                    Ok(()) => {
                        backoff_ms = 200;
                        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                    }
                    Err(e) => {
                        reconnects += 1;
                        tracing::warn!(error = %e, "streamer fetch failed, backing off {backoff_ms}ms");
                        tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                        backoff_ms = (backoff_ms * 2).min(5_000);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct FlakyThenGoodFetcher {
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl FetchLadder for FlakyThenGoodFetcher {
        async fn fetch(&self) -> anyhow::Result<(u64, String)> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                anyhow::bail!("simulated disconnect");
            }
            Ok((n as u64, r#"{"bids":[[100,1]],"asks":[[101,1]]}"#.to_string()))
        }
    }

    #[tokio::test]
    async fn reconnects_after_failure_and_updates_health() {
        let fetcher = Arc::new(FlakyThenGoodFetcher { calls: AtomicUsize::new(0) });
        let state = Arc::new(tokio::sync::RwLock::new(BotState::new()));
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);

        poll_once_with_retry(fetcher.clone(), "marketX", &state, &tx, 0)
            .await
            .expect("should recover after one retry");

        let snapshot = rx.try_recv().expect("snapshot should be sent");
        assert_eq!(snapshot.slot, 1);

        let health = state.read().await.stream_health.clone().unwrap();
        assert_eq!(health.reconnect_count, 1);
    }
}
