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

#[derive(serde::Deserialize)]
struct JupiterQuoteResponse {
    #[serde(rename = "outAmount")]
    out_amount: String,
    #[serde(rename = "priceImpactPct")]
    price_impact_pct: String,
}

/// Polls Jupiter's quote endpoint for a fixed reference size, deriving a
/// synthetic single-level bid/ask spread. Jupiter has no push/WS feed, so
/// this is polling, not streaming; run_streamer's existing backoff covers
/// fetch failures the same way it covered a dropped WS connection before.
pub struct JupiterQuotePoller {
    pub jupiter_base_url: String,
    pub base_mint: String,
    pub quote_mint: String,
}

const REFERENCE_QUOTE_ATOMS: u64 = 1_000_000; // 1 USDC at 6 decimals

#[async_trait::async_trait]
impl FetchLadder for JupiterQuotePoller {
    async fn fetch(&self) -> anyhow::Result<(u64, String)> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(500))
            .build()?;
        let url = format!(
            "{}/quote?inputMint={}&outputMint={}&amount={}",
            self.jupiter_base_url, self.quote_mint, self.base_mint, REFERENCE_QUOTE_ATOMS
        );
        let response = client.get(url).send().await?.error_for_status()?;
        let quote: JupiterQuoteResponse = response.json().await?;

        let ask_price: u64 = quote.out_amount.parse()?;
        let price_impact: f64 = quote.price_impact_pct.parse().unwrap_or(0.0);
        let impact_atoms = ((ask_price as f64) * price_impact).round() as u64;
        // Guarantee bid < ask even when priceImpactPct is 0 or unparsable,
        // since phoenix_decode::validate_ladder rejects a crossed book.
        let bid_price = ask_price.saturating_sub(impact_atoms.max(1));

        let raw_json = serde_json::json!({
            "bids": [[bid_price, 1]],
            "asks": [[ask_price, 1]],
        })
        .to_string();

        let slot = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok((slot, raw_json))
    }
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
    poll_interval_ms: u64,
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
                        tokio::time::sleep(tokio::time::Duration::from_millis(poll_interval_ms)).await;
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

    #[tokio::test]
    async fn jupiter_quote_poller_produces_a_valid_ladder() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/quote"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "outAmount": "5000000",
                "priceImpactPct": "0.001"
            })))
            .mount(&server)
            .await;

        let poller = JupiterQuotePoller {
            jupiter_base_url: server.uri(),
            base_mint: "base".to_string(),
            quote_mint: "quote".to_string(),
        };

        let (slot, raw_json) = poller.fetch().await.expect("fetch should succeed");
        assert!(slot > 0);
        let snapshot = decode_from_ladder_json("m", slot, &raw_json)
            .expect("poller output must be a valid, uncrossed ladder");
        assert!(snapshot.bids[0].0 < snapshot.asks[0].0);
    }

    #[tokio::test]
    async fn jupiter_quote_poller_errors_instead_of_panicking_on_bad_response() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/quote"))
            .respond_with(wiremock::ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let poller = JupiterQuotePoller {
            jupiter_base_url: server.uri(),
            base_mint: "base".to_string(),
            quote_mint: "quote".to_string(),
        };

        assert!(poller.fetch().await.is_err());
    }
}
