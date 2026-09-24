use crate::config::Config;
use crate::types::{BotState, TradeEvent};
use chrono::Utc;
use serde::Deserialize;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

pub struct SafetyGuardState {
    trade_timestamps: Vec<Instant>,
    daily_loss_accrued: u64,
}

impl SafetyGuardState {
    pub fn new() -> Self {
        Self { trade_timestamps: Vec::new(), daily_loss_accrued: 0 }
    }

    pub fn check(&mut self, config: &Config, trade_size: u64) -> Result<(), String> {
        if trade_size > config.max_trade_size {
            return Err(format!(
                "trade size {trade_size} exceeds max_trade_size {}",
                config.max_trade_size
            ));
        }
        if self.daily_loss_accrued >= config.max_daily_loss {
            return Err(format!(
                "daily loss cap breached: accrued {} >= max {}",
                self.daily_loss_accrued, config.max_daily_loss
            ));
        }
        let window = Duration::from_secs(60);
        let now = Instant::now();
        self.trade_timestamps.retain(|t| now.duration_since(*t) < window);
        if self.trade_timestamps.len() as u32 >= config.max_trades_per_window {
            return Err(format!(
                "rate limit exceeded: {} trades in the last window",
                self.trade_timestamps.len()
            ));
        }
        Ok(())
    }

    pub fn record_trade(&mut self) {
        self.trade_timestamps.push(Instant::now());
    }

    pub fn record_loss(&mut self, amount: u64) {
        self.daily_loss_accrued += amount;
    }
}

#[derive(Deserialize)]
struct JupiterQuote {
    #[serde(rename = "outAmount")]
    out_amount: String,
    #[serde(rename = "slippageBps")]
    slippage_bps: u16,
}

/// Fails closed: if existence can't be determined (e.g. a permission
/// error on a parent directory), this treats the kill switch as active
/// rather than absent. Only a definite "not found" counts as "safe to
/// trade".
fn kill_switch_present(path: &str) -> bool {
    match std::fs::metadata(path) {
        Ok(_) => true,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(_) => true,
    }
}

const QUOTE_TIMEOUT: Duration = Duration::from_millis(500);

async fn fetch_quote(jupiter_base_url: &str) -> Option<JupiterQuote> {
    let client = reqwest::Client::builder().timeout(QUOTE_TIMEOUT).build().ok()?;
    client
        .get(format!("{jupiter_base_url}/quote"))
        .send()
        .await
        .ok()?
        .json::<JupiterQuote>()
        .await
        .ok()
}

fn refusal_event(trade_size: u64, price: u64, dry_run: bool, note: String) -> TradeEvent {
    TradeEvent {
        timestamp: Utc::now(),
        side: "none".to_string(),
        size: trade_size,
        price,
        dry_run,
        note,
    }
}

pub async fn evaluate_trade(
    config: &Config,
    guard: &mut SafetyGuardState,
    jupiter_base_url: &str,
    state: &Arc<RwLock<BotState>>,
) -> TradeEvent {
    let trade_size = ((config.starting_capital as f64) * config.trade_size_pct) as u64; // interim: Task 4 replaces this with real equity/position awareness
    let dry_run = config.execution_mode != crate::config::ExecutionMode::Live;

    if kill_switch_present(&config.kill_switch_path) {
        state.write().await.kill_switch_active = true;
        return refusal_event(
            trade_size,
            0,
            dry_run,
            "refused: kill switch file present".to_string(),
        );
    }
    state.write().await.kill_switch_active = false;

    if let Err(reason) = guard.check(config, trade_size) {
        return refusal_event(trade_size, 0, dry_run, format!("refused: {reason}"));
    }

    let Some(quote) = fetch_quote(jupiter_base_url).await else {
        return refusal_event(
            trade_size,
            0,
            dry_run,
            "refused: quote fetch failed".to_string(),
        );
    };

    let quote_price = quote.out_amount.parse().unwrap_or(0);
    if quote.slippage_bps > config.max_slippage_bps {
        return refusal_event(
            trade_size,
            quote_price,
            dry_run,
            format!(
                "refused: slippage {} bps exceeds max {} bps",
                quote.slippage_bps, config.max_slippage_bps
            ),
        );
    }

    guard.record_trade();
    TradeEvent {
        timestamp: Utc::now(),
        side: "buy".to_string(),
        size: trade_size,
        price: quote_price,
        dry_run,
        note: "approved".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ExecutionMode};
    use crate::types::BotState;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn test_config(kill_switch_path: &str) -> Config {
        Config {
            solana_rpc_url: "https://x".into(),
            laya_server_url: "http://127.0.0.1:8787".into(),
            laya_confidence_threshold: 0.85,
            execution_mode: ExecutionMode::DryRun,
            solana_keypair_path: None,
            max_trade_size: 1_000_000,
            max_trades_per_window: 5,
            max_slippage_bps: 50,
            max_daily_loss: 5_000_000,
            kill_switch_path: kill_switch_path.to_string(),
            jupiter_base_url: "https://quote-api.jup.ag/v6".into(),
            base_mint: "So11111111111111111111111111111111111111112".into(),
            quote_mint: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v".into(),
            poll_interval_ms: 5000,
            trade_size_pct: 0.1,
            starting_capital: 1000,
        }
    }

    #[tokio::test]
    async fn rejects_trade_when_quote_slippage_exceeds_tolerance() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/quote"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "outAmount": "990000", "slippageBps": 200
            })))
            .mount(&server)
            .await;

        let config = test_config("/tmp/layatrade_test_kill_1");
        let mut guard = SafetyGuardState::new();
        let state = Arc::new(RwLock::new(BotState::new()));

        let event = evaluate_trade(&config, &mut guard, &server.uri(), &state).await;

        assert!(event.note.contains("slippage"));
        assert!(event.dry_run);
    }

    #[tokio::test]
    async fn refuses_when_kill_switch_file_present() {
        let kill_switch_file = "/tmp/layatrade_test_kill_2";
        std::fs::write(kill_switch_file, b"stop").unwrap();

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/quote"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "outAmount": "999900", "slippageBps": 10
            })))
            .mount(&server)
            .await;

        let config = test_config(kill_switch_file);
        let mut guard = SafetyGuardState::new();
        let state = Arc::new(RwLock::new(BotState::new()));

        let event = evaluate_trade(&config, &mut guard, &server.uri(), &state).await;

        assert!(event.note.contains("kill switch"));
        std::fs::remove_file(kill_switch_file).ok();
    }

    #[tokio::test]
    async fn kill_switch_fails_closed_when_path_cannot_be_checked() {
        // A path inside a directory with no execute permission can't be
        // stat()-ed: metadata() returns PermissionDenied, not NotFound.
        // The kill switch must treat that as "active" (fail closed), not
        // as "absent" (fail open).
        let dir = "/tmp/layatrade_test_kill_dir_unreadable";
        std::fs::create_dir_all(dir).unwrap();
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o000)).unwrap();
        let unreadable_path = format!("{dir}/kill_switch");

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/quote"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "outAmount": "999900", "slippageBps": 10
            })))
            .mount(&server)
            .await;

        let config = test_config(&unreadable_path);
        let mut guard = SafetyGuardState::new();
        let state = Arc::new(RwLock::new(BotState::new()));

        let event = evaluate_trade(&config, &mut guard, &server.uri(), &state).await;

        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::remove_dir_all(dir).ok();

        assert!(event.note.contains("kill switch"), "note was: {}", event.note);
    }

    #[tokio::test]
    async fn quote_fetch_times_out_instead_of_hanging() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/quote"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_secs(5))
                    .set_body_json(serde_json::json!({"outAmount": "999900", "slippageBps": 10})),
            )
            .mount(&server)
            .await;

        let config = test_config("/tmp/layatrade_test_kill_7");
        let mut guard = SafetyGuardState::new();
        let state = Arc::new(RwLock::new(BotState::new()));

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            evaluate_trade(&config, &mut guard, &server.uri(), &state),
        )
        .await;

        let event = result.expect("evaluate_trade should time out its own quote request, not hang");
        assert!(event.note.contains("quote fetch failed"));
    }

    #[test]
    fn rejects_trade_over_max_size() {
        let config = test_config("/tmp/layatrade_test_kill_3");
        let mut guard = SafetyGuardState::new();
        let result = guard.check(&config, config.max_trade_size + 1);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_after_daily_loss_cap_breached() {
        let config = test_config("/tmp/layatrade_test_kill_4");
        let mut guard = SafetyGuardState::new();
        guard.record_loss(config.max_daily_loss);
        let result = guard.check(&config, 1);
        assert!(result.unwrap_err().contains("daily loss"));
    }

    #[test]
    fn rejects_over_rate_limit() {
        let config = test_config("/tmp/layatrade_test_kill_5");
        let mut guard = SafetyGuardState::new();
        for _ in 0..config.max_trades_per_window {
            guard.check(&config, 1).unwrap();
            guard.record_trade();
        }
        let result = guard.check(&config, 1);
        assert!(result.unwrap_err().contains("rate limit"));
    }
}
