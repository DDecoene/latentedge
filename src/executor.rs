use crate::config::Config;
use crate::types::{BotState, Position, TradeEvent, WalletState};
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
        self.check_daily_loss_and_rate_limit(config)
    }

    /// Daily-loss and rate-limit checks only, no size cap. `max_trade_size`
    /// caps a buy's quote-atom spend; a sell's `trade_size` is the
    /// position's base-atom size (a different unit), so the size cap does
    /// not apply when closing an already-approved position.
    fn check_daily_loss_and_rate_limit(&mut self, config: &Config) -> Result<(), String> {
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

    pub fn daily_loss_accrued(&self) -> u64 {
        self.daily_loss_accrued
    }
}

#[derive(Deserialize)]
struct JupiterQuote {
    #[serde(rename = "outAmount")]
    out_amount: String,
    #[serde(rename = "priceImpactPct")]
    price_impact_pct: String,
}

fn slippage_bps_from_impact(price_impact_pct: &str) -> u16 {
    price_impact_pct
        .parse::<f64>()
        .map(|pct| (pct * 10_000.0).round().max(0.0) as u16)
        .unwrap_or(u16::MAX)
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

async fn fetch_quote(
    jupiter_base_url: &str,
    input_mint: &str,
    output_mint: &str,
    amount: u64,
) -> Option<JupiterQuote> {
    let client = reqwest::Client::builder().timeout(QUOTE_TIMEOUT).build().ok()?;
    let url = format!(
        "{jupiter_base_url}/quote?inputMint={input_mint}&outputMint={output_mint}&amount={amount}"
    );
    client.get(url).send().await.ok()?.json::<JupiterQuote>().await.ok()
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
    state: &Arc<RwLock<BotState>>,
    position: Option<Position>,
    wallet: WalletState,
) -> (TradeEvent, Option<Position>, WalletState) {
    let dry_run = config.execution_mode != crate::config::ExecutionMode::Live;
    let trade_size = match position {
        None => ((wallet.equity() as f64) * config.trade_size_pct) as u64,
        Some(p) => p.size, // selling: sell the whole position
    };

    if kill_switch_present(&config.kill_switch_path) {
        state.write().await.kill_switch_active = true;
        return (
            refusal_event(trade_size, 0, dry_run, "refused: kill switch file present".to_string()),
            position,
            wallet,
        );
    }
    state.write().await.kill_switch_active = false;

    let guard_result = match position {
        None => guard.check(config, trade_size),
        Some(_) => guard.check_daily_loss_and_rate_limit(config),
    };
    if let Err(reason) = guard_result {
        return (refusal_event(trade_size, 0, dry_run, format!("refused: {reason}")), position, wallet);
    }

    let (input_mint, output_mint) = match position {
        None => (config.quote_mint.as_str(), config.base_mint.as_str()),
        Some(_) => (config.base_mint.as_str(), config.quote_mint.as_str()),
    };

    let Some(quote) = fetch_quote(&config.jupiter_base_url, input_mint, output_mint, trade_size).await else {
        return (refusal_event(trade_size, 0, dry_run, "refused: quote fetch failed".to_string()), position, wallet);
    };

    let quote_price: u64 = quote.out_amount.parse().unwrap_or(0);
    let realized_slippage_bps = slippage_bps_from_impact(&quote.price_impact_pct);
    if realized_slippage_bps > config.max_slippage_bps {
        return (
            refusal_event(
                trade_size,
                quote_price,
                dry_run,
                format!(
                    "refused: slippage {realized_slippage_bps} bps exceeds max {} bps",
                    config.max_slippage_bps
                ),
            ),
            position,
            wallet,
        );
    }

    guard.record_trade();

    match position {
        None => {
            let new_position = Position { size: quote_price, entry_cost: trade_size };
            (
                TradeEvent {
                    timestamp: Utc::now(),
                    side: "buy".to_string(),
                    size: trade_size,
                    price: quote_price,
                    dry_run,
                    note: "approved".to_string(),
                },
                Some(new_position),
                wallet, // buying alone doesn't realize P&L
            )
        }
        Some(p) => {
            let mut new_wallet = wallet;
            new_wallet.realized_pnl += quote_price as i64 - p.entry_cost as i64;
            if quote_price < p.entry_cost {
                guard.record_loss(p.entry_cost - quote_price);
            }
            (
                TradeEvent {
                    timestamp: Utc::now(),
                    side: "sell".to_string(),
                    size: trade_size,
                    price: quote_price,
                    dry_run,
                    note: "approved".to_string(),
                },
                None,
                new_wallet,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, ExecutionMode};
    use crate::types::{BotState, Position, WalletState};
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
                "outAmount": "990000", "priceImpactPct": "0.02"
            })))
            .mount(&server)
            .await;

        let mut config = test_config("/tmp/layatrade_test_kill_1");
        config.jupiter_base_url = server.uri();
        let mut guard = SafetyGuardState::new();
        let state = Arc::new(RwLock::new(BotState::new()));

        let (event, _position, _wallet) =
            evaluate_trade(&config, &mut guard, &state, None, flat_wallet(1000)).await;

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
                "outAmount": "999900", "priceImpactPct": "0.001"
            })))
            .mount(&server)
            .await;

        let mut config = test_config(kill_switch_file);
        config.jupiter_base_url = server.uri();
        let mut guard = SafetyGuardState::new();
        let state = Arc::new(RwLock::new(BotState::new()));

        let (event, _position, _wallet) =
            evaluate_trade(&config, &mut guard, &state, None, flat_wallet(1000)).await;

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
                "outAmount": "999900", "priceImpactPct": "0.001"
            })))
            .mount(&server)
            .await;

        let mut config = test_config(&unreadable_path);
        config.jupiter_base_url = server.uri();
        let mut guard = SafetyGuardState::new();
        let state = Arc::new(RwLock::new(BotState::new()));

        let (event, _position, _wallet) =
            evaluate_trade(&config, &mut guard, &state, None, flat_wallet(1000)).await;

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
                    .set_body_json(serde_json::json!({"outAmount": "999900", "priceImpactPct": "0.001"})),
            )
            .mount(&server)
            .await;

        let mut config = test_config("/tmp/layatrade_test_kill_7");
        config.jupiter_base_url = server.uri();
        let mut guard = SafetyGuardState::new();
        let state = Arc::new(RwLock::new(BotState::new()));

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            evaluate_trade(&config, &mut guard, &state, None, flat_wallet(1000)),
        )
        .await;

        let (event, _position, _wallet) =
            result.expect("evaluate_trade should time out its own quote request, not hang");
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

    fn flat_wallet(starting_capital: u64) -> WalletState {
        WalletState { starting_capital, realized_pnl: 0 }
    }

    #[tokio::test]
    async fn flat_position_buys_a_fraction_of_equity() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/quote"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "outAmount": "5000000", "priceImpactPct": "0.001"
            })))
            .mount(&server)
            .await;

        let mut config = test_config("/tmp/layatrade_test_kill_8");
        config.jupiter_base_url = server.uri();
        config.trade_size_pct = 0.1;
        let mut guard = SafetyGuardState::new();
        let state = Arc::new(RwLock::new(BotState::new()));
        let wallet = flat_wallet(1000);

        let (event, new_position, new_wallet) =
            evaluate_trade(&config, &mut guard, &state, None, wallet).await;

        assert_eq!(event.side, "buy");
        let position = new_position.expect("a successful buy must open a position");
        assert_eq!(position.size, 5_000_000);
        assert_eq!(position.entry_cost, 100); // 10% of 1000 equity
        assert_eq!(new_wallet.equity(), 1000); // buying alone doesn't realize P&L
    }

    #[tokio::test]
    async fn holding_position_sells_and_records_loss_when_losing() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/quote"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "outAmount": "90", "priceImpactPct": "0.001"
            })))
            .mount(&server)
            .await;

        let mut config = test_config("/tmp/layatrade_test_kill_9");
        config.jupiter_base_url = server.uri();
        let mut guard = SafetyGuardState::new();
        let state = Arc::new(RwLock::new(BotState::new()));
        let position = Position { size: 5_000_000, entry_cost: 100 };
        let wallet = flat_wallet(1000);

        let (event, new_position, new_wallet) =
            evaluate_trade(&config, &mut guard, &state, Some(position), wallet).await;

        assert_eq!(event.side, "sell");
        assert!(new_position.is_none(), "position must clear after a sell");
        assert_eq!(guard.daily_loss_accrued(), 10); // entry_cost 100 - proceeds 90
        assert_eq!(new_wallet.equity(), 990); // 1000 - 10 realized loss
    }

    #[tokio::test]
    async fn holding_position_sells_and_grows_equity_when_winning() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/quote"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "outAmount": "200", "priceImpactPct": "0.001"
            })))
            .mount(&server)
            .await;

        let mut config = test_config("/tmp/layatrade_test_kill_10");
        config.jupiter_base_url = server.uri();
        let mut guard = SafetyGuardState::new();
        let state = Arc::new(RwLock::new(BotState::new()));
        let position = Position { size: 5_000_000, entry_cost: 100 };
        let wallet = flat_wallet(1000);

        let (event, new_position, new_wallet) =
            evaluate_trade(&config, &mut guard, &state, Some(position), wallet).await;

        assert_eq!(event.side, "sell");
        assert!(new_position.is_none());
        assert!(event.note.contains("approved"));
        assert_eq!(guard.daily_loss_accrued(), 0);
        assert_eq!(new_wallet.equity(), 1100); // 1000 + 100 realized gain
    }

    #[tokio::test]
    async fn next_buy_size_compounds_with_grown_equity() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/quote"))
            .and(wiremock::matchers::query_param("amount", "110"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "outAmount": "5000000", "priceImpactPct": "0.001"
            })))
            .mount(&server)
            .await;

        let mut config = test_config("/tmp/layatrade_test_kill_12");
        config.jupiter_base_url = server.uri();
        config.trade_size_pct = 0.1;
        let mut guard = SafetyGuardState::new();
        let state = Arc::new(RwLock::new(BotState::new()));

        // Equity already grew to 1100 from a prior winning round trip.
        let grown_wallet = flat_wallet(1000);
        let grown_wallet = WalletState { realized_pnl: 100, ..grown_wallet };
        let (event, _position, _wallet) =
            evaluate_trade(&config, &mut guard, &state, None, grown_wallet).await;

        assert_eq!(event.side, "buy");
        assert_eq!(event.price, 5_000_000); // outAmount, unaffected by spend size in this mock
        // 10% of 1100 equity = 110, not 100 — proves sizing tracks equity.
        assert_eq!(
            guard.daily_loss_accrued(),
            0,
            "sanity: no loss recorded on a buy"
        );
    }

    #[tokio::test]
    async fn equity_floor_holds_when_wallet_has_large_realized_losses() {
        let wallet = WalletState { starting_capital: 1000, realized_pnl: -5000 };
        assert_eq!(wallet.equity(), 0);
        // A buy against a zeroed-out wallet must spend 0, not underflow or panic.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/quote"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "outAmount": "0", "priceImpactPct": "0.001"
            })))
            .mount(&server)
            .await;
        let mut config = test_config("/tmp/layatrade_test_kill_13");
        config.jupiter_base_url = server.uri();
        let mut guard = SafetyGuardState::new();
        let state = Arc::new(RwLock::new(BotState::new()));

        let (event, _position, _wallet) =
            evaluate_trade(&config, &mut guard, &state, None, wallet).await;
        assert_eq!(event.size, 0);
    }
}
