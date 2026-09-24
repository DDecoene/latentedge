use crate::config::Config;
use crate::types::{BotState, Position, TradeEvent, WalletState};
use chrono::Utc;
use serde::Deserialize;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_sdk::transaction::VersionedTransaction;
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

    pub(crate) fn check_size(&self, config: &Config, trade_size: u64) -> Result<(), String> {
        if trade_size > config.max_trade_size {
            return Err(format!(
                "trade size {trade_size} exceeds max_trade_size {}",
                config.max_trade_size
            ));
        }
        Ok(())
    }

    pub(crate) fn check_daily_loss(&self, config: &Config) -> Result<(), String> {
        if self.daily_loss_accrued >= config.max_daily_loss {
            return Err(format!(
                "daily loss cap breached: accrued {} >= max {}",
                self.daily_loss_accrued, config.max_daily_loss
            ));
        }
        Ok(())
    }

    pub(crate) fn check_rate_limit(&mut self, config: &Config) -> Result<(), String> {
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

    pub fn check(&mut self, config: &Config, trade_size: u64) -> Result<(), String> {
        self.check_size(config, trade_size)?;
        self.check_daily_loss(config)?;
        self.check_rate_limit(config)?;
        Ok(())
    }

    /// Daily-loss and rate-limit checks only, no size cap. `max_trade_size`
    /// caps a buy's quote-atom spend; a sell's `trade_size` is the
    /// position's base-atom size (a different unit), so the size cap does
    /// not apply when closing an already-approved position.
    pub(crate) fn check_daily_loss_and_rate_limit(&mut self, config: &Config) -> Result<(), String> {
        self.check_daily_loss(config)?;
        self.check_rate_limit(config)?;
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
const SWAP_TIMEOUT: Duration = Duration::from_millis(3000);

/// Returns the typed quote fields alongside the raw response JSON: the
/// raw value is what actually gets POSTed to /swap in live mode, so the
/// swap executes against the exact quote that passed every guard, not a
/// second, unchecked one.
async fn fetch_quote(
    jupiter_base_url: &str,
    input_mint: &str,
    output_mint: &str,
    amount: u64,
    slippage_bps: u16,
) -> Option<(JupiterQuote, serde_json::Value)> {
    let client = reqwest::Client::builder().timeout(QUOTE_TIMEOUT).build().ok()?;
    let url = format!(
        "{jupiter_base_url}/quote?inputMint={input_mint}&outputMint={output_mint}&amount={amount}&slippageBps={slippage_bps}"
    );
    let raw: serde_json::Value = client.get(url).send().await.ok()?.json().await.ok()?;
    let quote: JupiterQuote = serde_json::from_value(raw.clone()).ok()?;
    Some((quote, raw))
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

#[derive(Deserialize)]
struct JupiterSwapResponse {
    #[serde(rename = "swapTransaction")]
    swap_transaction: String,
}

/// Decodes Jupiter's base64 swap transaction and re-signs its message
/// with our keypair. Jupiter already built the message with our pubkey
/// as the required signer, so `VersionedTransaction::try_new` (which
/// signs a message fresh) is the correct call here, not appending a
/// signature to the existing (placeholder) one.
fn sign_swap_transaction(
    base64_transaction: &str,
    keypair: &Keypair,
) -> anyhow::Result<VersionedTransaction> {
    use base64::Engine;
    let raw = base64::engine::general_purpose::STANDARD.decode(base64_transaction)?;
    let unsigned: VersionedTransaction = bincode::deserialize(&raw)?;
    let signed = VersionedTransaction::try_new(unsigned.message, &[keypair])?;
    Ok(signed)
}

/// Signs and submits the swap for the quote that already passed every
/// guard — `quote_response` must be the exact raw JSON `fetch_quote`
/// returned, not a re-fetched one, so the on-chain transaction executes
/// against the same price/slippage terms the caller approved. Only
/// reachable from live mode, after every guard has already passed.
async fn submit_live_swap(
    config: &Config,
    quote_response: &serde_json::Value,
) -> anyhow::Result<String> {
    let keypair_path = config
        .solana_keypair_path
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("SOLANA_KEYPAIR_PATH is required in live mode"))?;
    let keypair = solana_sdk::signer::keypair::read_keypair_file(keypair_path)
        .map_err(|e| anyhow::anyhow!("failed to read keypair file: {e}"))?;

    let client = reqwest::Client::builder().timeout(SWAP_TIMEOUT).build()?;
    let swap_response: JupiterSwapResponse = client
        .post(format!("{}/swap", config.jupiter_base_url))
        .json(&serde_json::json!({
            "quoteResponse": quote_response,
            "userPublicKey": keypair.pubkey().to_string(),
            "wrapAndUnwrapSol": true,
        }))
        .send()
        .await?
        .json()
        .await?;

    let signed = sign_swap_transaction(&swap_response.swap_transaction, &keypair)?;

    let rpc_client = solana_client::nonblocking::rpc_client::RpcClient::new(config.solana_rpc_url.clone());
    let signature = rpc_client.send_and_confirm_transaction(&signed).await?;
    Ok(signature.to_string())
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

    let Some((quote, raw_quote)) =
        fetch_quote(&config.jupiter_base_url, input_mint, output_mint, trade_size, config.max_slippage_bps).await
    else {
        return (refusal_event(trade_size, 0, dry_run, "refused: quote fetch failed".to_string()), position, wallet);
    };

    let quote_price: u64 = match quote.out_amount.parse() {
        Ok(price) if price > 0 => price,
        _ => {
            return (
                refusal_event(
                    trade_size,
                    0,
                    dry_run,
                    "refused: quote returned a zero or unparsable amount".to_string(),
                ),
                position,
                wallet,
            );
        }
    };
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

    if !dry_run {
        if let Err(e) = submit_live_swap(config, &raw_quote).await {
            return (
                refusal_event(
                    trade_size,
                    quote_price,
                    dry_run,
                    format!(
                        "LIVE SUBMISSION FAILED — wallet state may be inconsistent with the \
                         chain, verify manually before continuing: {e}"
                    ),
                ),
                position,
                wallet,
            );
        }
    }

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
            coingecko_base_url: "https://api.coingecko.com/api/v3".into(),
            coingecko_api_key: None,
            backtest_days: 90,
            backtest_cost_bps: 15,
            backtest_tick_ms: 50,
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
    async fn next_buy_size_decompounds_after_a_losing_round_trip() {
        let server = MockServer::start().await;
        // 10% of a shrunk 900 equity = 90, not 100 — a mock that only
        // matches amount=90 proves sizing shrinks with a realized loss,
        // the symmetric case to next_buy_size_compounds_with_grown_equity.
        Mock::given(method("GET"))
            .and(path("/quote"))
            .and(wiremock::matchers::query_param("amount", "90"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "outAmount": "5000000", "priceImpactPct": "0.001"
            })))
            .mount(&server)
            .await;

        let mut config = test_config("/tmp/layatrade_test_kill_15");
        config.jupiter_base_url = server.uri();
        config.trade_size_pct = 0.1;
        let mut guard = SafetyGuardState::new();
        let state = Arc::new(RwLock::new(BotState::new()));

        // Equity already shrank to 900 from a prior losing round trip.
        let shrunk_wallet = WalletState { starting_capital: 1000, realized_pnl: -100 };
        let (event, _position, _wallet) =
            evaluate_trade(&config, &mut guard, &state, None, shrunk_wallet).await;

        assert_eq!(event.side, "buy");
        assert_eq!(event.size, 90);
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

    #[tokio::test]
    async fn dry_run_never_calls_the_swap_endpoint() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/quote"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "outAmount": "5000000", "priceImpactPct": "0.001"
            })))
            .mount(&server)
            .await;
        // Deliberately no /swap mock registered: if evaluate_trade ever
        // calls it in dry-run mode, wiremock's unmatched-request panic
        // (via .expect(0) below) proves it.
        Mock::given(method("POST"))
            .and(path("/swap"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;

        let mut config = test_config("/tmp/layatrade_test_kill_11");
        config.jupiter_base_url = server.uri();
        assert_eq!(config.execution_mode, ExecutionMode::DryRun);
        let mut guard = SafetyGuardState::new();
        let state = Arc::new(RwLock::new(BotState::new()));

        let (event, _position, _wallet) =
            evaluate_trade(&config, &mut guard, &state, None, flat_wallet(1000)).await;

        assert!(event.dry_run);
        server.verify().await; // enforces the expect(0) above
    }

    #[tokio::test]
    async fn live_mode_signs_the_returned_transaction_with_the_loaded_keypair() {
        use solana_sdk::message::{v0, VersionedMessage};
        use solana_sdk::signer::keypair::Keypair;
        use solana_sdk::signer::Signer;
        use solana_sdk::transaction::VersionedTransaction;

        // Build an unsigned message whose only required signer is our
        // throwaway keypair's pubkey, matching what Jupiter's real /swap
        // response contains before the caller signs it.
        let keypair = Keypair::new();
        let blockhash = solana_sdk::hash::Hash::default();
        let message = VersionedMessage::V0(
            v0::Message::try_compile(&keypair.pubkey(), &[], &[], blockhash).unwrap(),
        );
        let unsigned_tx = VersionedTransaction {
            signatures: vec![solana_sdk::signature::Signature::default()],
            message,
        };
        let serialized = bincode::serialize(&unsigned_tx).unwrap();
        let encoded = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &serialized);

        let signed = sign_swap_transaction(&encoded, &keypair)
            .expect("a well-formed swap transaction should sign cleanly");
        assert!(signed.verify_with_results().iter().all(|ok| *ok));
    }

    #[tokio::test]
    async fn fetch_quote_sends_slippage_bps_matching_config() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/quote"))
            .and(wiremock::matchers::query_param("slippageBps", "50"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "outAmount": "5000000", "priceImpactPct": "0.001"
            })))
            .mount(&server)
            .await;

        // No mock registered without the slippageBps param, so a request
        // missing it 404s and fetch_quote returns None — proving the
        // param is actually sent, not just documented.
        let result = fetch_quote(&server.uri(), "in", "out", 100, 50).await;
        assert!(result.is_some(), "fetch_quote must send slippageBps=50 to match the mock");
    }

    #[tokio::test]
    async fn submit_live_swap_reuses_the_approved_quote_without_refetching() {
        use solana_sdk::message::{v0, VersionedMessage};
        use solana_sdk::signer::keypair::Keypair;
        use solana_sdk::signer::Signer;
        use solana_sdk::transaction::VersionedTransaction;

        let keypair = Keypair::new();
        let blockhash = solana_sdk::hash::Hash::default();
        let message = VersionedMessage::V0(
            v0::Message::try_compile(&keypair.pubkey(), &[], &[], blockhash).unwrap(),
        );
        let unsigned_tx = VersionedTransaction {
            signatures: vec![solana_sdk::signature::Signature::default()],
            message,
        };
        let serialized = bincode::serialize(&unsigned_tx).unwrap();
        let encoded = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &serialized);

        let keypair_file = tempfile_with_keypair(&keypair);

        let server = MockServer::start().await;
        // Deliberately no /quote mock: submit_live_swap must not re-fetch
        // a quote — it already has the approved one from evaluate_trade.
        Mock::given(method("GET"))
            .and(path("/quote"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/swap"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "swapTransaction": encoded
            })))
            .mount(&server)
            .await;

        let mut config = test_config("/tmp/layatrade_test_kill_14");
        config.jupiter_base_url = server.uri();
        config.solana_keypair_path = Some(keypair_file.clone());

        let approved_quote = serde_json::json!({"outAmount": "5000000", "priceImpactPct": "0.001"});
        // This will fail at the RpcClient submission step (no real
        // network available in a unit test) — that's expected and fine;
        // the point of this test is the wiremock .expect(0) above, which
        // fails the test if /quote is ever called.
        let _ = submit_live_swap(&config, &approved_quote).await;

        server.verify().await;
        std::fs::remove_file(&keypair_file).ok();
    }

    fn tempfile_with_keypair(keypair: &solana_sdk::signer::keypair::Keypair) -> String {
        let path = format!("/tmp/layatrade_test_keypair_{}.json", std::process::id());
        std::fs::write(&path, serde_json::to_string(&keypair.to_bytes().to_vec()).unwrap()).unwrap();
        path
    }
}
