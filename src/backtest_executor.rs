use crate::config::Config;
use crate::executor::SafetyGuardState;
use crate::laya_client::LayaClient;
use crate::phoenix_decode::decode_from_ladder_json;
use crate::types::{Position, TradeEvent, WalletState};
use chrono::{DateTime, Utc};

/// Matches `JupiterQuotePoller::fetch`'s convention exactly: `ask` is the
/// base-atom amount received for spending 1 USDC (1_000_000 quote atoms)
/// — i.e. lamports-per-USDC, the inverse of a USD-per-SOL spot price —
/// not a scaled USD price. Feeding Laya anything else would mean the
/// backtest answers a different question than live does.
pub fn synthetic_ladder_json(spot_price: f64, cost_bps: u32) -> String {
    let ask = if spot_price > 0.0 {
        (1_000_000_000.0 / spot_price).round().max(0.0) as u64
    } else {
        u64::MAX
    };
    let cost_frac = cost_bps as f64 / 10_000.0;
    let impact_atoms = ((ask as f64) * cost_frac).round().max(1.0) as u64;
    let bid = ask.saturating_sub(impact_atoms);
    serde_json::json!({
        "bids": [[bid, 1]],
        "asks": [[ask, 1]],
    })
    .to_string()
}

const QUOTE_ATOMS_PER_UNIT: f64 = 1_000_000.0; // USDC-style, 6 decimals
const BASE_ATOMS_PER_UNIT: f64 = 1_000_000_000.0; // SOL-style, 9 decimals

fn backtest_refusal_event(trade_size: u64, price: u64, note: String) -> TradeEvent {
    TradeEvent { timestamp: Utc::now(), side: "none".to_string(), size: trade_size, price, dry_run: true, note }
}

pub fn evaluate_backtest_trade(
    config: &Config,
    guard: &mut SafetyGuardState,
    position: Option<Position>,
    wallet: WalletState,
    spot_price: f64,
) -> (TradeEvent, Option<Position>, WalletState) {
    let trade_size = match position {
        None => ((wallet.equity() as f64) * config.trade_size_pct) as u64,
        Some(p) => p.size,
    };

    // No rate-limit check here: its window is wall-clock time, which has
    // no relationship to replayed historical ticks. Size and daily-loss
    // still apply — both cap real notional risk regardless of clock.
    let guard_result = match position {
        None => guard.check_size(config, trade_size).and_then(|_| guard.check_daily_loss(config)),
        Some(_) => guard.check_daily_loss(config),
    };
    if let Err(reason) = guard_result {
        return (backtest_refusal_event(trade_size, 0, format!("refused: {reason}")), position, wallet);
    }

    let cost_frac = config.backtest_cost_bps as f64 / 10_000.0;

    match position {
        None => {
            let effective_buy_price = spot_price * (1.0 + cost_frac);
            if !(effective_buy_price > 0.0) || trade_size == 0 {
                return (
                    backtest_refusal_event(trade_size, 0, "refused: invalid buy price or size".to_string()),
                    position,
                    wallet,
                );
            }
            let usd_spent = trade_size as f64 / QUOTE_ATOMS_PER_UNIT;
            let sol_bought = usd_spent / effective_buy_price;
            let size = (sol_bought * BASE_ATOMS_PER_UNIT) as u64;

            guard.record_trade();
            let new_position = Position { size, entry_cost: trade_size };
            (
                TradeEvent {
                    timestamp: Utc::now(),
                    side: "buy".to_string(),
                    size: trade_size,
                    price: size,
                    dry_run: true,
                    note: "approved".to_string(),
                },
                Some(new_position),
                wallet,
            )
        }
        Some(p) => {
            let effective_sell_price = (spot_price * (1.0 - cost_frac)).max(0.0);
            let sol_amount = p.size as f64 / BASE_ATOMS_PER_UNIT;
            let usd_received = sol_amount * effective_sell_price;
            let proceeds = (usd_received * QUOTE_ATOMS_PER_UNIT) as u64;

            guard.record_trade();
            let mut new_wallet = wallet;
            new_wallet.realized_pnl += proceeds as i64 - p.entry_cost as i64;
            if proceeds < p.entry_cost {
                guard.record_loss(p.entry_cost - proceeds);
            }
            (
                TradeEvent {
                    timestamp: Utc::now(),
                    side: "sell".to_string(),
                    size: trade_size,
                    price: proceeds,
                    dry_run: true,
                    note: "approved".to_string(),
                },
                None,
                new_wallet,
            )
        }
    }
}

#[derive(Debug, Clone)]
pub struct BacktestState {
    pub prices: Vec<(DateTime<Utc>, f64)>,
    pub current_index: usize,
    pub position: Option<Position>,
    pub wallet: WalletState,
    pub trades: Vec<TradeEvent>,
    pub equity_curve: Vec<f64>,
    pub wins: u32,
    pub losses: u32,
    /// Ticks where Laya returned `confidence == 0.0` — this includes both
    /// a genuine zero-confidence answer and a failed/timed-out call
    /// (`LayaClient::get_signal` collapses both to the same value), so a
    /// high count here is a signal to check the Laya server's health or
    /// latency, not proof the strategy itself was simply inactive.
    pub zero_confidence_predicts: u32,
}

impl BacktestState {
    pub fn new(prices: Vec<(DateTime<Utc>, f64)>, wallet: WalletState) -> Self {
        Self {
            prices,
            current_index: 0,
            position: None,
            wallet,
            trades: Vec::new(),
            equity_curve: Vec::new(),
            wins: 0,
            losses: 0,
            zero_confidence_predicts: 0,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.current_index >= self.prices.len()
    }

    pub fn win_rate(&self) -> f64 {
        let total = self.wins + self.losses;
        if total == 0 {
            0.0
        } else {
            self.wins as f64 / total as f64
        }
    }

    pub fn buy_and_hold_equity(&self) -> f64 {
        if self.prices.is_empty() {
            return self.wallet.starting_capital as f64;
        }
        let initial = self.prices[0].1;
        let last_index = self.current_index.min(self.prices.len() - 1);
        let current = self.prices[last_index].1;
        if initial <= 0.0 {
            return self.wallet.starting_capital as f64;
        }
        (self.wallet.starting_capital as f64) * (current / initial)
    }

    /// `wallet.equity()` alone ignores an open position — a buy never
    /// debits the wallet, it only records `Position`, so while holding,
    /// `wallet.equity()` is stale (last updated at the previous sell).
    /// This marks the open position to the last-replayed price: the
    /// capital that bought it (`entry_cost`) is backed out and replaced
    /// with what that position is worth right now.
    pub fn mark_to_market_equity(&self) -> u64 {
        let Some(p) = self.position else {
            return self.wallet.equity();
        };
        if self.prices.is_empty() {
            return self.wallet.equity();
        }
        let last_index = self.current_index.saturating_sub(1).min(self.prices.len() - 1);
        let spot_price = self.prices[last_index].1.max(0.0);
        let sol_amount = p.size as f64 / BASE_ATOMS_PER_UNIT;
        let position_value = (sol_amount * spot_price * QUOTE_ATOMS_PER_UNIT) as u64;
        ((self.wallet.equity() as i64) - (p.entry_cost as i64) + (position_value as i64)).max(0) as u64
    }
}

pub async fn advance_one_tick(
    state: &mut BacktestState,
    config: &Config,
    guard: &mut SafetyGuardState,
    laya: &LayaClient,
) -> bool {
    if state.is_finished() {
        return false;
    }
    let (timestamp, spot_price) = state.prices[state.current_index];
    let raw_json = synthetic_ladder_json(spot_price, config.backtest_cost_bps);
    let market = format!("{}/{}", config.base_mint, config.quote_mint);
    let slot = timestamp.timestamp().max(0) as u64;
    let mut snapshot = decode_from_ladder_json(&market, slot, &raw_json)
        .expect("synthetic ladder must always be valid — bid<ask guaranteed by construction");
    snapshot.timestamp = timestamp;

    let sig = laya.get_signal(&snapshot).await;
    if sig.confidence == 0.0 {
        state.zero_confidence_predicts += 1;
    }

    if sig.should_trade {
        let prev_realized = state.wallet.realized_pnl;
        let (event, new_position, new_wallet) =
            evaluate_backtest_trade(config, guard, state.position, state.wallet, spot_price);
        let is_sell = event.side == "sell";
        state.position = new_position;
        state.wallet = new_wallet;
        if is_sell {
            let delta = state.wallet.realized_pnl - prev_realized;
            if delta > 0 {
                state.wins += 1;
            } else if delta < 0 {
                state.losses += 1;
            }
        }
        state.trades.push(event);
    }

    state.current_index += 1;
    state.equity_curve.push(state.mark_to_market_equity() as f64);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        Config::from_map(&std::collections::HashMap::new()).expect("defaults must parse")
    }

    fn flat_wallet(starting_capital: u64) -> WalletState {
        WalletState { starting_capital, realized_pnl: 0 }
    }

    #[test]
    fn produces_a_valid_uncrossed_ladder_at_a_normal_price() {
        let raw = synthetic_ladder_json(150.0, 15);
        let snapshot = decode_from_ladder_json("backtest", 0, &raw).expect("must be valid");
        assert!(snapshot.bids[0].0 < snapshot.asks[0].0);
    }

    #[test]
    fn stays_uncrossed_at_extreme_cost_bps() {
        let raw = synthetic_ladder_json(150.0, 50_000); // 500%, absurd but must not crash or cross
        let snapshot = decode_from_ladder_json("backtest", 0, &raw).expect("must still be valid");
        assert!(snapshot.bids[0].0 < snapshot.asks[0].0);
    }

    #[test]
    fn stays_uncrossed_at_a_zero_spot_price() {
        let raw = synthetic_ladder_json(0.0, 15);
        let snapshot = decode_from_ladder_json("backtest", 0, &raw).expect("must still be valid");
        assert!(snapshot.bids[0].0 < snapshot.asks[0].0);
    }

    #[test]
    fn ladder_matches_the_live_pollers_convention() {
        // Live's JupiterQuotePoller::fetch quotes spending 1 USDC
        // (1_000_000 quote atoms) into base_mint and uses the returned
        // base-atom amount as `ask` — i.e. ask = lamports of SOL per 1
        // USDC, the inverse of a USD-per-SOL spot price, not a scaled
        // USD price. At $100/SOL, 1 USDC buys 0.01 SOL = 10_000_000
        // lamports. The backtest must feed Laya the same magnitude and
        // direction, or it's answering a different question than live.
        let raw = synthetic_ladder_json(100.0, 15);
        let snapshot = decode_from_ladder_json("backtest", 0, &raw).expect("must be valid");
        let (ask, _) = snapshot.asks[0];
        assert!(
            (9_900_000..=10_100_000).contains(&ask),
            "ask should be ~10_000_000 lamports (1 USDC worth of SOL at $100), got {ask}"
        );
    }

    #[test]
    fn flat_position_buys_a_fraction_of_equity_at_the_cost_adjusted_price() {
        let mut config = test_config();
        config.trade_size_pct = 0.1;
        config.backtest_cost_bps = 0; // isolate sizing from cost for this test
        let mut guard = SafetyGuardState::new();

        let (event, position, wallet) =
            evaluate_backtest_trade(&config, &mut guard, None, flat_wallet(1_000_000_000), 100.0);

        assert_eq!(event.side, "buy");
        assert_eq!(event.size, 100_000_000); // 10% of 1000 USD equity, in quote atoms
        let position = position.expect("buy must open a position");
        // spend $100 at $100/SOL with zero cost = 1 SOL = 1_000_000_000 base atoms
        assert_eq!(position.size, 1_000_000_000);
        assert_eq!(wallet.equity(), 1_000_000_000, "buying alone doesn't realize P&L");
    }

    #[test]
    fn round_trip_at_a_flat_price_loses_exactly_the_two_way_cost() {
        let mut config = test_config();
        config.trade_size_pct = 1.0; // spend all equity, makes the cost's effect easy to compute
        config.backtest_cost_bps = 100; // 1% per leg
        config.max_trade_size = 2_000_000_000; // deliberately above the full-equity spend this test exercises
        let mut guard = SafetyGuardState::new();
        let wallet = flat_wallet(1_000_000_000); // 1000 USD

        let (buy_event, position, wallet) =
            evaluate_backtest_trade(&config, &mut guard, None, wallet, 100.0);
        assert_eq!(buy_event.side, "buy");
        let position = position.expect("buy must open a position");

        let (sell_event, position, wallet) =
            evaluate_backtest_trade(&config, &mut guard, Some(position), wallet, 100.0);
        assert_eq!(sell_event.side, "sell");
        assert!(position.is_none(), "position must clear after the sell");

        // Buy at 101 (100 * 1.01), sell at 99 (100 * 0.99) — round trip
        // loses ~2% even though the spot price never moved, proving the
        // cost model is applied on both legs, not just one.
        assert!(
            wallet.equity() < 1_000_000_000,
            "a flat-price round trip must still lose money to the two-way cost, got {}",
            wallet.equity()
        );
        let loss_pct = 1.0 - (wallet.equity() as f64 / 1_000_000_000.0);
        assert!(loss_pct > 0.019 && loss_pct < 0.021, "expected ~2% loss, got {}", loss_pct * 100.0);
    }

    #[test]
    fn holding_position_sells_and_records_loss_when_price_drops() {
        let mut config = test_config();
        config.backtest_cost_bps = 0;
        let mut guard = SafetyGuardState::new();
        let position = Position { size: 1_000_000_000, entry_cost: 100_000_000 }; // bought 1 SOL for $100
        let wallet = flat_wallet(1_000_000_000);

        let (event, position, wallet) =
            evaluate_backtest_trade(&config, &mut guard, Some(position), wallet, 90.0); // price dropped to $90

        assert_eq!(event.side, "sell");
        assert!(position.is_none());
        assert_eq!(guard.daily_loss_accrued(), 10_000_000); // lost $10 (in quote atoms)
        assert_eq!(wallet.equity(), 990_000_000);
    }

    #[test]
    fn holding_position_sells_and_grows_equity_when_price_rises() {
        let mut config = test_config();
        config.backtest_cost_bps = 0;
        let mut guard = SafetyGuardState::new();
        let position = Position { size: 1_000_000_000, entry_cost: 100_000_000 };
        let wallet = flat_wallet(1_000_000_000);

        let (event, position, wallet) =
            evaluate_backtest_trade(&config, &mut guard, Some(position), wallet, 120.0);

        assert_eq!(event.side, "sell");
        assert!(position.is_none());
        assert_eq!(guard.daily_loss_accrued(), 0);
        assert_eq!(wallet.equity(), 1_020_000_000);
    }

    #[test]
    fn next_buy_size_compounds_with_grown_equity() {
        let mut config = test_config();
        config.trade_size_pct = 0.1;
        config.backtest_cost_bps = 0;
        let mut guard = SafetyGuardState::new();
        let grown_wallet = WalletState { starting_capital: 1_000_000_000, realized_pnl: 100_000_000 };

        let (event, _position, _wallet) =
            evaluate_backtest_trade(&config, &mut guard, None, grown_wallet, 100.0);

        assert_eq!(event.side, "buy");
        assert_eq!(event.size, 110_000_000); // 10% of 1100, not 1000 — proves sizing tracks equity
    }

    #[test]
    fn a_zero_or_negative_price_refuses_a_buy_instead_of_opening_a_broken_position() {
        let config = test_config();
        let mut guard = SafetyGuardState::new();

        let (event, position, _wallet) =
            evaluate_backtest_trade(&config, &mut guard, None, flat_wallet(1_000_000_000), 0.0);

        assert_eq!(event.side, "none");
        assert!(position.is_none());
    }

    #[test]
    fn equity_floor_holds_under_a_losing_streak() {
        let wallet = WalletState { starting_capital: 1_000_000_000, realized_pnl: -5_000_000_000 };
        assert_eq!(wallet.equity(), 0);
        let config = test_config();
        let mut guard = SafetyGuardState::new();

        let (event, _position, _wallet) =
            evaluate_backtest_trade(&config, &mut guard, None, wallet, 100.0);
        assert_eq!(event.size, 0);
    }

    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sample_prices() -> Vec<(DateTime<Utc>, f64)> {
        vec![(Utc::now(), 100.0), (Utc::now(), 101.0), (Utc::now(), 99.0)]
    }

    #[test]
    fn new_state_starts_flat_and_unfinished() {
        let state = BacktestState::new(sample_prices(), flat_wallet(1000));
        assert!(state.position.is_none());
        assert!(!state.is_finished());
        assert_eq!(state.current_index, 0);
    }

    #[test]
    fn win_rate_is_zero_with_no_closed_trades() {
        let state = BacktestState::new(sample_prices(), flat_wallet(1000));
        assert_eq!(state.win_rate(), 0.0);
    }

    #[test]
    fn buy_and_hold_equity_tracks_price_change_from_the_first_tick() {
        let mut state = BacktestState::new(sample_prices(), flat_wallet(1000));
        state.current_index = 1; // price moved 100 -> 101
        let expected = 1000.0 * (101.0 / 100.0);
        assert!((state.buy_and_hold_equity() - expected).abs() < 0.001);
    }

    #[tokio::test]
    async fn advance_one_tick_finishes_after_the_last_price_point() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/predict"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"confidence": 0.1})))
            .mount(&server)
            .await;
        let laya = LayaClient::new(server.uri(), 0.85);
        let config = test_config();
        let mut guard = SafetyGuardState::new();
        let mut state = BacktestState::new(vec![(Utc::now(), 100.0)], flat_wallet(1_000_000_000));

        assert!(advance_one_tick(&mut state, &config, &mut guard, &laya).await);
        assert!(state.is_finished());
        assert_eq!(state.equity_curve.len(), 1);

        assert!(!advance_one_tick(&mut state, &config, &mut guard, &laya).await);
    }

    #[tokio::test]
    async fn advance_one_tick_opens_a_position_when_laya_says_trade() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/predict"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"confidence": 0.99})))
            .mount(&server)
            .await;
        let laya = LayaClient::new(server.uri(), 0.85);
        let mut config = test_config();
        config.backtest_cost_bps = 0;
        let mut guard = SafetyGuardState::new();
        let mut state = BacktestState::new(vec![(Utc::now(), 100.0)], flat_wallet(1_000_000_000));

        advance_one_tick(&mut state, &config, &mut guard, &laya).await;

        assert!(state.position.is_some());
        assert_eq!(state.trades.len(), 1);
        assert_eq!(state.trades[0].side, "buy");
    }

    #[test]
    fn mark_to_market_equals_wallet_equity_when_flat() {
        let state = BacktestState::new(sample_prices(), flat_wallet(1000));
        assert_eq!(state.mark_to_market_equity(), 1000);
    }

    #[test]
    fn mark_to_market_reflects_an_open_positions_current_value() {
        let mut state = BacktestState::new(vec![(Utc::now(), 100.0), (Utc::now(), 110.0)], flat_wallet(1_000_000_000));
        // Bought 1 SOL for $100 (entry_cost 100_000_000); price is now $110.
        state.position = Some(Position { size: 1_000_000_000, entry_cost: 100_000_000 });
        state.current_index = 2; // both ticks "replayed" — last price is prices[1] = 110.0

        // 1000 (unchanged wallet.equity()) - 100 (entry_cost) + 110 (current value) = 1010
        assert_eq!(state.mark_to_market_equity(), 1_010_000_000);
    }

    #[test]
    fn mark_to_market_matches_wallet_equity_at_the_unchanged_entry_price() {
        let mut state = BacktestState::new(vec![(Utc::now(), 100.0)], flat_wallet(1_000_000_000));
        state.position = Some(Position { size: 1_000_000_000, entry_cost: 100_000_000 });
        state.current_index = 1; // last price is prices[0] = 100.0, same as the buy price

        // Converting cash to SOL and back at the same price changes nothing.
        assert_eq!(state.mark_to_market_equity(), 1_000_000_000);
    }

    #[tokio::test]
    async fn advance_one_tick_pushes_mark_to_market_equity_not_stale_wallet_equity() {
        // Tick 1: high confidence, buys and opens a position. Tick 2: low
        // confidence, no trade — the position stays open while the price
        // rises to 110. If equity_curve read the stale wallet.equity()
        // instead of the mark-to-market value, tick 2 would wrongly still
        // show 1_000_000_000 even though the open position is now worth
        // more.
        let buy_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/predict"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"confidence": 0.99})))
            .mount(&buy_server)
            .await;
        let hold_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/predict"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"confidence": 0.1})))
            .mount(&hold_server)
            .await;

        let buy_laya = LayaClient::new(buy_server.uri(), 0.85);
        let hold_laya = LayaClient::new(hold_server.uri(), 0.85);
        let mut config = test_config();
        config.backtest_cost_bps = 0;
        let mut guard = SafetyGuardState::new();
        let mut state = BacktestState::new(
            vec![(Utc::now(), 100.0), (Utc::now(), 110.0)],
            flat_wallet(1_000_000_000),
        );

        advance_one_tick(&mut state, &config, &mut guard, &buy_laya).await; // buys
        assert_eq!(state.equity_curve[0], 1_000_000_000.0, "buying alone doesn't change equity");

        advance_one_tick(&mut state, &config, &mut guard, &hold_laya).await; // below threshold: still holding
        assert!(state.position.is_some(), "position must still be open");
        assert_eq!(
            state.equity_curve[1], 1_010_000_000.0,
            "equity_curve must mark the open position to the new price, not repeat stale wallet.equity()"
        );
    }

    #[tokio::test]
    async fn zero_confidence_predicts_are_counted() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/predict"))
            .respond_with(ResponseTemplate::new(500)) // a failed call collapses to confidence 0.0
            .mount(&server)
            .await;
        let laya = LayaClient::new(server.uri(), 0.85);
        let config = test_config();
        let mut guard = SafetyGuardState::new();
        let mut state = BacktestState::new(vec![(Utc::now(), 100.0)], flat_wallet(1_000_000_000));

        advance_one_tick(&mut state, &config, &mut guard, &laya).await;

        assert_eq!(state.zero_confidence_predicts, 1);
    }
}
