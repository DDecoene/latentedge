use crate::config::Config;
use crate::executor::SafetyGuardState;
use crate::types::{Position, TradeEvent, WalletState};
use chrono::Utc;

pub fn synthetic_ladder_json(spot_price: f64, cost_bps: u32) -> String {
    let price_atoms = (spot_price * 1_000_000.0).round().max(0.0) as u64;
    let cost_atoms = ((price_atoms as f64) * (cost_bps as f64) / 10_000.0).round().max(1.0) as u64;
    let bid = price_atoms.saturating_sub(cost_atoms);
    let ask = price_atoms.saturating_add(cost_atoms);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phoenix_decode::decode_from_ladder_json;

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
}
