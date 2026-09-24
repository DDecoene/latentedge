use chrono::Utc;
use layatrade_rs::backtest_executor::{advance_one_tick, BacktestState};
use layatrade_rs::config::Config;
use layatrade_rs::executor::SafetyGuardState;
use layatrade_rs::laya_client::LayaClient;
use layatrade_rs::types::WalletState;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn test_config() -> Config {
    let mut vars = std::collections::HashMap::new();
    vars.insert("STARTING_CAPITAL".to_string(), "1000000000".to_string());
    Config::from_map(&vars).expect("should parse with defaults + override")
}

#[tokio::test]
async fn a_ten_point_replay_produces_a_sane_report_with_no_panics() {
    let server = MockServer::start().await;
    // Confidence stays above threshold every tick, so should_trade is
    // true throughout; since a flat state buys and a holding state
    // sells, this alone alternates buy/sell without needing the mock to
    // vary its response.
    Mock::given(method("POST"))
        .and(path("/predict"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"confidence": 0.99})))
        .mount(&server)
        .await;

    let laya = LayaClient::new(server.uri(), 0.85);
    let config = test_config();
    let mut guard = SafetyGuardState::new();

    let prices: Vec<(chrono::DateTime<chrono::Utc>, f64)> =
        (0..10).map(|i| (Utc::now(), 100.0 + (i as f64))).collect();
    let wallet = WalletState { starting_capital: config.starting_capital, realized_pnl: 0 };
    let mut state = BacktestState::new(prices, wallet);

    let mut ticks = 0;
    while advance_one_tick(&mut state, &config, &mut guard, &laya).await {
        ticks += 1;
        assert!(ticks <= 10, "replay must not loop past its own price series");
    }

    assert_eq!(state.current_index, 10);
    assert!(state.is_finished());
    assert_eq!(state.equity_curve.len(), 10);
    assert!(!state.trades.is_empty(), "confidence 0.99 every tick must produce at least one trade");
    // win_rate/buy_and_hold_equity must not panic on a fully-replayed series:
    let _ = state.win_rate();
    let _ = state.buy_and_hold_equity();
}

#[tokio::test]
async fn a_four_point_round_trip_produces_the_exact_expected_win_and_loss() {
    // Zero cost and prices chosen so every division lands on a whole
    // number of atoms, pinning exact expected values instead of only
    // checking "it didn't panic" — this is the test the plan's Review
    // Focus item ("win rate and final equity must be trustworthy") is
    // actually about.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/predict"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"confidence": 0.99})))
        .mount(&server)
        .await;

    let laya = LayaClient::new(server.uri(), 0.85);
    let mut vars = std::collections::HashMap::new();
    vars.insert("STARTING_CAPITAL".to_string(), "1000000000".to_string());
    vars.insert("BACKTEST_COST_BPS".to_string(), "0".to_string());
    let config = Config::from_map(&vars).expect("should parse");
    let mut guard = SafetyGuardState::new();

    // tick0 buy@100, tick1 sell@200 (a win), tick2 buy@100, tick3 sell@50 (a loss).
    let prices = vec![
        (Utc::now(), 100.0),
        (Utc::now(), 200.0),
        (Utc::now(), 100.0),
        (Utc::now(), 50.0),
    ];
    let wallet = WalletState { starting_capital: config.starting_capital, realized_pnl: 0 };
    let mut state = BacktestState::new(prices, wallet);

    while advance_one_tick(&mut state, &config, &mut guard, &laya).await {}

    assert_eq!(state.trades.len(), 4, "all four ticks must trade: buy, sell, buy, sell");
    assert_eq!(state.trades[0].side, "buy");
    assert_eq!(state.trades[1].side, "sell");
    assert_eq!(state.trades[2].side, "buy");
    assert_eq!(state.trades[3].side, "sell");

    // tick0: spend 10% of 1000 = 100 at $100/SOL -> exactly 1 SOL (1_000_000_000 base atoms).
    // tick1: sell 1 SOL at $200 -> 200 proceeds vs 100 entry_cost = +100 realized. Equity: 1_100_000_000. A win.
    // tick2: spend 10% of 1100 = 110 at $100/SOL -> exactly 1.1 SOL (1_100_000_000 base atoms).
    // tick3: sell 1.1 SOL at $50 -> 55 proceeds vs 110 entry_cost = -55 realized. Equity: 1_045_000_000. A loss.
    assert_eq!(state.wins, 1);
    assert_eq!(state.losses, 1);
    assert_eq!(state.win_rate(), 0.5);
    assert_eq!(state.wallet.realized_pnl, 45_000_000);
    assert_eq!(state.mark_to_market_equity(), 1_045_000_000);
    assert!(state.position.is_none(), "the last tick's sell must have closed the position");
}
