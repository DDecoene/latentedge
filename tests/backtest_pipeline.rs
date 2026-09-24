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
    // Alternates should_trade so the replay actually opens and closes
    // positions, exercising both the buy and sell paths.
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
