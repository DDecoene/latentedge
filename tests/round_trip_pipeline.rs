use layatrade_rs::config::{Config, ExecutionMode};
use layatrade_rs::executor::{evaluate_trade, SafetyGuardState};
use layatrade_rs::types::{BotState, WalletState};
use std::sync::Arc;
use tokio::sync::RwLock;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn test_config(jupiter_base_url: String) -> Config {
    let mut vars = std::collections::HashMap::new();
    vars.insert("JUPITER_BASE_URL".to_string(), jupiter_base_url);
    vars.insert("STARTING_CAPITAL".to_string(), "1000".to_string());
    Config::from_map(&vars).expect("should parse with defaults + override")
}

#[tokio::test]
async fn buy_then_sell_completes_a_round_trip_and_compounds_equity() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/quote"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "outAmount": "5000000", "priceImpactPct": "0.0001"
        })))
        .mount(&server)
        .await;

    let config = test_config(server.uri());
    assert_eq!(config.execution_mode, ExecutionMode::DryRun);
    let mut guard = SafetyGuardState::new();
    let state = Arc::new(RwLock::new(BotState::new()));
    let wallet = WalletState { starting_capital: config.starting_capital, realized_pnl: 0 };

    let (buy_event, position, wallet) =
        evaluate_trade(&config, &mut guard, &state, None, wallet).await;
    assert_eq!(buy_event.side, "buy");
    let position = position.expect("buy should open a position");
    assert_eq!(wallet.equity(), config.starting_capital, "buying alone doesn't change equity");

    let (sell_event, position_after_sell, wallet_after_sell) =
        evaluate_trade(&config, &mut guard, &state, Some(position), wallet).await;
    assert_eq!(sell_event.side, "sell");
    assert!(position_after_sell.is_none(), "position must clear after the sell");
    // The mocked quote returns the same outAmount (5_000_000) on both the
    // buy and the sell call, so proceeds (5_000_000) exceed entry_cost
    // (10% of 1000 = 100): this sell is a "win" and equity grows.
    assert!(
        wallet_after_sell.equity() > config.starting_capital,
        "a winning sell must grow equity, got {}",
        wallet_after_sell.equity()
    );
}
