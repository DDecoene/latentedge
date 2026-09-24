use layatrade_rs::backtest_executor::BacktestState;
use layatrade_rs::backtest_tui::run_backtest_tui;
use layatrade_rs::config::Config;
use layatrade_rs::executor::SafetyGuardState;
use layatrade_rs::historical_data::fetch_or_load_cached;
use layatrade_rs::laya_client::LayaClient;
use layatrade_rs::types::WalletState;

fn format_summary(state: &BacktestState) -> String {
    let position_note = if state.position.is_some() {
        " (a position was still open at the end — marked to the last replayed price)"
    } else {
        ""
    };
    format!(
        "Backtest complete: {}/{} ticks replayed\n\
         Final equity: {} (mark-to-market, started at {}){}\n\
         Buy & hold would be worth: {:.0}\n\
         Trades: {} (wins {} / losses {}, win rate {:.1}%)\n\
         Realized P&L: {}\n\
         Zero-confidence Laya predicts: {} (includes any failed/timed-out calls, not just genuine zero answers)",
        state.current_index,
        state.prices.len(),
        state.mark_to_market_equity(),
        state.wallet.starting_capital,
        position_note,
        state.buy_and_hold_equity(),
        state.trades.len(),
        state.wins,
        state.losses,
        state.win_rate() * 100.0,
        state.wallet.realized_pnl,
        state.zero_confidence_predicts,
    )
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let config = Config::from_env()?;

    let laya = LayaClient::new(config.laya_server_url.clone(), config.laya_confidence_threshold);
    laya.health_check().await.map_err(|e| {
        anyhow::anyhow!("Laya server not reachable at {}: {e}", config.laya_server_url)
    })?;

    let cache_path = std::path::PathBuf::from(format!("backtest_cache/solana-{}d.json", config.backtest_days));
    let prices = fetch_or_load_cached(
        &config.coingecko_base_url,
        config.coingecko_api_key.as_deref(),
        config.backtest_days,
        &cache_path,
    )
    .await?;

    if prices.is_empty() {
        anyhow::bail!("historical data fetch returned no price points");
    }

    let wallet = WalletState { starting_capital: config.starting_capital, realized_pnl: 0 };
    let mut state = BacktestState::new(prices, wallet);
    let mut guard = SafetyGuardState::new();

    run_backtest_tui(&mut state, &config, &mut guard, &laya).await?;

    println!("{}", format_summary(&state));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn summary_reflects_whatever_portion_of_the_replay_completed() {
        let mut state = BacktestState::new(
            vec![(Utc::now(), 100.0), (Utc::now(), 110.0), (Utc::now(), 90.0)],
            WalletState { starting_capital: 1000, realized_pnl: 0 },
        );
        // Simulate stopping after only the first tick (an early quit):
        state.current_index = 1;
        state.equity_curve.push(1000.0);

        let summary = format_summary(&state);

        assert!(summary.contains("1"), "must reflect the actual tick count reached: {summary}");
        assert!(summary.contains(&state.wallet.equity().to_string()));
        assert!(summary.contains(&format!("{:.1}", state.win_rate() * 100.0)));
    }
}
