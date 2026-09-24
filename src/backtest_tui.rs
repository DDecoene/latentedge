use crate::backtest_executor::{advance_one_tick, BacktestState};
use crate::config::Config;
use crate::executor::SafetyGuardState;
use crate::laya_client::LayaClient;
use crate::tui::{is_quit_key, RawModeGuard};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::symbols;
use ratatui::widgets::{Axis, Block, Chart, Dataset, GraphType, List, ListItem};
use ratatui::Terminal;

/// Price (USD) and equity (quote atoms) live on wildly different scales
/// — plotting them raw on one y-axis makes one line look flat. Both are
/// shown as % change from their first value instead, which also makes
/// the strategy directly comparable to the buy & hold baseline at a
/// glance.
fn render_backtest_frame(frame: &mut ratatui::Frame, state: &BacktestState) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(frame.area());

    let initial_price = state.prices.first().map(|(_, p)| *p).filter(|p| *p > 0.0);
    let price_points: Vec<(f64, f64)> = match initial_price {
        Some(initial) => state
            .prices
            .iter()
            .take(state.current_index.max(1))
            .enumerate()
            .map(|(i, (_, price))| (i as f64, (price / initial - 1.0) * 100.0))
            .collect(),
        None => Vec::new(),
    };
    let starting_capital = state.wallet.starting_capital as f64;
    let equity_points: Vec<(f64, f64)> = if starting_capital > 0.0 {
        state
            .equity_curve
            .iter()
            .enumerate()
            .map(|(i, equity)| (i as f64, (equity / starting_capital - 1.0) * 100.0))
            .collect()
    } else {
        Vec::new()
    };

    let max_x = state.prices.len().max(1) as f64;
    let all_pct_values = price_points.iter().chain(equity_points.iter()).map(|(_, v)| *v);
    let min_y = all_pct_values.clone().fold(0.0_f64, f64::min) - 1.0;
    let max_y = all_pct_values.fold(0.0_f64, f64::max) + 1.0;

    let datasets = vec![
        Dataset::default()
            .name("price %")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&price_points),
        Dataset::default()
            .name("equity %")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(&equity_points),
    ];

    let chart = Chart::new(datasets)
        .block(Block::bordered().title("price & equity, % change from start"))
        .x_axis(Axis::default().bounds([0.0, max_x]))
        .y_axis(Axis::default().bounds([min_y, max_y]));
    frame.render_widget(chart, layout[0]);

    let position_line = match &state.position {
        Some(p) => format!("position: holding size={} entry_cost={}", p.size, p.entry_cost),
        None => "position: flat".to_string(),
    };
    let lines = vec![
        format!(
            "tick {}/{} equity={} (mark-to-market) buy_and_hold={:.0} realized_pnl={}",
            state.current_index,
            state.prices.len(),
            state.mark_to_market_equity(),
            state.buy_and_hold_equity(),
            state.wallet.realized_pnl
        ),
        position_line,
        format!(
            "trades={} wins={} losses={} win_rate={:.1}% zero_confidence_predicts={}",
            state.trades.len(),
            state.wins,
            state.losses,
            state.win_rate() * 100.0,
            state.zero_confidence_predicts,
        ),
    ];
    let items: Vec<ListItem> = lines.into_iter().map(ListItem::new).collect();
    frame.render_widget(List::new(items).block(Block::bordered().title("summary")), layout[1]);
}

pub async fn run_backtest_tui(
    state: &mut BacktestState,
    config: &Config,
    guard: &mut SafetyGuardState,
    laya: &LayaClient,
) -> anyhow::Result<()> {
    let _raw_mode = RawModeGuard::new()?;
    let backend = CrosstermBackend::new(std::io::stdout());
    let mut terminal = Terminal::new(backend)?;

    loop {
        terminal.draw(|frame| render_backtest_frame(frame, state))?;

        if crossterm::event::poll(std::time::Duration::from_millis(1))? {
            if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                if is_quit_key(key) {
                    break;
                }
            }
        }

        let more = advance_one_tick(state, config, guard, laya).await;
        if !more {
            break;
        }

        tokio::time::sleep(std::time::Duration::from_millis(config.backtest_tick_ms)).await;
    }

    Ok(())
}
