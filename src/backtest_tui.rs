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

fn render_backtest_frame(frame: &mut ratatui::Frame, state: &BacktestState) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(frame.area());

    let price_points: Vec<(f64, f64)> = state
        .prices
        .iter()
        .take(state.current_index.max(1))
        .enumerate()
        .map(|(i, (_, price))| (i as f64, *price))
        .collect();
    let equity_points: Vec<(f64, f64)> = state
        .equity_curve
        .iter()
        .enumerate()
        .map(|(i, equity)| (i as f64, *equity))
        .collect();

    let max_x = state.prices.len().max(1) as f64;
    let max_price = state.prices.iter().map(|(_, p)| *p).fold(0.0, f64::max).max(1.0);
    let max_equity = state.equity_curve.iter().cloned().fold(state.wallet.starting_capital as f64, f64::max);

    let datasets = vec![
        Dataset::default()
            .name("price")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&price_points),
        Dataset::default()
            .name("equity")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(&equity_points),
    ];

    let chart = Chart::new(datasets)
        .block(Block::bordered().title("price & equity"))
        .x_axis(Axis::default().bounds([0.0, max_x]))
        .y_axis(Axis::default().bounds([0.0, max_price.max(max_equity) * 1.1]));
    frame.render_widget(chart, layout[0]);

    let position_line = match &state.position {
        Some(p) => format!("position: holding size={} entry_cost={}", p.size, p.entry_cost),
        None => "position: flat".to_string(),
    };
    let lines = vec![
        format!(
            "tick {}/{} equity={} buy_and_hold={:.0} realized_pnl={}",
            state.current_index,
            state.prices.len(),
            state.wallet.equity(),
            state.buy_and_hold_equity(),
            state.wallet.realized_pnl
        ),
        position_line,
        format!(
            "trades={} wins={} losses={} win_rate={:.1}%",
            state.trades.len(),
            state.wins,
            state.losses,
            state.win_rate() * 100.0
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
