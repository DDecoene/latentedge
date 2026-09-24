use crate::types::BotState;
use ratatui::backend::CrosstermBackend;
use ratatui::widgets::{List, ListItem};
use ratatui::Terminal;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

pub struct DashboardLines {
    pub lines: Vec<String>,
}

pub fn render_lines(state: &BotState) -> DashboardLines {
    let mut lines = Vec::new();

    match &state.last_snapshot {
        Some(snap) => {
            let best_bid = snap.bids.first().copied().unwrap_or((0, 0));
            let best_ask = snap.asks.first().copied().unwrap_or((0, 0));
            lines.push(format!("market {} slot {}", snap.market, snap.slot));
            lines.push(format!("best bid {:?} / best ask {:?}", best_bid, best_ask));
        }
        None => lines.push("no snapshot yet".to_string()),
    }

    match &state.last_signal {
        Some(sig) => lines.push(format!(
            "signal: should_trade={} confidence={:.2}",
            sig.should_trade, sig.confidence
        )),
        None => lines.push("no signal yet".to_string()),
    }

    match &state.stream_health {
        Some(health) => lines.push(format!(
            "stream: last update {} reconnects {}",
            health.last_update, health.reconnect_count
        )),
        None => lines.push("stream health unknown".to_string()),
    }

    if state.kill_switch_active {
        lines.push("!!! KILL SWITCH ACTIVE !!!".to_string());
    }

    match &state.position {
        Some(p) => lines.push(format!("position: holding size={} entry_cost={}", p.size, p.entry_cost)),
        None => lines.push("position: flat".to_string()),
    }
    lines.push(format!(
        "wallet: equity={} (starting_capital={} realized_pnl={})",
        state.wallet.equity(), state.wallet.starting_capital, state.wallet.realized_pnl
    ));

    lines.push(format!("recent trades ({})", state.recent_trades.len()));
    for trade in state.recent_trades.iter().rev().take(10) {
        lines.push(format!(
            "  {} {} size={} price={} [{}] {}",
            trade.timestamp,
            trade.side,
            trade.size,
            trade.price,
            if trade.dry_run { "dry_run" } else { "LIVE" },
            trade.note
        ));
    }

    DashboardLines { lines }
}

pub(crate) fn is_quit_key(key: crossterm::event::KeyEvent) -> bool {
    use crossterm::event::{KeyCode, KeyModifiers};
    key.code == KeyCode::Char('q')
        || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
}

/// Raw mode is disabled on drop, so it's restored on every exit path —
/// including an early `?` return — not just the happy path.
pub(crate) struct RawModeGuard;

impl RawModeGuard {
    pub(crate) fn new() -> anyhow::Result<Self> {
        crossterm::terminal::enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = crossterm::terminal::disable_raw_mode();
    }
}

/// Raw mode disables the terminal's SIGINT generation (`ISIG`), so Ctrl+C
/// arrives here as a key event, not a signal — this is what actually quits
/// the bot; `tokio::signal::ctrl_c()` in `main` never fires while the TUI
/// owns the terminal. Detecting a quit key here trips `shutdown_tx` so
/// every other task (and `main`) shuts down too.
pub async fn run_tui(
    state: Arc<RwLock<BotState>>,
    shutdown_tx: broadcast::Sender<()>,
    mut shutdown_rx: broadcast::Receiver<()>,
) -> anyhow::Result<()> {
    let _raw_mode = RawModeGuard::new()?;
    let backend = CrosstermBackend::new(std::io::stdout());
    let mut terminal = Terminal::new(backend)?;

    loop {
        let snapshot = { render_lines(&*state.read().await) };
        terminal.draw(|frame| {
            let items: Vec<ListItem> = snapshot.lines.iter().map(|l| ListItem::new(l.clone())).collect();
            let list = List::new(items);
            frame.render_widget(list, frame.area());
        })?;

        tokio::select! {
            _ = shutdown_rx.recv() => break,
            quit = tokio::task::spawn_blocking(|| {
                if crossterm::event::poll(std::time::Duration::from_millis(200)).unwrap_or(false) {
                    if let Ok(crossterm::event::Event::Key(key)) = crossterm::event::read() {
                        return is_quit_key(key);
                    }
                }
                false
            }) => {
                if quit.unwrap_or(false) {
                    let _ = shutdown_tx.send(());
                    break;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BotState, OrderBookSnapshot, Signal};
    use chrono::Utc;

    #[test]
    fn shows_no_data_placeholder_when_state_is_empty() {
        let state = BotState::new();
        let lines = render_lines(&state);
        assert!(lines.lines.iter().any(|l| l.contains("no snapshot yet")));
    }

    #[test]
    fn shows_best_bid_ask_and_signal_when_present() {
        let mut state = BotState::new();
        state.last_snapshot = Some(OrderBookSnapshot {
            market: "m".to_string(),
            slot: 5,
            bids: vec![(100, 10)],
            asks: vec![(101, 10)],
            timestamp: Utc::now(),
        });
        state.last_signal = Some(Signal { should_trade: true, confidence: 0.9 });

        let lines = render_lines(&state);
        assert!(lines.lines.iter().any(|l| l.contains("100")));
        assert!(lines.lines.iter().any(|l| l.contains("0.9")));
    }

    #[test]
    fn flags_kill_switch_active() {
        let mut state = BotState::new();
        state.kill_switch_active = true;
        let lines = render_lines(&state);
        assert!(lines.lines.iter().any(|l| l.to_uppercase().contains("KILL SWITCH")));
    }

    #[test]
    fn q_key_is_a_quit_key() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let key = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE);
        assert!(is_quit_key(key));
    }

    #[test]
    fn ctrl_c_is_a_quit_key() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let key = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(is_quit_key(key));
    }

    #[test]
    fn other_keys_are_not_quit_keys() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let key = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE);
        assert!(!is_quit_key(key));
    }

    #[test]
    fn shows_flat_when_no_position() {
        let state = BotState::new();
        let lines = render_lines(&state);
        assert!(lines.lines.iter().any(|l| l.to_lowercase().contains("flat")));
    }

    #[test]
    fn shows_position_size_and_entry_cost_when_holding() {
        let mut state = BotState::new();
        state.position = Some(crate::types::Position { size: 5_000_000, entry_cost: 1000 });
        let lines = render_lines(&state);
        assert!(lines.lines.iter().any(|l| l.contains("5000000") || l.contains("5_000_000")));
        assert!(lines.lines.iter().any(|l| l.contains("1000")));
    }

    #[test]
    fn shows_current_equity() {
        let mut state = BotState::new();
        state.wallet = crate::types::WalletState { starting_capital: 1000, realized_pnl: 250 };
        let lines = render_lines(&state);
        assert!(lines.lines.iter().any(|l| l.to_lowercase().contains("equity") && l.contains("1250")));
    }
}
