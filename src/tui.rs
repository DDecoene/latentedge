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

pub async fn run_tui(
    state: Arc<RwLock<BotState>>,
    mut shutdown_rx: broadcast::Receiver<()>,
) -> anyhow::Result<()> {
    crossterm::terminal::enable_raw_mode()?;
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
            _ = tokio::time::sleep(tokio::time::Duration::from_millis(250)) => {}
            _ = tokio::task::spawn_blocking(|| {
                if crossterm::event::poll(std::time::Duration::from_millis(0)).unwrap_or(false) {
                    if let Ok(crossterm::event::Event::Key(key)) = crossterm::event::read() {
                        return key.code == crossterm::event::KeyCode::Char('q');
                    }
                }
                false
            }) => {}
        }
    }

    crossterm::terminal::disable_raw_mode()?;
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
}
