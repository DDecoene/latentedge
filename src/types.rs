use chrono::{DateTime, Utc};
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub market: String,
    pub slot: u64,
    pub bids: Vec<(u64, u64)>,
    pub asks: Vec<(u64, u64)>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy)]
pub struct Signal {
    pub should_trade: bool,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct TradeEvent {
    pub timestamp: DateTime<Utc>,
    pub side: String,
    pub size: u64,
    pub price: u64,
    pub dry_run: bool,
    pub note: String,
}

#[derive(Debug, Clone)]
pub struct StreamHealth {
    pub last_update: DateTime<Utc>,
    pub reconnect_count: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub size: u64,
    pub entry_cost: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct WalletState {
    pub starting_capital: u64,
    pub realized_pnl: i64,
}

impl WalletState {
    pub fn equity(&self) -> u64 {
        (self.starting_capital as i64 + self.realized_pnl).max(0) as u64
    }
}

impl Default for WalletState {
    fn default() -> Self {
        Self { starting_capital: 0, realized_pnl: 0 }
    }
}

const MAX_RECENT_TRADES: usize = 20;

#[derive(Debug, Default)]
pub struct BotState {
    pub last_snapshot: Option<OrderBookSnapshot>,
    pub last_signal: Option<Signal>,
    pub stream_health: Option<StreamHealth>,
    pub recent_trades: VecDeque<TradeEvent>,
    pub kill_switch_active: bool,
    pub position: Option<Position>,
    pub wallet: WalletState,
}

impl BotState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_trade(&mut self, event: TradeEvent) {
        self.recent_trades.push_back(event);
        while self.recent_trades.len() > MAX_RECENT_TRADES {
            self.recent_trades.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bot_state_keeps_last_20_trades_only() {
        let mut state = BotState::new();
        for i in 0..25 {
            state.push_trade(TradeEvent {
                timestamp: chrono::Utc::now(),
                side: "buy".to_string(),
                size: i,
                price: 100,
                dry_run: true,
                note: format!("trade {i}"),
            });
        }
        assert_eq!(state.recent_trades.len(), 20);
        // oldest trades dropped, newest kept
        assert_eq!(state.recent_trades.back().unwrap().size, 24);
    }

    #[test]
    fn bot_state_position_defaults_to_none() {
        let state = BotState::new();
        assert!(state.position.is_none());
    }

    #[test]
    fn wallet_equity_reflects_realized_pnl() {
        let wallet = WalletState { starting_capital: 1000, realized_pnl: 250 };
        assert_eq!(wallet.equity(), 1250);

        let losing_wallet = WalletState { starting_capital: 1000, realized_pnl: -400 };
        assert_eq!(losing_wallet.equity(), 600);
    }

    #[test]
    fn wallet_equity_never_goes_negative() {
        let wallet = WalletState { starting_capital: 1000, realized_pnl: -5000 };
        assert_eq!(wallet.equity(), 0);
    }
}
