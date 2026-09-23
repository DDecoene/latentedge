use std::collections::HashMap;
use std::env;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionMode {
    DryRun,
    Live,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub solana_ws_url: String,
    pub solana_rpc_url: String,
    pub phoenix_market_address: String,
    pub laya_server_url: String,
    pub laya_confidence_threshold: f64,
    pub execution_mode: ExecutionMode,
    pub solana_keypair_path: Option<String>,
    pub max_trade_size: u64,
    pub max_trades_per_window: u32,
    pub max_slippage_bps: u16,
    pub max_daily_loss: u64,
    pub kill_switch_path: String,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let vars: HashMap<String, String> = env::vars().collect();
        Self::from_map(&vars)
    }

    pub fn from_map(vars: &HashMap<String, String>) -> anyhow::Result<Self> {
        let get = |key: &str| -> anyhow::Result<String> {
            vars.get(key)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing required env var {key}"))
        };
        let get_or = |key: &str, default: &str| -> String {
            vars.get(key).cloned().unwrap_or_else(|| default.to_string())
        };

        let execution_mode = match get_or("EXECUTION_MODE", "dry_run").as_str() {
            "dry_run" => ExecutionMode::DryRun,
            "live" => ExecutionMode::Live,
            other => anyhow::bail!("EXECUTION_MODE must be dry_run or live, got {other}"),
        };

        Ok(Config {
            solana_ws_url: get("SOLANA_WS_URL")?,
            solana_rpc_url: get("SOLANA_RPC_URL")?,
            phoenix_market_address: get("PHOENIX_MARKET_ADDRESS")?,
            laya_server_url: get_or("LAYA_SERVER_URL", "http://127.0.0.1:8787"),
            laya_confidence_threshold: get_or("LAYA_CONFIDENCE_THRESHOLD", "0.85").parse()?,
            execution_mode,
            solana_keypair_path: vars.get("SOLANA_KEYPAIR_PATH").cloned().filter(|s| !s.is_empty()),
            max_trade_size: get_or("MAX_TRADE_SIZE", "1000000").parse()?,
            max_trades_per_window: get_or("MAX_TRADES_PER_WINDOW", "5").parse()?,
            max_slippage_bps: get_or("MAX_SLIPPAGE_BPS", "50").parse()?,
            max_daily_loss: get_or("MAX_DAILY_LOSS", "5000000").parse()?,
            kill_switch_path: get_or("KILL_SWITCH_PATH", "./KILL_SWITCH"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_env_map() {
        let mut vars = std::collections::HashMap::new();
        vars.insert("SOLANA_WS_URL".to_string(), "wss://x".to_string());
        vars.insert("SOLANA_RPC_URL".to_string(), "https://x".to_string());
        vars.insert("PHOENIX_MARKET_ADDRESS".to_string(), "abc".to_string());
        vars.insert("LAYA_SERVER_URL".to_string(), "http://127.0.0.1:8787".to_string());
        vars.insert("LAYA_CONFIDENCE_THRESHOLD".to_string(), "0.85".to_string());
        vars.insert("EXECUTION_MODE".to_string(), "dry_run".to_string());
        vars.insert("MAX_TRADE_SIZE".to_string(), "1000000".to_string());
        vars.insert("MAX_TRADES_PER_WINDOW".to_string(), "5".to_string());
        vars.insert("MAX_SLIPPAGE_BPS".to_string(), "50".to_string());
        vars.insert("MAX_DAILY_LOSS".to_string(), "5000000".to_string());
        vars.insert("KILL_SWITCH_PATH".to_string(), "./KILL_SWITCH".to_string());

        let config = Config::from_map(&vars).expect("should parse");
        assert_eq!(config.execution_mode, ExecutionMode::DryRun);
        assert_eq!(config.laya_confidence_threshold, 0.85);
        assert_eq!(config.solana_keypair_path, None);
    }

    #[test]
    fn rejects_invalid_execution_mode() {
        let mut vars = std::collections::HashMap::new();
        vars.insert("EXECUTION_MODE".to_string(), "yolo".to_string());
        let err = Config::from_map(&vars).unwrap_err();
        assert!(err.to_string().contains("EXECUTION_MODE"));
    }
}
