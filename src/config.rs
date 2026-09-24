use std::collections::HashMap;
use std::env;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionMode {
    DryRun,
    Live,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub solana_rpc_url: String,
    pub laya_server_url: String,
    pub laya_confidence_threshold: f64,
    pub execution_mode: ExecutionMode,
    pub solana_keypair_path: Option<String>,
    pub max_trade_size: u64,
    pub max_trades_per_window: u32,
    pub max_slippage_bps: u16,
    pub max_daily_loss: u64,
    pub kill_switch_path: String,
    pub jupiter_base_url: String,
    pub base_mint: String,
    pub quote_mint: String,
    pub poll_interval_ms: u64,
    pub trade_size_pct: f64,
    pub starting_capital: u64,
    pub coingecko_base_url: String,
    pub coingecko_api_key: Option<String>,
    pub backtest_days: u32,
    pub backtest_cost_bps: u32,
    pub backtest_tick_ms: u64,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let vars: HashMap<String, String> = env::vars().collect();
        Self::from_map(&vars)
    }

    pub fn from_map(vars: &HashMap<String, String>) -> anyhow::Result<Self> {
        let get_or = |key: &str, default: &str| -> String {
            vars.get(key).cloned().unwrap_or_else(|| default.to_string())
        };

        let execution_mode = match get_or("EXECUTION_MODE", "dry_run").as_str() {
            "dry_run" => ExecutionMode::DryRun,
            "live" => ExecutionMode::Live,
            other => anyhow::bail!("EXECUTION_MODE must be dry_run or live, got {other}"),
        };

        Ok(Config {
            solana_rpc_url: get_or("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
            laya_server_url: get_or("LAYA_SERVER_URL", "http://127.0.0.1:8787"),
            laya_confidence_threshold: get_or("LAYA_CONFIDENCE_THRESHOLD", "0.85").parse()?,
            execution_mode,
            solana_keypair_path: vars.get("SOLANA_KEYPAIR_PATH").cloned().filter(|s| !s.is_empty()),
            max_trade_size: get_or("MAX_TRADE_SIZE", "500000000").parse()?,
            max_trades_per_window: get_or("MAX_TRADES_PER_WINDOW", "5").parse()?,
            max_slippage_bps: get_or("MAX_SLIPPAGE_BPS", "50").parse()?,
            max_daily_loss: get_or("MAX_DAILY_LOSS", "500000000").parse()?,
            kill_switch_path: get_or("KILL_SWITCH_PATH", "./KILL_SWITCH"),
            jupiter_base_url: get_or("JUPITER_BASE_URL", "https://quote-api.jup.ag/v6"),
            base_mint: get_or("BASE_MINT", "So11111111111111111111111111111111111111112"),
            quote_mint: get_or("QUOTE_MINT", "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"),
            poll_interval_ms: get_or("POLL_INTERVAL_MS", "5000").parse()?,
            trade_size_pct: {
                let pct: f64 = get_or("TRADE_SIZE_PCT", "0.1").parse()?;
                if !(pct > 0.0 && pct <= 1.0) {
                    anyhow::bail!("TRADE_SIZE_PCT must be in (0, 1], got {pct}");
                }
                pct
            },
            starting_capital: get_or("STARTING_CAPITAL", "1000000000").parse()?,
            coingecko_base_url: get_or("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3"),
            coingecko_api_key: vars.get("COINGECKO_API_KEY").cloned().filter(|s| !s.is_empty()),
            backtest_days: get_or("BACKTEST_DAYS", "90").parse()?,
            backtest_cost_bps: get_or("BACKTEST_COST_BPS", "15").parse()?,
            backtest_tick_ms: get_or("BACKTEST_TICK_MS", "50").parse()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_env_map() {
        let mut vars = std::collections::HashMap::new();
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
    fn jupiter_streaming_config_has_sensible_defaults() {
        let vars = std::collections::HashMap::new();
        let config = Config::from_map(&vars).expect("should parse with all defaults");
        assert_eq!(config.jupiter_base_url, "https://quote-api.jup.ag/v6");
        assert_eq!(config.base_mint, "So11111111111111111111111111111111111111112");
        assert_eq!(config.quote_mint, "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v");
        assert_eq!(config.poll_interval_ms, 5000);
    }

    #[test]
    fn jupiter_streaming_config_is_overridable() {
        let mut vars = std::collections::HashMap::new();
        vars.insert("JUPITER_BASE_URL".to_string(), "http://127.0.0.1:9999".to_string());
        vars.insert("BASE_MINT".to_string(), "mintA".to_string());
        vars.insert("QUOTE_MINT".to_string(), "mintB".to_string());
        vars.insert("POLL_INTERVAL_MS".to_string(), "1000".to_string());
        let config = Config::from_map(&vars).expect("should parse overrides");
        assert_eq!(config.jupiter_base_url, "http://127.0.0.1:9999");
        assert_eq!(config.base_mint, "mintA");
        assert_eq!(config.quote_mint, "mintB");
        assert_eq!(config.poll_interval_ms, 1000);
    }

    #[test]
    fn money_management_config_has_sensible_defaults() {
        let vars = std::collections::HashMap::new();
        let config = Config::from_map(&vars).expect("should parse with all defaults");
        assert_eq!(config.trade_size_pct, 0.1);
        assert_eq!(config.starting_capital, 1_000_000_000);
    }

    #[test]
    fn money_management_config_is_overridable() {
        let mut vars = std::collections::HashMap::new();
        vars.insert("TRADE_SIZE_PCT".to_string(), "0.25".to_string());
        vars.insert("STARTING_CAPITAL".to_string(), "500".to_string());
        let config = Config::from_map(&vars).expect("should parse overrides");
        assert_eq!(config.trade_size_pct, 0.25);
        assert_eq!(config.starting_capital, 500);
    }

    #[test]
    fn rejects_trade_size_pct_outside_zero_to_one() {
        let mut too_high = std::collections::HashMap::new();
        too_high.insert("TRADE_SIZE_PCT".to_string(), "10".to_string());
        let err = Config::from_map(&too_high).unwrap_err();
        assert!(err.to_string().contains("TRADE_SIZE_PCT"));

        let mut negative = std::collections::HashMap::new();
        negative.insert("TRADE_SIZE_PCT".to_string(), "-0.1".to_string());
        assert!(Config::from_map(&negative).is_err());

        let mut zero = std::collections::HashMap::new();
        zero.insert("TRADE_SIZE_PCT".to_string(), "0".to_string());
        assert!(Config::from_map(&zero).is_err());
    }

    #[test]
    fn backtest_config_has_sensible_defaults() {
        let vars = std::collections::HashMap::new();
        let config = Config::from_map(&vars).expect("should parse with all defaults");
        assert_eq!(config.coingecko_base_url, "https://api.coingecko.com/api/v3");
        assert_eq!(config.coingecko_api_key, None);
        assert_eq!(config.backtest_days, 90);
        assert_eq!(config.backtest_cost_bps, 15);
        assert_eq!(config.backtest_tick_ms, 50);
    }

    #[test]
    fn backtest_config_is_overridable() {
        let mut vars = std::collections::HashMap::new();
        vars.insert("COINGECKO_BASE_URL".to_string(), "http://127.0.0.1:9999".to_string());
        vars.insert("COINGECKO_API_KEY".to_string(), "demo-key".to_string());
        vars.insert("BACKTEST_DAYS".to_string(), "30".to_string());
        vars.insert("BACKTEST_COST_BPS".to_string(), "25".to_string());
        vars.insert("BACKTEST_TICK_MS".to_string(), "10".to_string());
        let config = Config::from_map(&vars).expect("should parse overrides");
        assert_eq!(config.coingecko_base_url, "http://127.0.0.1:9999");
        assert_eq!(config.coingecko_api_key, Some("demo-key".to_string()));
        assert_eq!(config.backtest_days, 30);
        assert_eq!(config.backtest_cost_bps, 25);
        assert_eq!(config.backtest_tick_ms, 10);
    }

    #[test]
    fn max_trade_size_and_max_daily_loss_defaults_allow_a_default_buy() {
        let vars = std::collections::HashMap::new();
        let config = Config::from_map(&vars).expect("should parse with all defaults");
        let default_buy_spend = (config.starting_capital as f64 * config.trade_size_pct) as u64;
        assert!(
            config.max_trade_size > default_buy_spend,
            "MAX_TRADE_SIZE default ({}) must exceed a default buy's spend ({})",
            config.max_trade_size, default_buy_spend
        );
        assert_eq!(config.max_trade_size, 500_000_000);
        assert_eq!(config.max_daily_loss, 500_000_000);
    }

    #[test]
    fn rejects_invalid_execution_mode() {
        let mut vars = std::collections::HashMap::new();
        vars.insert("EXECUTION_MODE".to_string(), "yolo".to_string());
        let err = Config::from_map(&vars).unwrap_err();
        assert!(err.to_string().contains("EXECUTION_MODE"));
    }

    #[test]
    fn live_mode_is_accepted_now_that_submission_exists() {
        let mut vars = std::collections::HashMap::new();
        vars.insert("EXECUTION_MODE".to_string(), "live".to_string());
        let config = Config::from_map(&vars).expect("live mode should now parse");
        assert_eq!(config.execution_mode, ExecutionMode::Live);
    }
}
