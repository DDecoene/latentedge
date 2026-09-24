pub fn synthetic_ladder_json(spot_price: f64, cost_bps: u32) -> String {
    let price_atoms = (spot_price * 1_000_000.0).round().max(0.0) as u64;
    let cost_atoms = ((price_atoms as f64) * (cost_bps as f64) / 10_000.0).round().max(1.0) as u64;
    let bid = price_atoms.saturating_sub(cost_atoms);
    let ask = price_atoms.saturating_add(cost_atoms);
    serde_json::json!({
        "bids": [[bid, 1]],
        "asks": [[ask, 1]],
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phoenix_decode::decode_from_ladder_json;

    #[test]
    fn produces_a_valid_uncrossed_ladder_at_a_normal_price() {
        let raw = synthetic_ladder_json(150.0, 15);
        let snapshot = decode_from_ladder_json("backtest", 0, &raw).expect("must be valid");
        assert!(snapshot.bids[0].0 < snapshot.asks[0].0);
    }

    #[test]
    fn stays_uncrossed_at_extreme_cost_bps() {
        let raw = synthetic_ladder_json(150.0, 50_000); // 500%, absurd but must not crash or cross
        let snapshot = decode_from_ladder_json("backtest", 0, &raw).expect("must still be valid");
        assert!(snapshot.bids[0].0 < snapshot.asks[0].0);
    }

    #[test]
    fn stays_uncrossed_at_a_zero_spot_price() {
        let raw = synthetic_ladder_json(0.0, 15);
        let snapshot = decode_from_ladder_json("backtest", 0, &raw).expect("must still be valid");
        assert!(snapshot.bids[0].0 < snapshot.asks[0].0);
    }
}
