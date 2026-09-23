use crate::types::OrderBookSnapshot;
use chrono::Utc;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct LadderJson {
    pub bids: Vec<(u64, u64)>,
    pub asks: Vec<(u64, u64)>,
}

pub fn validate_ladder(ladder: &LadderJson) -> anyhow::Result<()> {
    if let (Some(&(best_bid, _)), Some(&(best_ask, _))) =
        (ladder.bids.first(), ladder.asks.first())
    {
        if best_bid >= best_ask {
            anyhow::bail!("crossed book: best bid {best_bid} >= best ask {best_ask}");
        }
    }
    Ok(())
}

pub fn ladder_to_snapshot(market: &str, slot: u64, ladder: &LadderJson) -> OrderBookSnapshot {
    OrderBookSnapshot {
        market: market.to_string(),
        slot,
        bids: ladder.bids.clone(),
        asks: ladder.asks.clone(),
        timestamp: Utc::now(),
    }
}

pub fn decode_from_ladder_json(
    market: &str,
    slot: u64,
    raw_json: &str,
) -> anyhow::Result<OrderBookSnapshot> {
    let ladder: LadderJson = serde_json::from_str(raw_json)?;
    validate_ladder(&ladder)?;
    Ok(ladder_to_snapshot(market, slot, &ladder))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_valid_ladder_json_into_snapshot() {
        let raw = std::fs::read_to_string("tests/fixtures/phoenix_snapshot_valid.json")
            .expect("fixture must exist");
        let ladder: LadderJson = serde_json::from_str(&raw).unwrap();
        let snapshot = ladder_to_snapshot("marketABC", 42, &ladder);

        assert_eq!(snapshot.market, "marketABC");
        assert_eq!(snapshot.slot, 42);
        assert_eq!(snapshot.bids, vec![(10050, 200), (10040, 500)]);
        assert_eq!(snapshot.asks, vec![(10060, 150), (10070, 400)]);
    }

    #[test]
    fn rejects_crossed_book() {
        let ladder = LadderJson {
            bids: vec![(10100, 10)],
            asks: vec![(10050, 10)], // best ask below best bid: invalid
        };
        let err = validate_ladder(&ladder).unwrap_err();
        assert!(err.to_string().contains("crossed"));
    }
}
