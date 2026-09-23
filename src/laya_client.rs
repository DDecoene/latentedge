use crate::types::{OrderBookSnapshot, Signal};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Serialize)]
struct PredictRequest {
    state: String,
}

#[derive(Deserialize)]
struct PredictResponse {
    confidence: f64,
}

pub struct LayaClient {
    base_url: String,
    threshold: f64,
    http: reqwest::Client,
}

impl LayaClient {
    pub fn new(base_url: String, threshold: f64) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(200))
            .build()
            .expect("client build should not fail");
        Self { base_url, threshold, http }
    }

    pub async fn health_check(&self) -> anyhow::Result<()> {
        let resp = self.http.get(format!("{}/health", self.base_url)).send().await?;
        if !resp.status().is_success() {
            anyhow::bail!("laya health check returned {}", resp.status());
        }
        Ok(())
    }

    fn describe_state(snapshot: &OrderBookSnapshot) -> String {
        let best_bid = snapshot.bids.first().copied().unwrap_or((0, 0));
        let best_ask = snapshot.asks.first().copied().unwrap_or((0, 0));
        format!(
            "market={} slot={} best_bid={:?} best_ask={:?} bid_depth={} ask_depth={}",
            snapshot.market,
            snapshot.slot,
            best_bid,
            best_ask,
            snapshot.bids.len(),
            snapshot.asks.len(),
        )
    }

    pub async fn get_signal(&self, snapshot: &OrderBookSnapshot) -> Signal {
        let request = PredictRequest { state: Self::describe_state(snapshot) };
        let result = self
            .http
            .post(format!("{}/predict", self.base_url))
            .json(&request)
            .send()
            .await;

        match result {
            Ok(resp) if resp.status().is_success() => match resp.json::<PredictResponse>().await {
                Ok(parsed) => Signal {
                    should_trade: parsed.confidence >= self.threshold,
                    confidence: parsed.confidence,
                },
                Err(_) => Signal { should_trade: false, confidence: 0.0 },
            },
            _ => Signal { should_trade: false, confidence: 0.0 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::OrderBookSnapshot;
    use chrono::Utc;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sample_snapshot() -> OrderBookSnapshot {
        OrderBookSnapshot {
            market: "m".to_string(),
            slot: 1,
            bids: vec![(100, 10)],
            asks: vec![(101, 10)],
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn returns_should_trade_true_above_threshold() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/predict"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"confidence": 0.92})))
            .mount(&server)
            .await;

        let client = LayaClient::new(server.uri(), 0.85);
        let signal = client.get_signal(&sample_snapshot()).await;

        assert!(signal.should_trade);
        assert_eq!(signal.confidence, 0.92);
    }

    #[tokio::test]
    async fn returns_should_trade_false_below_threshold() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/predict"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"confidence": 0.40})))
            .mount(&server)
            .await;

        let client = LayaClient::new(server.uri(), 0.85);
        let signal = client.get_signal(&sample_snapshot()).await;

        assert!(!signal.should_trade);
    }

    #[tokio::test]
    async fn fails_safe_to_no_trade_on_server_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/predict"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let client = LayaClient::new(server.uri(), 0.85);
        let signal = client.get_signal(&sample_snapshot()).await;

        assert!(!signal.should_trade);
        assert_eq!(signal.confidence, 0.0);
    }

    #[tokio::test]
    async fn health_check_fails_when_endpoint_down() {
        let client = LayaClient::new("http://127.0.0.1:1".to_string(), 0.85);
        assert!(client.health_check().await.is_err());
    }
}
