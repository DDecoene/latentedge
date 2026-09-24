use chrono::{DateTime, Utc};
use serde::Deserialize;

#[derive(Deserialize)]
struct MarketChartResponse {
    prices: Vec<(i64, f64)>,
}

pub async fn fetch_or_load_cached(
    coingecko_base_url: &str,
    api_key: Option<&str>,
    days: u32,
    cache_path: &std::path::Path,
) -> anyhow::Result<Vec<(DateTime<Utc>, f64)>> {
    if let Ok(cached) = std::fs::read_to_string(cache_path) {
        if let Ok(series) = serde_json::from_str::<Vec<(DateTime<Utc>, f64)>>(&cached) {
            return Ok(series);
        }
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .user_agent("layatrade-rs-backtest/0.1")
        .build()?;
    let url = format!("{coingecko_base_url}/coins/solana/market_chart?vs_currency=usd&days={days}");
    let mut request = client.get(url);
    if let Some(key) = api_key {
        request = request.header("x-cg-demo-api-key", key);
    }
    let response: MarketChartResponse = request.send().await?.error_for_status()?.json().await?;

    let series: Vec<(DateTime<Utc>, f64)> = response
        .prices
        .into_iter()
        .filter_map(|(ms, price)| DateTime::from_timestamp_millis(ms).map(|ts| (ts, price)))
        .collect();

    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(cache_path, serde_json::to_string(&series)?)?;

    Ok(series)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn fetches_and_parses_prices_on_a_cold_cache() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/coins/solana/market_chart"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "prices": [[1700000000000i64, 150.5], [1700003600000i64, 151.2]],
                "market_caps": [],
                "total_volumes": []
            })))
            .mount(&server)
            .await;

        let cache_path = std::env::temp_dir().join(format!("layatrade_test_cache_{}.json", std::process::id()));
        std::fs::remove_file(&cache_path).ok();

        let series = fetch_or_load_cached(&server.uri(), None, 90, &cache_path)
            .await
            .expect("should fetch and parse");

        assert_eq!(series.len(), 2);
        assert_eq!(series[0].1, 150.5);
        assert_eq!(series[1].1, 151.2);
        std::fs::remove_file(&cache_path).ok();
    }

    #[tokio::test]
    async fn a_warm_cache_never_makes_a_second_http_request() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/coins/solana/market_chart"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "prices": [[1700000000000i64, 150.5]],
                "market_caps": [],
                "total_volumes": []
            })))
            .expect(1) // exactly once — the second call below must hit the cache, not the network
            .mount(&server)
            .await;

        let cache_path = std::env::temp_dir().join(format!("layatrade_test_cache_warm_{}.json", std::process::id()));
        std::fs::remove_file(&cache_path).ok();

        let first = fetch_or_load_cached(&server.uri(), None, 90, &cache_path).await.expect("first fetch");
        let second = fetch_or_load_cached(&server.uri(), None, 90, &cache_path).await.expect("second, cached");

        assert_eq!(first.len(), second.len());
        server.verify().await; // enforces the expect(1) above
        std::fs::remove_file(&cache_path).ok();
    }

    #[tokio::test]
    async fn sends_the_api_key_header_when_configured() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/coins/solana/market_chart"))
            .and(wiremock::matchers::header("x-cg-demo-api-key", "my-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "prices": [[1700000000000i64, 150.5]],
                "market_caps": [],
                "total_volumes": []
            })))
            .mount(&server)
            .await;

        let cache_path = std::env::temp_dir().join(format!("layatrade_test_cache_key_{}.json", std::process::id()));
        std::fs::remove_file(&cache_path).ok();

        let result = fetch_or_load_cached(&server.uri(), Some("my-key"), 90, &cache_path).await;
        assert!(result.is_ok(), "request must have sent the header to match the mock: {result:?}");
        std::fs::remove_file(&cache_path).ok();
    }

    #[tokio::test]
    async fn sends_a_user_agent_header() {
        // reqwest sends no User-Agent by default; CoinGecko's real API
        // (confirmed against the live endpoint) returns 403 without one.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/coins/solana/market_chart"))
            .and(wiremock::matchers::header_exists("User-Agent"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "prices": [[1700000000000i64, 150.5]],
                "market_caps": [],
                "total_volumes": []
            })))
            .mount(&server)
            .await;

        let cache_path = std::env::temp_dir().join(format!("layatrade_test_cache_ua_{}.json", std::process::id()));
        std::fs::remove_file(&cache_path).ok();

        let result = fetch_or_load_cached(&server.uri(), None, 90, &cache_path).await;
        assert!(result.is_ok(), "request must send a User-Agent header to match the mock: {result:?}");
        std::fs::remove_file(&cache_path).ok();
    }
}
