use layatrade_rs::phoenix_decode::decode_from_ladder_json;
use layatrade_rs::types::BotState;

#[test]
fn fixture_snapshot_decodes_and_state_updates() {
    let raw = std::fs::read_to_string("tests/fixtures/phoenix_snapshot_valid.json").unwrap();
    let snapshot = decode_from_ladder_json("marketX", 1, &raw).unwrap();

    let mut state = BotState::new();
    state.last_snapshot = Some(snapshot.clone());

    assert_eq!(state.last_snapshot.unwrap().market, "marketX");
}
