from latentedge import config


def test_pool_and_token_constants():
    assert config.TOKEN0_SYMBOL == "USDC"
    assert config.TOKEN0_DECIMALS == 6
    assert config.TOKEN1_SYMBOL == "WETH"
    assert config.TOKEN1_DECIMALS == 18
    assert config.FEE_TIER_BPS == 5
    assert config.BAR_INTERVAL_SECONDS == 60
    assert config.LABEL_HORIZON_SECONDS == 1800
    assert config.REFERENCE_NOTIONAL_USD == 1000.0
    assert config.GAS_USED_PER_SWAP == 150_000
