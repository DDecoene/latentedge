import pytest

from latentedge.uniswap_math import (
    price_to_sqrt_price_x96,
    sqrt_price_x96_to_price,
    sqrt_price_x96_to_weth_usdc_price,
)


def test_round_trip_price_conversion():
    # 1 raw token0 unit "buys" 0.0005 raw token1 units, i.e. a
    # constructed, self-verifying price — not a claim about any real
    # historical price.
    raw_price = 0.0005
    sqrt_price_x96 = price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18)
    recovered = sqrt_price_x96_to_price(sqrt_price_x96, decimals0=6, decimals1=18)
    assert recovered == pytest.approx(raw_price, rel=1e-9)


def test_weth_usdc_price_is_positive_and_plausible():
    # Construct a sqrtPriceX96 for a known, plausible WETH price of
    # $3,000 (in human terms) and confirm the pool-specific helper
    # recovers it, given token0=USDC(6dp)/token1=WETH(18dp).
    #
    # price_to_sqrt_price_x96 already applies the decimal adjustment
    # internally, so the human-terms WETH-per-USDC price (the inverse of
    # USDC-per-WETH) is passed directly, with no additional scaling.
    human_usdc_per_weth = 3000.0
    human_weth_per_usdc = 1.0 / human_usdc_per_weth
    sqrt_price_x96 = price_to_sqrt_price_x96(human_weth_per_usdc, decimals0=6, decimals1=18)
    usdc_per_weth = sqrt_price_x96_to_weth_usdc_price(sqrt_price_x96)
    assert usdc_per_weth == pytest.approx(human_usdc_per_weth, rel=1e-6)
