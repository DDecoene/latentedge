import pytest

from latentedge.costs import estimate_gas_cost_usd, estimate_slippage_fraction
from latentedge.uniswap_math import price_to_sqrt_price_x96


def _sqrt_price_for(human_usdc_per_weth: float) -> int:
    # See Task 2's ruling: price_to_sqrt_price_x96 already decimal-adjusts
    # internally, so pass 1.0 / human_price directly, no extra scaling.
    raw_price = 1.0 / human_usdc_per_weth
    return price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18)


def test_slippage_decreases_with_more_liquidity():
    sqrt_price = _sqrt_price_for(3000.0)
    thin = estimate_slippage_fraction(notional_usd=1000.0, liquidity=10**12, sqrt_price_x96=sqrt_price)
    deep = estimate_slippage_fraction(notional_usd=1000.0, liquidity=10**18, sqrt_price_x96=sqrt_price)
    assert deep < thin
    assert thin > 0
    assert deep > 0


def test_slippage_increases_with_trade_size():
    sqrt_price = _sqrt_price_for(3000.0)
    small = estimate_slippage_fraction(notional_usd=100.0, liquidity=10**15, sqrt_price_x96=sqrt_price)
    large = estimate_slippage_fraction(notional_usd=10_000.0, liquidity=10**15, sqrt_price_x96=sqrt_price)
    assert large > small


def test_gas_cost_scales_with_base_fee():
    cheap = estimate_gas_cost_usd(base_fee_wei=10_000_000_000, gas_used=150_000, weth_usdc_price=3000.0)
    expensive = estimate_gas_cost_usd(base_fee_wei=100_000_000_000, gas_used=150_000, weth_usdc_price=3000.0)
    assert expensive == pytest.approx(cheap * 10, rel=1e-6)
    assert cheap > 0
