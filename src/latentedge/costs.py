"""Slippage and gas-cost estimation.

Slippage: a concentrated-liquidity approximation, not exact tick-crossing
math (spec non-goal). Local liquidity L near the current price implies a
virtual USDC-denominated depth of roughly L * sqrt(price) (a standard
Uniswap v3 approximation for in-range depth in quote-token terms). Trade
impact is modeled the way constant-product slippage is commonly
approximated: impact_fraction ~= notional / (2 * virtual_depth_usd).
"""

from latentedge.uniswap_math import Q96, sqrt_price_x96_to_weth_usdc_price

WEI_PER_ETH = 10**18


def _virtual_depth_usd(liquidity: int, sqrt_price_x96: int) -> float:
    sqrt_price = sqrt_price_x96 / Q96
    # L * sqrtP, scaled from raw units to USDC (6 decimals) terms.
    virtual_reserve_token1_raw = liquidity * sqrt_price
    weth_price = sqrt_price_x96_to_weth_usdc_price(sqrt_price_x96)
    virtual_reserve_weth = virtual_reserve_token1_raw / (10**18)
    return virtual_reserve_weth * weth_price


def estimate_slippage_fraction(notional_usd: float, liquidity: int, sqrt_price_x96: int) -> float:
    depth_usd = _virtual_depth_usd(liquidity, sqrt_price_x96)
    if depth_usd <= 0:
        raise ValueError("non-positive virtual depth; check liquidity/sqrt_price_x96 inputs")
    return notional_usd / (2 * depth_usd)


def estimate_gas_cost_usd(base_fee_wei: int, gas_used: int, weth_usdc_price: float) -> float:
    cost_wei = base_fee_wei * gas_used
    cost_eth = cost_wei / WEI_PER_ETH
    return cost_eth * weth_usdc_price
