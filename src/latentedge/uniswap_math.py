"""Uniswap v3 sqrtPriceX96 <-> human price conversions.

Uniswap v3 stores price as sqrt(price) * 2**96, where "price" is the
amount of token1 you get per unit of token0, in raw (undecimaled) token
units. Converting to a human-readable price requires adjusting for each
token's decimals.
"""

Q96 = 2**96


def sqrt_price_x96_to_price(sqrt_price_x96: int, decimals0: int, decimals1: int) -> float:
    """Price of token0 in terms of token1, decimal-adjusted."""
    raw_price = (sqrt_price_x96 / Q96) ** 2
    return raw_price * (10 ** (decimals0 - decimals1))


def price_to_sqrt_price_x96(price_token1_per_token0: float, decimals0: int, decimals1: int) -> int:
    """Inverse of sqrt_price_x96_to_price. Test-fixture helper."""
    raw_price = price_token1_per_token0 / (10 ** (decimals0 - decimals1))
    return int((raw_price**0.5) * Q96)


def sqrt_price_x96_to_weth_usdc_price(sqrt_price_x96: int) -> float:
    """USDC price per 1 WETH for the fixed WETH/USDC pool (token0=USDC,
    token1=WETH). sqrt_price_x96_to_price gives WETH-per-USDC; invert."""
    weth_per_usdc = sqrt_price_x96_to_price(sqrt_price_x96, decimals0=6, decimals1=18)
    return 1.0 / weth_per_usdc
