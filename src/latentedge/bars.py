import pandas as pd

from latentedge import config
from latentedge.uniswap_math import sqrt_price_x96_to_weth_usdc_price


WEI_PER_GWEI = 1_000_000_000


def build_bars(swaps: pd.DataFrame, interval_seconds: int) -> pd.DataFrame:
    if swaps.empty:
        return pd.DataFrame(
            columns=[
                "bar_start", "price_usdc_per_weth", "swap_count", "volume_usdc", "net_flow_usdc", "max_swap_usdc",
                "base_fee_gwei", "has_gap",
            ]
        )

    df = swaps.copy()
    df["price_usdc_per_weth"] = df["sqrt_price_x96"].apply(sqrt_price_x96_to_weth_usdc_price)
    # amount0 is decoded from the raw Swap event in token0's (USDC's) raw
    # 6-decimal units, not human dollars — confirmed against live RPC data.
    # A bare abs(amount0) would be 1,000,000x too large.
    df["volume_usdc"] = df["amount0"].abs() / (10**config.TOKEN0_DECIMALS)
    # Positive amount0 is USDC paid into the pool, i.e. someone buying WETH,
    # so the signed sum is net buying pressure in dollars (negative = selling).
    df["signed_volume_usdc"] = df["amount0"] / (10**config.TOKEN0_DECIMALS)
    df["base_fee_gwei"] = df["base_fee_wei"] / WEI_PER_GWEI
    df["bar_start"] = (df["timestamp"] // interval_seconds) * interval_seconds

    first_bar = int(df["bar_start"].min())
    last_bar = int(df["bar_start"].max())
    all_bar_starts = range(first_bar, last_bar + interval_seconds, interval_seconds)

    grouped = df.groupby("bar_start").agg(
        price_usdc_per_weth=("price_usdc_per_weth", "last"),
        swap_count=("price_usdc_per_weth", "count"),
        volume_usdc=("volume_usdc", "sum"),
        net_flow_usdc=("signed_volume_usdc", "sum"),
        max_swap_usdc=("volume_usdc", "max"),
        base_fee_gwei=("base_fee_gwei", "last"),
    )

    bars = grouped.reindex(all_bar_starts)
    bars.index.name = "bar_start"
    bars["has_gap"] = bars["swap_count"].isna()
    bars["swap_count"] = bars["swap_count"].fillna(0).astype(int)
    bars["volume_usdc"] = bars["volume_usdc"].fillna(0.0)
    bars["net_flow_usdc"] = bars["net_flow_usdc"].fillna(0.0)
    bars["max_swap_usdc"] = bars["max_swap_usdc"].fillna(0.0)
    bars["price_usdc_per_weth"] = bars["price_usdc_per_weth"].ffill()
    # base_fee is chain-wide state, not pool-specific — it exists whether
    # or not this pool traded in a given bar, so a gap should carry the
    # last known fee forward, the same as price, rather than reading 0.
    bars["base_fee_gwei"] = bars["base_fee_gwei"].ffill()

    return bars.reset_index()
