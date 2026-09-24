import pandas as pd

from latentedge.uniswap_math import sqrt_price_x96_to_weth_usdc_price


def build_bars(swaps: pd.DataFrame, interval_seconds: int) -> pd.DataFrame:
    if swaps.empty:
        return pd.DataFrame(columns=["bar_start", "price_usdc_per_weth", "swap_count", "volume_usdc", "has_gap"])

    df = swaps.copy()
    df["price_usdc_per_weth"] = df["sqrt_price_x96"].apply(sqrt_price_x96_to_weth_usdc_price)
    df["volume_usdc"] = df["amount0"].abs()
    df["bar_start"] = (df["timestamp"] // interval_seconds) * interval_seconds

    first_bar = int(df["bar_start"].min())
    last_bar = int(df["bar_start"].max())
    all_bar_starts = range(first_bar, last_bar + interval_seconds, interval_seconds)

    grouped = df.groupby("bar_start").agg(
        price_usdc_per_weth=("price_usdc_per_weth", "last"),
        swap_count=("price_usdc_per_weth", "count"),
        volume_usdc=("volume_usdc", "sum"),
    )

    bars = grouped.reindex(all_bar_starts)
    bars.index.name = "bar_start"
    bars["has_gap"] = bars["swap_count"].isna()
    bars["swap_count"] = bars["swap_count"].fillna(0).astype(int)
    bars["volume_usdc"] = bars["volume_usdc"].fillna(0.0)
    bars["price_usdc_per_weth"] = bars["price_usdc_per_weth"].ffill()

    return bars.reset_index()
