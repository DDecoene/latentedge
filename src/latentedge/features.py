import pandas as pd


def compute_features(bars: pd.DataFrame, return_windows: list[int], volatility_window: int) -> pd.DataFrame:
    result = bars.copy()
    price = result["price_usdc_per_weth"]

    for n in return_windows:
        result[f"return_{n}"] = price.pct_change(periods=n)

    one_bar_return = price.pct_change(periods=1)
    result["volatility"] = one_bar_return.rolling(window=volatility_window, min_periods=volatility_window).std()

    had_swap = result["swap_count"] > 0
    groups = had_swap.cumsum()
    result["bars_since_swap"] = result.groupby(groups).cumcount()
    result.loc[had_swap, "bars_since_swap"] = 0

    return result
