from pathlib import Path

import pandas as pd

from latentedge.schema import SwapRecord

DEDUPE_KEYS = ["tx_hash", "log_index"]

# sqrt_price_x96 (uint160) and liquidity (uint128) routinely exceed
# int64's range (Q96 scaling alone puts sqrt_price_x96 at ~2**96 for any
# price near 1.0), which overflows Parquet's default int64 column
# inference. Store them as strings and parse back to int on read.
BIG_INT_COLUMNS = ["sqrt_price_x96", "liquidity"]


class EmptyIngestError(Exception):
    pass


def write_swaps(records: list[SwapRecord], path: Path) -> None:
    if not records:
        raise EmptyIngestError("no swap records to write — an empty block range or a data-source issue upstream, not a valid ingest result")

    new_df = pd.DataFrame([r.model_dump() for r in records])
    for column in BIG_INT_COLUMNS:
        new_df[column] = new_df[column].astype(str)

    if path.exists():
        existing_df = pd.read_parquet(path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=DEDUPE_KEYS, keep="first")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined.to_parquet(path, index=False)


def read_swaps(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for column in BIG_INT_COLUMNS:
        # .astype(int) coerces to numpy int64 and overflows on
        # Q96-scale values; .apply(int) keeps arbitrary-precision Python
        # ints in an object column instead.
        df[column] = df[column].apply(int)
    return df.sort_values("timestamp").reset_index(drop=True)
