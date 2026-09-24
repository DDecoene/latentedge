import pandas as pd

MIN_ROWS_PER_SPLIT = 100


class InsufficientDataError(Exception):
    pass


def chronological_split(df: pd.DataFrame, train_fraction: float, validate_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("bar_start").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_fraction)
    validate_end = train_end + int(n * validate_fraction)

    train = df.iloc[:train_end]
    validate = df.iloc[train_end:validate_end]
    test = df.iloc[validate_end:]

    for name, split_df in [("train", train), ("validate", validate), ("test", test)]:
        if len(split_df) < MIN_ROWS_PER_SPLIT:
            raise InsufficientDataError(f"{name} split has only {len(split_df)} rows, need at least {MIN_ROWS_PER_SPLIT}")

    return train, validate, test
