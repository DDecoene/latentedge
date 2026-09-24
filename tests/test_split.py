import pandas as pd
import pytest

from latentedge.split import InsufficientDataError, chronological_split


def test_split_is_chronological_and_non_overlapping():
    df = pd.DataFrame({"bar_start": range(1000), "value": range(1000)})
    train, validate, test = chronological_split(df, train_fraction=0.7, validate_fraction=0.15)

    assert train["bar_start"].max() < validate["bar_start"].min()
    assert validate["bar_start"].max() < test["bar_start"].min()
    assert len(train) + len(validate) + len(test) == len(df)


def test_split_raises_on_insufficient_data():
    df = pd.DataFrame({"bar_start": range(50), "value": range(50)})
    with pytest.raises(InsufficientDataError):
        chronological_split(df, train_fraction=0.7, validate_fraction=0.15)
