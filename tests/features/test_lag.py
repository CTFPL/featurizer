import pytest
import polars as pl

from src.features.lag import LagTicksFeaturizer, LagTimeDeltaFeaturizer


@pytest.mark.parametrize("lag", [0, 1, 10, 100])
def test_lag(t_asset, lag):
    featurizer = LagTicksFeaturizer(
        "lag_0", 
        feature="close",
        lag=lag,
        add_to_asset=True,
    )
    featurizer.get_features_batch(t_asset)
    # assert t_asset.to_polars().select((pl.col("lag_0") == pl.col("close")).mean()) == 1
