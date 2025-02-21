import pytest
import polars as pl

from src.features.lag import LagTicksFeaturizer, LagTimeDeltaFeaturizer


@pytest.mark.parametrize("lag", [0, 1, 10, 100])
def test_lag_ticks_features(t_asset, lag):
    featurizer = LagTicksFeaturizer(
        "lag", 
        feature="close",
        lag=lag,
        add_to_asset=True,
    )
    featurizer.get_features_batch(t_asset)

    shifted_series = t_asset.to_polars().select(
        (pl.col("lag") == pl.col("close").shift(lag)).alias("eq")
    )

    # проверка, что кол-во нулов совпадает с размером лага
    assert shifted_series.null_count()["eq"][0] == lag
    # проверка, что лаги совпадают
    assert shifted_series.mean()["eq"][0] == 1
