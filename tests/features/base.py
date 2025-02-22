from copy import deepcopy

import polars as pl
from tqdm.auto import tqdm

from src.data.index import CandlesAssetData
from src.features.base import BaseFeaturizer


def base_test_featurizer_eq(asset: CandlesAssetData, featurizer_batch: BaseFeaturizer, featurizer_iter: BaseFeaturizer):
    dates = asset.sorted_dates

    featurizer_batch.get_features_batch(asset)

    for dt in tqdm(dates):
        featurizer_iter.get_features_iter(asset, dt=dt)

    dd = asset.to_polars().select("ts", "batch", "iter", (pl.col("batch") == pl.col("iter")).alias("eq"))

    assert dd["batch"].null_count() == dd["iter"].null_count()
    assert dd["eq"].mean() == 1
