import pytest
import talib

from src.features.talib import TALibFeaturizer
from tests.features.base import base_test_featurizer_eq


def test_talib(t_asset):
    featurizer = TALibFeaturizer(
        "ma_low", 
        func=lambda kwargs: talib.MA(real=kwargs["open"], timeperiod=360), 
        add_to_asset=True
    )
    featurizer.get_features_batch(t_asset)
    
    dd = t_asset.to_polars()
    assert "ma_low" in dd.columns

    t_asset.delete_field("ma_low")

    assert "ma_low" not in t_asset.to_polars().columns


@pytest.mark.parametrize("timeperiod", [1, 100])
def test_talib_batch_eq_iter(t_asset_small, timeperiod):
    asset = t_asset_small
    base_test_featurizer_eq(
        asset, 
        TALibFeaturizer(
            "batch", 
            func=lambda kwargs: talib.MA(real=kwargs["open"], timeperiod=timeperiod), 
            add_to_asset=True
        ),
        TALibFeaturizer(
            "iter", 
            func=lambda kwargs: talib.MA(real=kwargs["open"], timeperiod=timeperiod), 
            add_to_asset=True
        )
    )
