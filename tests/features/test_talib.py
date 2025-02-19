import talib

from src.features.talib import TALibFeaturizer


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
