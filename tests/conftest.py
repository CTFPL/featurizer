import pytest

import polars as pl

from src.features.average import MAFeatureCalculator
from src.data.asset import AssetPriceStorage


@pytest.fixture(scope="session")
def test_minute_data():
    return pl.read_parquet("./tests/static/test_minute.parquet")


@pytest.fixture()
def asset_test(test_minute_data):
    return AssetPriceStorage("test", test_minute_data)


@pytest.fixture(scope="session")
def ma_features(test_minute_data):
    asset = AssetPriceStorage("test", test_minute_data)
    ma_calc = MAFeatureCalculator(periods=120, verbose=False, override_features=True)
    ma_calc.name = "ma"
    ma_calc.calculate(asset)

    return asset