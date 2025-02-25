import pytest
import polars as pl

from src.data.index import CandlesAssetData


@pytest.fixture()
def t_asset():
    return CandlesAssetData.from_polars(pl.read_parquet("./tests/static/T.parquet"), name="test")


@pytest.fixture()
def t_asset_small():
    return CandlesAssetData.from_polars(pl.read_parquet("./tests/static/T.parquet").head(1000), name="test")
