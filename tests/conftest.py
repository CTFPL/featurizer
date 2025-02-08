import pytest

import polars as pl


@pytest.fixture()
def t_minute_data():
    return pl.read_parquet("./tests/static/T.parquet")
