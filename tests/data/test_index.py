from datetime import datetime, timedelta
import zoneinfo

import pytest


def test_smoke(t_asset):
    t_asset.get_last_n_ticks(datetime(2025, 1, 1, 0, 0, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC')), 2)
    t_asset.to_polars()
    t_asset.delete_field("lot")


def test_get(t_asset):
    test_idx = 1000
    test_dt = t_asset.sorted_dates[test_idx]

    assert t_asset.get(test_dt)[t_asset.dt_col] == test_dt
    
    assert t_asset.get(test_dt + timedelta(seconds=1))[t_asset.dt_col] == test_dt

    assert t_asset.get(test_dt - timedelta(seconds=1))[t_asset.dt_col] == t_asset.sorted_dates[test_idx - 1]


@pytest.mark.parametrize("ticks", [0, 1, 100])
def test_get_lag_ticks(t_asset, ticks):
    test_idx = 1000
    test_dt = t_asset.sorted_dates[test_idx]

    assert t_asset.get_lag_ticks(test_dt, ticks=ticks)[t_asset.dt_col] == t_asset.sorted_dates[test_idx - ticks]


def test_get_lag_timedelta(t_asset):
    test_idx = 1000
    test_dt = t_asset.sorted_dates[test_idx]

    dt = t_asset.get_lag_timedelta(test_dt, timedelta(seconds=0))[t_asset.dt_col]
    assert dt == test_dt

    dt = t_asset.get_lag_timedelta(test_dt, timedelta(seconds=30))[t_asset.dt_col]
    assert dt == test_dt

    dt = t_asset.get_lag_timedelta(test_dt, timedelta(seconds=90))[t_asset.dt_col]
    assert dt == t_asset.sorted_dates[test_idx - 1]


def test_get_slice(t_asset):
    test_idx = 1000
    n_ticks = 10
    
    dt_to = t_asset.sorted_dates[test_idx]
    dt_from = t_asset.sorted_dates[test_idx - n_ticks]

    slice_data = t_asset.get_slice(dt_from=dt_from, dt_to=dt_to)

    assert slice_data[0][t_asset.dt_col] == dt_from
    assert slice_data[-1][t_asset.dt_col] == dt_to
    assert len(slice_data) == n_ticks + 1


def test_get_slice_jitter_dt_from(t_asset):
    test_idx = 1000
    n_ticks = 10
    
    dt_to = t_asset.sorted_dates[test_idx]
    dt_from = t_asset.sorted_dates[test_idx - n_ticks] + timedelta(seconds=30)

    slice_data = t_asset.get_slice(dt_from=dt_from, dt_to=dt_to)

    assert slice_data[0][t_asset.dt_col] == t_asset.sorted_dates[test_idx - n_ticks + 1]
    assert slice_data[-1][t_asset.dt_col] == dt_to
    assert len(slice_data) == n_ticks


def test_get_slice_jitter_dt_to(t_asset):
    test_idx = 1000
    n_ticks = 10
    
    dt_to = t_asset.sorted_dates[test_idx] - timedelta(seconds=30)
    dt_from = t_asset.sorted_dates[test_idx - n_ticks]

    slice_data = t_asset.get_slice(dt_from=dt_from, dt_to=dt_to)

    assert slice_data[0][t_asset.dt_col] == dt_from
    assert slice_data[-1][t_asset.dt_col] == t_asset.sorted_dates[test_idx - 1]
    assert len(slice_data) == n_ticks


def test_get_slice_jitter_both(t_asset):
    test_idx = 1000
    n_ticks = 10
    
    dt_to = t_asset.sorted_dates[test_idx] - timedelta(seconds=30)
    dt_from = t_asset.sorted_dates[test_idx - n_ticks] + timedelta(seconds=30)

    slice_data = t_asset.get_slice(dt_from=dt_from, dt_to=dt_to)

    assert slice_data[0][t_asset.dt_col] == t_asset.sorted_dates[test_idx - n_ticks + 1]
    assert slice_data[-1][t_asset.dt_col] == t_asset.sorted_dates[test_idx - 1]
    assert len(slice_data) == n_ticks - 1


def test_get_slice_right_birder(t_asset):
    test_idx = len(t_asset.sorted_dates) - 1
    n_ticks = 10
    
    dt_to = t_asset.sorted_dates[test_idx]
    dt_from = t_asset.sorted_dates[test_idx - n_ticks]

    slice_data = t_asset.get_slice(dt_from=dt_from, dt_to=dt_to)

    assert slice_data[0][t_asset.dt_col] == dt_from
    assert slice_data[-1][t_asset.dt_col] == dt_to
    assert len(slice_data) == n_ticks + 1


def test_get_last_n_ticks(t_asset):
    test_idx = 1000
    n_ticks = 10
    
    dt_to = t_asset.sorted_dates[test_idx]
    dt_from = t_asset.sorted_dates[test_idx - n_ticks + 1]

    slice_data = t_asset.get_last_n_ticks(dt_to=dt_to, n_ticks=n_ticks)

    assert slice_data[0][t_asset.dt_col] == dt_from
    assert slice_data[-1][t_asset.dt_col] == dt_to
    assert len(slice_data) == n_ticks


def test_get_last_n_ticks_jitter_right(t_asset):
    test_idx = 1000
    n_ticks = 10
    
    dt_to = t_asset.sorted_dates[test_idx] + timedelta(seconds=30)
    dt_from = t_asset.sorted_dates[test_idx - n_ticks + 1]

    slice_data = t_asset.get_last_n_ticks(dt_to=dt_to, n_ticks=n_ticks)

    assert slice_data[0][t_asset.dt_col] == dt_from
    assert slice_data[-1][t_asset.dt_col] == dt_to - timedelta(seconds=30)
    assert len(slice_data) == n_ticks
