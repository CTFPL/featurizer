import bisect
from datetime import datetime, timezone, timedelta

import polars as pl
from tqdm.auto import tqdm


class CandlesAssetData:
    dt_col = "ts"
    low_col = "low"
    high_col = "high"
    close_col = "close"
    open_col = "open"
    volume_col = "volume"

    def __init__(
        self,
        data: dict[datetime, dict] | None = None,
        check_columns: bool = False
    ):
        if data is None:
            data = {}
        
        self.sorted_dates = sorted(list(data.keys()))
        self.data = {}
        for dt in self.sorted_dates:
            if check_columns:
                self.assert_data(data[dt])
            self.data[dt] = data[dt]

    def __len__(self):
        return len(self.sorted_dates)

    @classmethod
    def assert_data(cls, data: dict):
        assert cls.low_col in data, data
        assert cls.high_col in data, data
        assert cls.close_col in data, data
        assert cls.open_col in data, data
        assert cls.volume_col in data, data

    @classmethod
    def from_polars(cls, df: pl.DataFrame, check_columns: bool = True):
        data = {
            d[cls.dt_col]: d 
            for d in df.to_dicts()
        }

        return cls(data=data, check_columns=check_columns)
    
    def _get_closest_right_index(self, dt: datetime):
        dt = dt.replace(tzinfo=timezone.utc)
        return bisect.bisect_right(self.sorted_dates, dt) - 1
    
    def get(self, dt: datetime) -> dict | None:
        """
        Включая dt
        """
        dt = dt.replace(tzinfo=timezone.utc)
        if dt in self.data:
            return self.data[dt]
        
        index = self._get_closest_right_index(dt)
        # Значит что dt < min(dt)
        if index == 0:
            return None
        
        dt_index = self.sorted_dates[index]
        return self.data[dt_index]
    
    def get_lag_ticks(self, dt: datetime, ticks: int = 1) -> dict | None:
        dt = dt.replace(tzinfo=timezone.utc)

        index_to = self._get_closest_right_index(dt)
        index_lag = index_to - ticks

        if index_lag < 0:
            return None
        
        index_dt = self.sorted_dates[index_lag]

        return self.data[index_dt]
    
    def get_lag_timedelta(self, dt: datetime, td: timedelta) -> dict | None:
        dt = dt.replace(tzinfo=timezone.utc)
        dt_lag = dt - td

        index = bisect.bisect_left(self.sorted_dates, dt_lag)
        dt_index = self.sorted_dates[index]

        return self.data[dt_index]
    
    def get_slice(self, dt_from: datetime, dt_to: datetime, as_np: bool = True) -> list[dict]:
        """
        Включая dt_to
        Включая dt_from
        """
        dt_from = dt_from.replace(tzinfo=timezone.utc)
        dt_to = dt_to.replace(tzinfo=timezone.utc)

        start_idx = bisect.bisect_left(self.sorted_dates, dt_from)
        end_idx = bisect.bisect_right(self.sorted_dates, dt_to)

        dt_indices = self.sorted_dates[start_idx:end_idx]
        return [self.data[dt] for dt in dt_indices]

    def get_last_n_ticks(self, dt_to: datetime, n_ticks: int, as_np: bool = True) -> list[dict]:
        """
        Включая dt_to
        """
        index_to = self._get_closest_right_index(dt_to) + 1
        index_from = max(0, index_to - n_ticks)

        return [self.data[dt] for dt in self.sorted_dates[index_from:index_to]]
    
    def cumulative_iterator(self):
        returned_data = []
        for dt in self.sorted_dates:
            yield returned_data
            returned_data.append(self.data[dt])

    def to_polars(self) -> pl.DataFrame:
        return pl.from_dicts(list(self.data.values()), infer_schema_length=None).sort(self.dt_col)
    
    def update(self, dt: datetime, data: dict):
        self.data[dt].update(data)

    def update_batch(self, dts: list[datetime], values: list[dict]):
        assert len(dts) == len(values)
        assert all(dt in self.data for dt in dts)

        for dt, data in tqdm(zip(dts, values)):
            self.data[dt].update(data)

    def delete_field(self, name: str):
        for d in self.data.values():
            d.pop(name, None)
