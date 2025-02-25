import bisect
from datetime import datetime, timezone, timedelta
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

from src.db.clickhouse import get_connector, read_data


class CandlesAssetData:
    dt_col = "ts"
    low_col = "low"
    high_col = "high"
    close_col = "close"
    open_col = "open"
    volume_col = "volume"

    def __init__(
        self,
        name: str,
        data: dict[datetime, dict] | None = None,
        save_dir: Path | None = None,  
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

        self.name = name
        self.save_dir = save_dir

    def __len__(self):
        return len(self.sorted_dates)

    def trim_dt(self, date_from: datetime | None = None, date_to: datetime | None = None):
        if date_from is not None:
            self.sorted_dates = [d for d in self.sorted_dates if d >= date_from]

        if date_to is not None:
            self.sorted_dates = [d for d in self.sorted_dates if d <= date_to]

        self.data = {dt: self.data[dt] for dt in self.sorted_dates}

        return self
    
    def trim_ticks(self, head: int | None = None, tail: int | None = None):
        if head is not None:
            self.sorted_dates = self.sorted_dates[:head]

        if tail is not None:
            self.sorted_dates = self.sorted_dates[-tail:]

        self.data = {dt: self.data[dt] for dt in self.sorted_dates}

        return self

    @classmethod
    def assert_data(cls, data: dict):
        assert cls.low_col in data, data
        assert cls.high_col in data, data
        assert cls.close_col in data, data
        assert cls.open_col in data, data
        assert cls.volume_col in data, data

    @classmethod
    def from_polars(cls, df: pl.DataFrame, name: str, save_dir: Path | None = None, check_columns: bool = True):
        data = {
            d[cls.dt_col]: d 
            for d in df.to_dicts()
        }

        return cls(name=name, data=data, save_dir=save_dir, check_columns=check_columns)
    
    @classmethod
    def from_clickhouse(
        cls, 
        uid: str, 
        date_from: str, 
        date_to: str, 
        name: str,
        filter_weekend: bool = True,
        save_dir: Path | None = None, 
        check_columns: bool = True,
    ):
        connector = get_connector()
        filter_weekend_str = "toDayOfWeek(toDateTime(`time`), 0) <= 5" if filter_weekend else "1=1"
        query = f"""
        select
            uid
            , high `{cls.high_col}`
            , low `{cls.low_col}`
            , `close` `{cls.close_col}`
            , `open` `{cls.open_col}`
            , volume_from_trades `{cls.volume_col}`
            , toDateTime(`time`) `{cls.dt_col}`
        FROM `default`.aggregations_fast
        where 1=1
            and uid = '{uid}'
            and toDateTime(`time`) >= '{date_from}'
            and toDateTime(`time`) <= '{date_to}'
            and {filter_weekend_str}
        order by ts asc
        """
        df_pd = connector.query_df(query)
        data = {
            d[cls.dt_col].to_pydatetime(): {
                k: v if k != cls.dt_col else v.to_pydatetime()
                for k, v in d.items()
            }
            for d in df_pd.to_dict(orient="records")
        }

        return cls(name=name, data=data, save_dir=save_dir, check_columns=check_columns)
    
    def merge(self, other):
        other_sorted_dates = other.sorted_dates
        max_sorted_date = self.sorted_dates[-1]

        for dt in other_sorted_dates:
            if dt <= max_sorted_date:
                continue

            self.sorted_dates.append(dt)
            self.data[dt] = other.data[dt]

        return self
    
    def __preprocess_dt(self, dt: datetime):
        return dt
        return dt.replace(tzinfo=timezone.utc)
    
    def _get_closest_right_index(self, dt: datetime):
        return bisect.bisect_right(self.sorted_dates, dt) - 1
    
    def get(self, dt: datetime) -> dict | None:
        """
        Включая dt
        """
        # dt = dt.replace(tzinfo=timezone.utc)
        dt = self.__preprocess_dt(dt)
        if dt in self.data:
            return self.data[dt]
        
        index = self._get_closest_right_index(dt)
        # Значит что dt < min(dt)
        if index == 0:
            return None
        
        dt_index = self.sorted_dates[index]
        return self.data[dt_index]
    
    def get_lag_ticks(self, dt: datetime, ticks: int = 1) -> dict | None:
        # dt = dt.replace(tzinfo=timezone.utc)
        dt = self.__preprocess_dt(dt)

        index_to = self._get_closest_right_index(dt)
        index_lag = index_to - ticks

        if index_lag < 0:
            return None
        
        index_dt = self.sorted_dates[index_lag]

        return self.data[index_dt]
    
    def get_lag_timedelta(self, dt: datetime, td: timedelta) -> dict | None:
        # dt = dt.replace(tzinfo=timezone.utc)
        dt = self.__preprocess_dt(dt)
        dt_lag = dt - td

        index = bisect.bisect_left(self.sorted_dates, dt_lag)
        dt_index = self.sorted_dates[index]

        return self.data[dt_index]
    
    def get_slice(self, dt_from: datetime, dt_to: datetime, as_np: bool = True) -> list[dict]:
        """
        Включая dt_to
        Включая dt_from
        """
        # dt_from = dt_from.replace(tzinfo=timezone.utc)
        # dt_to = dt_to.replace(tzinfo=timezone.utc)
        dt_from = self.__preprocess_dt(dt_from)
        dt_to = self.__preprocess_dt(dt_to)

        start_idx = bisect.bisect_left(self.sorted_dates, dt_from)
        end_idx = bisect.bisect_right(self.sorted_dates, dt_to)

        dt_indices = self.sorted_dates[start_idx:end_idx]
        return [self.data[dt] for dt in dt_indices]

    def get_last_n_ticks(self, dt_to: datetime, n_ticks: int, as_np: bool = True) -> list[dict]:
        """
        Включая dt_to
        """
        dt_to = self.__preprocess_dt(dt_to)

        index_to = self._get_closest_right_index(dt_to) + 1
        index_from = max(0, index_to - n_ticks)

        return [self.data[dt] for dt in self.sorted_dates[index_from:index_to]]
    
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
