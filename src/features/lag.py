from datetime import datetime, timedelta

import polars as pl

from src.data.asset import AssetFeatureCalculatorIterable, AssetPriceStorage


class LagTicksWindowCalculator(AssetFeatureCalculatorIterable):
    """
    Base class for calculating stats on time frame from past
    given window (ticks) and lag (ticks)
    """
    def __init__(
            self, 
            verbose: bool = False, 
            override_features: bool = False, 
            window: int = 1,
            lag: int = 1, 
            value_col: str = "close"
        ):
        super().__init__(verbose, override_features)
        self.window = window
        self.lag = lag
        self.value_col = value_col
        self.name = f"LAG_{self.lag}_WINDOW_{self.window}"

    def get_df(self, asset_price_storage: AssetPriceStorage, dt: datetime) -> pl.DataFrame | None:
        """
        Returns window dataframe
        """
        df = asset_price_storage.get_slice_by_ticks(dt, n_ticks=self.window + self.lag)

        # insufficient data
        if len(df) <= 0:
            return None
        
        return df[:self.window]
    

class LagTicksFeatures(LagTicksWindowCalculator):
    def __init__(
            self, 
            verbose: bool = False, 
            override_features: bool = False, 
            lag: int = 1, 
            value_col: str = "close"
        ):
        super().__init__(
            verbose=verbose,
            override_features=override_features,
            window=1,
            lag=lag,
            value_col=value_col
        )
        self.name += "_LAG_FEATURES"

    def calculate_one_unit(self, asset_price_storage: AssetPriceStorage, dt: datetime) -> float | None:
        df = self.get_df(asset_price_storage, dt)

        if df is None:
            return None
        
        return df[self.value_col][0]


class LagDTWindowCalculator(AssetFeatureCalculatorIterable):
    """
    Base class for calculating stats on time frame from past
    given window (dt) and lag (dt)
    """
    def __init__(
            self, 
            verbose: bool = False, 
            override_features: bool = False, 
            window_sec: int = 1,
            lag_sec: int = 0, 
            value_col: str = "close",
        ):
        super().__init__(verbose, override_features)
        self.window_sec = window_sec
        self.lag_sec = lag_sec
        self.window_timedelta = timedelta(seconds=self.window_sec)
        self.lag_timedelta = timedelta(seconds=self.lag_sec)
        self.total_timedelta = self.window_timedelta + self.lag_timedelta

        self.value_col = value_col
        self.name = f"LAG_{self.lag_sec}_SEC_WINDOW_{self.window_sec}_SEC"

    def get_df(self, asset_price_storage: AssetPriceStorage, dt: datetime) -> pl.DataFrame | None:
        """
        Returns window dataframe
        """
        if self.window_sec == 0:
            df = asset_price_storage.get_slice_by_ticks(dt - self.total_timedelta, n_ticks=1)
        else:
            df = asset_price_storage.get_slice_by_borders(dt - self.total_timedelta, dt - self.lag_timedelta)

        # insufficient data
        if df is None:
            return df
        
        return df


class LagIntervalCalculator(LagDTWindowCalculator):
    def __init__(
            self, 
            verbose: bool = False, 
            override_features: bool = False, 
            interval_sec: int = 1, 
            value_col: str = "close",
        ):
        super().__init__(
            verbose=verbose,
            override_features=override_features,
            window_sec=1
        )
        self.name = f"LAG_Interval_{self.interval_sec}"

    def calculate_one_unit(self, asset_price_storage: AssetPriceStorage, dt: datetime) -> float | None:
        assert "trading_session" in asset_price_storage.df.columns
        
        df = asset_price_storage.get_slice_by_interval(dt, interval=self.interval_timedelta)

        if len(df) <= 0:
            # insufficient data
            return None
        
        return df[self.value_col][0]
