from datetime import datetime


from features.base import LagDTWindowCalculator, LagTicksWindowCalculator
from src.data.asset import AssetPriceStorage


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
