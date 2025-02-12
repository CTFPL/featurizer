from datetime import datetime

from features.base import LagTicksWindowCalculator
from src.data.asset import AssetFeatureCalculatorIterable, AssetPriceStorage


class MAFeatureCalculator(LagTicksWindowCalculator):
    def __init__(self, verbose: bool = False, override_features: bool = False, periods: int = 1, value_col: str = "close"):
        super().__init__(
            verbose=verbose, 
            override_features=override_features,
            window=periods,
            lag=0,
            value_col=value_col,
        )
        self.name = f"MA_{self.name}"

    def calculate_one_unit(self, asset_price_storage: AssetPriceStorage, dt: datetime) -> float | None:
        df = self.get_df(asset_price_storage, dt)

        if df is None:
            return None
        
        return df[self.value_col].mean()


class EMAFeatureCalculator(AssetFeatureCalculatorIterable):
    def __init__(
            self, 
            verbose: bool = False, 
            override_features: bool = False, 
            periods: int = 1, 
            smoothing: int = 2,
            value_col: str = "close"
        ):
        super().__init__(verbose, override_features)
        self.periods = periods
        self.smoothing = smoothing
        self.alpha = smoothing / (1 + self.periods)

        self.value_col = value_col
        self.name = f"EMA_{self.periods}_{self.smoothing:.5f}"

        self.last_ema = None

    def calculate_one_unit(self, asset_price_storage: AssetPriceStorage, dt: datetime) -> float | None:
        # only true for first value
        if self.last_ema is None:
            self.last_ema = asset_price_storage.df[self.value_col][0]
            return self.last_ema

        current_price = asset_price_storage.get_slice_by_ticks(dt, n_ticks=1)
        assert current_price.shape[0] == 1
        current_price = current_price[self.value_col][0]
        
        value = self.alpha * current_price + (1 - self.alpha) * self.last_ema
        self.last_ema = value

        return value