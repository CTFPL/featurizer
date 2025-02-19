from datetime import datetime, timedelta

from src.data.index import CandlesAssetData
from src.features.base import BaseFeaturizer


class LagTicksFeaturizer(BaseFeaturizer):
    def __init__(self, name: str, feature: str, lag: int, add_to_asset: bool = False):
        super().__init__(add_to_asset=add_to_asset)

        self.name = name
        self.feature = feature
        self.lag = lag

    def get_features_batch(self, asset: CandlesAssetData):
        output = []

        for dt in asset.sorted_dates:
            output.append(self.get_features_iter(asset, dt))

        return (asset.sorted_dates, output)

    def get_features_iter(self, asset: CandlesAssetData, dt: datetime):
        output = asset.get_lag_ticks(dt=dt, ticks=self.lag)
        output = output[self.feature] if output is not None else None

        if self.add_to_asset:
            asset.update(dt, {self.name: output})
        
        return (dt, output)


class LagTimeDeltaFeaturizer(BaseFeaturizer):
    def __init__(self, name: str, feature: str, lag: timedelta, add_to_asset: bool = False):
        super().__init__(add_to_asset=add_to_asset)

        self.name = name
        self.feature = feature
        self.lag = lag

    def get_features_batch(self, asset: CandlesAssetData):
        output = []

        for dt in asset.sorted_dates:
            output.append(self.get_features_iter(asset, dt))

        return (asset.sorted_dates, output)

    def get_features_iter(self, asset: CandlesAssetData, dt: datetime):
        output = asset.get_lag_timedelta(dt=dt, td=self.lag)

        if self.add_to_asset:
            asset.update(dt, {self.name: output})
        
        return (dt, output)
