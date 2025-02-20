from datetime import datetime
from typing import Callable

from tqdm.auto import tqdm

from src.features.base import BaseFeaturizer
from src.data.index import CandlesAssetData


class FuncFeaturizer(BaseFeaturizer):
    def __init__(self, name: str, func: Callable[[CandlesAssetData, datetime | None], float | None], add_to_asset: bool = False):
        super().__init__(add_to_asset=add_to_asset)

        self.name = name
        self.func = func

    def get_features_batch(self, asset: CandlesAssetData):
        output = []

        for dt in tqdm(asset.sorted_dates):
            output.append(self.get_features_iter(asset, dt))

        return (asset.sorted_dates, output)

    def get_features_iter(self, asset: CandlesAssetData, dt: datetime):
        output = self.func(asset, dt)

        if self.add_to_asset:
            asset.update(dt, {self.name: output})

        return (dt, output)