from datetime import datetime
from abc import ABC, abstractmethod

from src.data.index import CandlesAssetData


class BaseFeaturizer(ABC):
    def __init__(self, add_to_asset: bool = False):
        self.add_to_asset = add_to_asset

    @abstractmethod
    def get_features_batch(self, asset: CandlesAssetData):
        pass

    @abstractmethod
    def get_features_iter(self, asset: CandlesAssetData, dt: datetime):
        pass
