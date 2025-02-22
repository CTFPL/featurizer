from datetime import datetime

import numpy as np

from src.data.index import CandlesAssetData
from src.features.base import BaseFeaturizer


class TALibFeaturizer(BaseFeaturizer):
    def __init__(self, names: list[str] | str, func, add_to_asset: bool = False):
        super().__init__(add_to_asset=add_to_asset)

        self.names = [names, ] if isinstance(names, str) else names
        self.func = func

    def get_features_batch(self, asset: CandlesAssetData):
        """
        По дефолту talib использует текущее значение для расчета индикаторов, но кажется,
        что нужно исключать текущую цену из рассмотрения
        """
        open_ = np.array([asset.data[dt][asset.open_col] for dt in asset.sorted_dates])
        high_ = np.array([asset.data[dt][asset.high_col] for dt in asset.sorted_dates])
        low_ = np.array([asset.data[dt][asset.low_col] for dt in asset.sorted_dates])
        close_ = np.array([asset.data[dt][asset.close_col] for dt in asset.sorted_dates])
        volume_ = np.array([asset.data[dt][asset.volume_col] for dt in asset.sorted_dates])

        outputs = self.func(dict(open=open_, high=high_, low=low_, close=close_, volume=volume_))
        if isinstance(outputs, np.ndarray):
            outputs = [outputs, ]

        assert len(outputs) == len(self.names)
        for name, output_ in zip(self.names, outputs):
            output = [{name: None}, ]
            output.extend([{name: v} for v in output_[:-1].tolist()])
            if self.add_to_asset:
                asset.update_batch(asset.sorted_dates, output)

        return (asset.sorted_dates, outputs)

    def get_features_iter(self, asset: CandlesAssetData, dt: datetime):
        raise NotImplementedError()
