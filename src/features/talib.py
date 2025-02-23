from datetime import datetime

import numpy as np

from src.data.index import CandlesAssetData
from src.features.base import BaseFeaturizer


class TALibFeaturizer(BaseFeaturizer):
    def __init__(
        self, 
        names: list[str] | str, 
        func, 
        add_to_asset: bool = False
    ):
        super().__init__(add_to_asset=add_to_asset)

        self.names = [names, ] if isinstance(names, str) else names
        self.func = func

    def _get_func_input(self, asset: CandlesAssetData, dt: datetime | None = None):
        def check_dt(dt_):
            return (dt is None) or (dt_ <= dt)
        
        open_ = np.array([asset.data[dt_][asset.open_col] for dt_ in asset.sorted_dates if check_dt(dt_)])
        high_ = np.array([asset.data[dt_][asset.high_col] for dt_ in asset.sorted_dates if check_dt(dt_)])
        low_ = np.array([asset.data[dt_][asset.low_col] for dt_ in asset.sorted_dates if check_dt(dt_)])
        close_ = np.array([asset.data[dt_][asset.close_col] for dt_ in asset.sorted_dates if check_dt(dt_)])
        volume_ = np.array([asset.data[dt_][asset.volume_col] for dt_ in asset.sorted_dates if check_dt(dt_)])

        return dict(open=open_, high=high_, low=low_, close=close_, volume=volume_)

    def get_features_batch(self, asset: CandlesAssetData):
        """
        По дефолту talib использует текущее значение для расчета индикаторов, но кажется,
        что нужно исключать текущую цену из рассмотрения
        """
        func_input = self._get_func_input(asset)

        outputs = self.func(func_input)
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
        func_input = self._get_func_input(asset, dt)

        outputs = self.func(func_input)
        if isinstance(outputs, np.ndarray):
            outputs = [outputs, ]

        for name, outputs_ in zip(self.names, outputs):
            # -2 так как в talib есть лаг
            if len(outputs_) < 2:
                asset.update(dt, {name: None})    
            else:
                asset.update(dt, {name: outputs_[-2]})

    def get_name(self):
        if len(self.names) == 1:
            return self.names[0]
        else:
            raise TypeError(f"Featurizers returns multiple names")