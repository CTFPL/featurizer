from abc import abstractmethod
from datetime import datetime

from tqdm.auto import tqdm
import pandas as pd
import numpy as np

from src.features.base import BaseFeaturizer
from src.data.index import CandlesAssetData


class LagModelPredictPriceFeaturizer(BaseFeaturizer):
    def __init__(
        self, 
        name: str, 
        model_features: list[str], 
        target_col: str, 
        model_params: dict | None = None,
        n_lags: int = 100,
        lag_target: int = 10,
        add_to_asset: bool = False,
    ):
        super().__init__(add_to_asset=add_to_asset)

        self.name = name
        self.model_features = model_features
        self.target_col = target_col
        self.model_params = {} if model_params is None else model_params
        
        self.lag_target = lag_target
        self.n_lags = n_lags + lag_target

    @abstractmethod
    def _create_model(self):
        pass

    @abstractmethod
    def _fit_model(self, df: pd.DataFrame, model):
        pass

    @abstractmethod
    def _predict_model(self, df: pd.DataFrame, model) -> np.ndarray:
        pass

    def get_features_batch(self, asset: CandlesAssetData):
        output = []

        for dt in tqdm(asset.sorted_dates):
            output.append(self.get_features_iter(asset, dt))

        return (asset.sorted_dates, output)
    
    def get_features_iter(self, asset: CandlesAssetData, dt: datetime):
        data = asset.get_last_n_ticks(dt, n_ticks=self.n_lags)

        if data is None:
            output = None
        elif len(data) < self.n_lags:
            output = None
        else:
            df = pd.DataFrame.from_records(data).sort_values(asset.dt_col)
            df["target"] = df[self.target_col].shift(-self.lag_target)

            df_train = df.iloc[:-self.lag_target]
            df_test = df.iloc[-self.lag_target:]

            model = self._create_model()
            self._fit_model(df_train, model)
            output = self._predict_model(df_test, model)[-1]
        
        if self.add_to_asset:
            asset.update(dt, {self.name: output})

        return (dt, output)
