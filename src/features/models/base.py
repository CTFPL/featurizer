from abc import abstractmethod
from datetime import datetime, timedelta
import random

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
        cached_model_seconds: int = 0,
    ):
        super().__init__(add_to_asset=add_to_asset)

        self.name = name
        self.model_features = model_features
        self.target_col = target_col
        self.model_params = {} if model_params is None else model_params
        
        self.lag_target = lag_target
        self.n_lags = n_lags + lag_target

        self.cached_model_seconds = cached_model_seconds
        self.cached_model = None
        self.cached_model_dt = None

    @abstractmethod
    def _create_model(self):
        pass

    @abstractmethod
    def _fit_model(self, df: pd.DataFrame, model):
        pass

    @abstractmethod
    def _predict_model(self, df: pd.DataFrame, model) -> np.ndarray:
        pass

    @abstractmethod
    def _check_model_is_fitted(self, model):
        pass

    def create_model(self, dt: datetime):
        if self.cached_model_dt is None:
            return self._create_model()
        
        if self.cached_model_seconds == 0:
            return self._create_model()
        
        if self.cached_model_dt + timedelta(seconds=self.cached_model_seconds) < dt:
            return self._create_model()
        
        return self.cached_model
    
    def fit_model(self, df: pd.DataFrame, model, dt: datetime):
        if self._check_model_is_fitted(model):
            return 
        
        self._fit_model(df, model)
        self.cached_model = model
        self.cached_model_dt = dt

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

            model = self.create_model(dt)
            self.fit_model(df_train, model, dt)
            output = self._predict_model(df_test, model)[-1]
        
        if self.add_to_asset:
            asset.update(dt, {self.name: output})

        return (dt, output)
    

class DumbRegressionModelFeaturizer(LagModelPredictPriceFeaturizer):
    class DumbModel:
        def fit(self, *args, **kwargs):
            pass

        def predict(self, features: pd.DataFrame, *args, **kwargs):
            return np.array([random.random() for _ in range(len(features))])


    def _fit_model(self, df: pd.DataFrame, model):
        model.fit(df[self.model_features], df["target"], sample_weight=df["sample_weight"])

    def _create_model(self):
        return self.DumbModel()

    def _predict_model(self, df: pd.DataFrame, model) -> np.ndarray:
        return model.predict(df[self.model_features])
    
    def _check_model_is_fitted(self, model):
        return True


class LagClassificationModelPredictPriceFeaturizer(LagModelPredictPriceFeaturizer):
    def __init__(
        self, 
        name: str, 
        model_features: list[str], 
        target_col: str, 
        model_params: dict | None = None,
        n_lags: int = 100,
        lag_target: int = 10,
        add_to_asset: bool = False,
        predict_up: bool = True,
        min_up_perc: float = 0.0008,
        use_time_weights: bool = False,
        cached_model_seconds: int = 0,
    ):
        super().__init__(
            name=name, 
            model_features=model_features, 
            target_col=target_col, 
            model_params=model_params,
            n_lags=n_lags,
            lag_target=lag_target,
            add_to_asset=add_to_asset,
            cached_model_seconds=cached_model_seconds,
        )

        self.predict_up = predict_up
        self.min_up_perc = min_up_perc
        self.use_time_weights = use_time_weights

    def _add_sample_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sample_weight"] = 1
        if self.use_time_weights:
            df["sample_weight"] = np.arange(df.shape[0])[::-1] + 1
            df["sample_weight"] = 1 / np.log(df["sample_weight"] + 1)

        return df

    def _preprocess_datasets(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self._add_sample_weights(df)
        df["max_target_next_periods"] = df[self.target_col][::-1].rolling(self.lag_target, min_periods=1).max()[::-1].shift(-1)
            
        if self.predict_up:
            # проверка, что цена в следующий период будет выше на min_up_perc
            max_price_next_shift = df[self.target_col][::-1].rolling(self.lag_target, min_periods=1).max()[::-1].shift(-1)
            df["target"] = df[self.target_col] * (1 + self.min_up_perc) < max_price_next_shift
        else:
            # цена будет ниже на min_up_perc, чем сейчас
            min_price_next_shift = df[self.target_col][::-1].rolling(self.lag_target, min_periods=1).min()[::-1].shift(-1)
            df["target"] = df[self.target_col] * (1 - self.min_up_perc) > min_price_next_shift

        df_train = df.iloc[:-self.lag_target]
        df_test = df.iloc[-self.lag_target:]

        return df_train, df_test

    def get_features_iter(self, asset: CandlesAssetData, dt: datetime):
        data = asset.get_last_n_ticks(dt, n_ticks=self.n_lags)

        if data is None:
            output = None
        elif len(data) < self.n_lags:
            output = None
        else:
            df = pd.DataFrame.from_records(data).sort_values(asset.dt_col)
            df_train, df_test = self._preprocess_datasets(df)

            if df_train["target"].nunique() == 1:
                output = df_train["target"].max()
            else:
                model = self.create_model(dt)
                self.fit_model(df_train, model, dt)
                output = self._predict_model(df_test, model)[-1]
        
        if self.add_to_asset:
            asset.update(dt, {self.name: output})

        return (dt, output)


class DumbClassificationModelFeaturizer(LagClassificationModelPredictPriceFeaturizer):
    class DumbModel:
        def fit(self, *args, **kwargs):
            pass

        def predict(self, features: pd.DataFrame, *args, **kwargs):
            return np.array([random.random() for _ in range(len(features))]) < 0.5
        

    def _fit_model(self, df: pd.DataFrame, model):
        model.fit(df[self.model_features], df["target"], sample_weight=df["sample_weight"])

    def _create_model(self):
        return self.DumbModel()

    def _predict_model(self, df: pd.DataFrame, model) -> np.ndarray:
        return model.predict(df[self.model_features])
    
    def _check_model_is_fitted(self, model):
        return True
