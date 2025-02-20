import pandas as pd
import numpy as np
import catboost as cb

from src.features.models.base import LagModelPredictPriceFeaturizer


class ModelPredictPriceFeaturizer(LagModelPredictPriceFeaturizer):
    _default_model_params = {
        "iterations": 20,
        "random_state": 32,
        "verbose": 0,
    }

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
        super().__init__(
            name=name, 
            model_features=model_features, 
            target_col=target_col, 
            model_params=model_params if model_params is not None else self._default_model_params,
            n_lags=n_lags,
            lag_target=lag_target,
            add_to_asset=add_to_asset,
        )

    def _create_model(self):
        return cb.CatBoostRegressor(**self.model_params)
    
    def _fit_model(self, df: pd.DataFrame, model):
        model.fit(df[self.model_features], df["target"])

    def _predict_model(self, df: pd.DataFrame, model) -> np.ndarray:
        return model.predict(df[self.model_features])
