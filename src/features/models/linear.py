from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd

from src.features.models.base import LagModelPredictPriceFeaturizer


class DropNAColsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nan_threshold: float = 0.8):
        self.use_cols = None
        self.nan_threshold = nan_threshold

    def fit(self, X, y=None):
        self.use_cols = (np.isnan(X).mean() <= self.nan_threshold)
        return self

    def transform(self, X, y=None):
        X = X.loc[:, self.use_cols]
        return X


class LinearModelPredictPriceFeaturizer(LagModelPredictPriceFeaturizer):
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
            model_params=model_params if model_params is not None else {},
            n_lags=n_lags,
            lag_target=lag_target,
            add_to_asset=add_to_asset,
        )

    def _create_model(self):
        return make_pipeline(DropNAColsTransformer(), SimpleImputer(), StandardScaler(), LinearRegression(n_jobs=1))
    
    def _fit_model(self, df: pd.DataFrame, model):
        model.fit(df[self.model_features], df["target"])

    def _predict_model(self, df: pd.DataFrame, model) -> np.ndarray:
        return model.predict(df[self.model_features])
