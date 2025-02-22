from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from sklearn.discriminant_analysis import StandardScaler
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd

from src.features.models.base import LagClassificationModelPredictPriceFeaturizer, LagModelPredictPriceFeaturizer


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
    def _create_model(self):
        return make_pipeline(DropNAColsTransformer(), SimpleImputer(), StandardScaler(), LinearRegression(n_jobs=1))

    def _fit_model(self, df: pd.DataFrame, model):
        model.fit(df[self.model_features], df["target"])

    def _predict_model(self, df: pd.DataFrame, model) -> np.ndarray:
        return model.predict(df[self.model_features])
    
    def _check_model_is_fitted(self, model):
        try:
            check_is_fitted(model)
        except NotFittedError as e:
            return False
        
        return True


class LogregClassifierFeaturizer(LagClassificationModelPredictPriceFeaturizer):
    """
    Предсказываем, что в следующие lag_target тиков цена будет выше чем текущая 
    на min_up_perc процентов
    """
    def _fit_model(self, df: pd.DataFrame, model):
        model.fit(df[self.model_features], df["target"], logisticregression__sample_weight=df["sample_weight"])

    def _create_model(self):
        return make_pipeline(DropNAColsTransformer(), SimpleImputer(), StandardScaler(), LogisticRegression(n_jobs=1))

    def _predict_model(self, df: pd.DataFrame, model) -> np.ndarray:
        return model.predict(df[self.model_features])

    def _check_model_is_fitted(self, model):
        try:
            check_is_fitted(model)
        except NotFittedError as e:
            return False
        
        return True
