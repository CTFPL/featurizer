from datetime import datetime
import pandas as pd
import numpy as np
import catboost as cb

from src.data.index import CandlesAssetData
from src.features.models.base import LagClassificationModelPredictPriceFeaturizer, LagModelPredictPriceFeaturizer


class ModelPredictPriceFeaturizer(LagModelPredictPriceFeaturizer):
    def _create_model(self):
        return cb.CatBoostRegressor(**self.model_params)
    
    def _fit_model(self, df: pd.DataFrame, model):
        model.fit(df[self.model_features], df["target"])

    def _predict_model(self, df: pd.DataFrame, model) -> np.ndarray:
        return model.predict(df[self.model_features])
    
    def _check_model_is_fitted(self, model):
        return model.is_fitted()


class CatboostClassifierFeaturizer(LagClassificationModelPredictPriceFeaturizer):
    """
    Предсказываем, что в следующие lag_target тиков цена будет выше чем текущая 
    на min_up_perc процентов
    """
    def _fit_model(self, df: pd.DataFrame, model):
        model.fit(df[self.model_features], df["target"], sample_weight=df["sample_weight"])

    def _create_model(self):
        return cb.CatBoostClassifier(**self.model_params)

    def _predict_model(self, df: pd.DataFrame, model) -> np.ndarray:
        return model.predict(df[self.model_features])
    
    def _check_model_is_fitted(self, model):
        return model.is_fitted()
