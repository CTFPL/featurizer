from datetime import datetime
from functools import partial

import talib

from src.features.models.cb import CatboostClassifierFeaturizer
from src.features.lag import LagTicksFeaturizer
from src.features.func import FuncFeaturizer
from src.features.talib import TALibFeaturizer
from src.data.index import CandlesAssetData


def get_featurizers():
    all_featurizers = []

    sma_high_period = 60
    sma_high_featurizer = TALibFeaturizer(
        f"sma_{sma_high_period}", 
        func=lambda kwargs: talib.SMA(real=kwargs["close"], timeperiod=sma_high_period), 
        add_to_asset=True,
    )
    sma_high_lag_featurizer = LagTicksFeaturizer(
        f"sma_{sma_high_period}_lag", 
        feature=f"sma_{sma_high_period}",
        lag=1,
        add_to_asset=True,
    )
    all_featurizers.extend([
        sma_high_featurizer,
        sma_high_lag_featurizer,
    ])

    sma_low_period = 15
    sma_low_featurizer = TALibFeaturizer(
        f"sma_{sma_low_period}", 
        func=lambda kwargs: talib.SMA(real=kwargs["close"], timeperiod=sma_low_period), 
        add_to_asset=True,
    )
    sma_low_lag_featurizer = LagTicksFeaturizer(
        f"sma_{sma_low_period}_lag", 
        feature=f"sma_{sma_low_period}",
        lag=1,
        add_to_asset=True,
    )
    all_featurizers.extend([
        sma_low_featurizer,
        sma_low_lag_featurizer,
    ])

    sma_periods = [15, 30, 60, 120]
    for period in sma_periods:
        all_featurizers.append(TALibFeaturizer(
            f"volatility_{period}",
            func=lambda kwargs, p=period: talib.NATR(high=kwargs["high"], low=kwargs["low"], close=kwargs["close"], timeperiod=p) ,
            add_to_asset=True,
        ))

    # features for catboost
    for period in sma_periods:
        all_featurizers.append(
            TALibFeaturizer(
                f"sma_{period}", 
                # https://gist.github.com/gisbi-kim/2e5648225cc118fc72ac933ef63c2d64
                func=lambda kwargs, p=period: talib.SMA(real=kwargs["close"], timeperiod=p), 
                add_to_asset=True,
            )
        )

    for period in sma_periods:
        all_featurizers.append(LagTicksFeaturizer(
            f"sma_{period}_lag_5", 
            feature=f"sma_{period}", 
            lag=5, 
            add_to_asset=True,
        ))

    lags = (1, 5, 15)
    for lag in lags:
        all_featurizers.append(LagTicksFeaturizer(
            f"close_lag_{lag}", 
            feature=f"close", 
            lag=lag, 
            add_to_asset=True,
        ))

    for lag in lags:
        all_featurizers.append(LagTicksFeaturizer(
            f"volume_lag_{lag}", 
            feature="volume", 
            lag=lag, 
            add_to_asset=True,
        ))
    
    def min_max_norm(asset: CandlesAssetData, dt: datetime, window_size: int = 60, col: str = "close"):
        data = asset.get_last_n_ticks(dt, window_size)

        if len(data) < window_size:
            return None
        
        min_ = min(d[col] for d in data if d[col] is not None)
        max_ = max(d[col] for d in data if d[col] is not None)
        curr = data[-1][col]

        if min_ is None:
            return None
        
        if max_ is None:
            return None
        
        if curr is None:
            return None
        
        return (curr - min_) / (max_ - min_)
    
    def norm_(asset: CandlesAssetData, dt: datetime, lag: int = 60, col: str = "close"):
        data_lag = asset.get_lag_ticks(dt, lag)
        data_curr = asset.get(dt)

        if (data_lag is None) or (col not in data_lag) or (data_lag[col] is None):
            return None
        
        if (data_curr is None) or (col not in data_curr) or (data_curr[col] is None):
            return None
        
        return 2 * (data_curr[col] - data_lag[col]) / (data_curr[col] + data_curr[col])
    
    norm_window_sizes = [15, 45, 180]
    min_max_cols = ["close", f"sma_{sma_low_period}", f"sma_{sma_high_period}", "volume"]
    for col in min_max_cols:
        for window_size in norm_window_sizes:
            all_featurizers.append(
                FuncFeaturizer(
                    f"norm_{col}_{window_size}",
                    partial(norm_, lag=window_size, col=col),
                    add_to_asset=True
                )
            )
            all_featurizers.append(
                FuncFeaturizer(
                    f"min_max_{col}_{window_size}",
                    partial(min_max_norm, window_size=window_size, col=col),
                    add_to_asset=True
                )
            )

    def diff_cols(asset, dt, col1, col2):
        data = asset.get(dt)

        if data is None:
            return None

        if data[col1] is None:
            return None

        if data[col2] is None:
            return None

        return data[col1] - data[col2]

    
    for period in sma_periods:
        col1 = f"sma_{period}"
        col2 = f"sma_{period}_lag_5"
        
        all_featurizers.append(FuncFeaturizer(
            f"diff_{col1}_{col2}", 
            partial(diff_cols, col1=col1, col2=col2), 
            add_to_asset=True
        ))

    for col in ("volume", "close"):
        col1 = f"{col}_lag_5"
        col2 = f"{col}_lag_1"
        
        all_featurizers.append(FuncFeaturizer(
            f"diff_{col1}_{col2}", 
            partial(diff_cols, col1=col1, col2=col2), 
            add_to_asset=True,
        ))

    model_features = [
        # *[f"sma_{period}" for period in sma_periods],
        # *[f"sma_{period}_lag_5" for period in sma_periods],
        *[f"volatility_{period}" for period in sma_periods],
        *[f"min_max_{col}_{window_size}" for col in min_max_cols for window_size in norm_window_sizes],
        *[f"norm_{col}_{window_size}" for col in min_max_cols for window_size in norm_window_sizes],
        # 'close_lag_1',
        # 'volume_lag_1',
        # 'close_lag_5',
        # 'volume_lag_5',
        # 'close_lag_15',
        # 'volume_lag_15',
        # 'diff_sma_15_sma_15_lag_5',
        # 'diff_sma_30_sma_30_lag_5',
        # 'diff_sma_60_sma_60_lag_5',
        # 'diff_sma_120_sma_120_lag_5',
        # 'diff_volume_lag_5_volume_lag_1',
        # 'diff_close_lag_5_close_lag_1',
    ]

    cb_clf = CatboostClassifierFeaturizer(
        name="predict_up",
        model_features=model_features, 
        target_col="close", 
        model_params={
            "iterations": 100,
            "random_state": 42,
            "verbose": 0
        }, 
        n_lags=5000, 
        lag_target=4 * 60, 
        add_to_asset=False, 
        predict_up=True, 
        use_time_weights=True,
        min_up_perc=0.01,
        quantile=0.95,
        threshold=0.5,
    )

    cb_clf_down = CatboostClassifierFeaturizer(
        name="predict_down",
        model_features=model_features, 
        target_col="close", 
        model_params={
            "iterations": 100,
            "random_state": 42,
            "verbose": 0
        }, 
        n_lags=5000, 
        lag_target=4 * 60, 
        add_to_asset=False, 
        predict_up=False, 
        use_time_weights=True,
        min_up_perc=0.01,
        quantile=0.95,
        threshold=0.5,
    )

    def is_entry(asset: CandlesAssetData, dt: datetime):
        data = asset.get(dt)

        if data is None: 
            return False

        for col in (
            sma_high_featurizer.get_name(),
            sma_high_lag_featurizer.get_name(),
            sma_low_featurizer.get_name(),
            sma_low_lag_featurizer.get_name(),
        ):
            if data[col] is None:
                return False
            
        is_entry = (
            (data[sma_high_lag_featurizer.get_name()] > data[sma_low_lag_featurizer.get_name()])
            and (data[sma_high_featurizer.get_name()] <= data[sma_low_featurizer.get_name()])
        )

        cb_clf_predict = False
        if is_entry:
            _, cb_clf_predict = cb_clf.get_features_iter(asset, dt)
            # print(dt, type(cb_clf_predict), cb_clf_predict, is_entry, cb_clf_predict and is_entry)

        if cb_clf_predict is None:
            return False

        return cb_clf_predict and is_entry


    def is_exit(asset: CandlesAssetData, dt: datetime):
        data = asset.get(dt)

        if data is None: 
            return False

        for col in (
            sma_high_featurizer.get_name(),
            sma_high_lag_featurizer.get_name(),
            sma_low_featurizer.get_name(),
            sma_low_lag_featurizer.get_name(),
        ):
            if data[col] is None:
                return False
            
        is_exit_ = (
            (data[sma_high_lag_featurizer.get_name()] < data[sma_low_lag_featurizer.get_name()])
            and (data[sma_high_featurizer.get_name()] >= data[sma_low_featurizer.get_name()])
        )

        cb_clf_predict = False
        if is_exit_:
            _, cb_clf_predict = cb_clf_down.get_features_iter(asset, dt)
            # print(dt, type(cb_clf_predict), cb_clf_predict, is_entry, cb_clf_predict and is_entry)

        if cb_clf_predict is None:
            return False

        return cb_clf_predict and is_exit_


    sma_is_entry_featurizer = FuncFeaturizer(
        name="is_entry",
        func=is_entry,
        add_to_asset=True,
    )
    all_featurizers.append(sma_is_entry_featurizer)
    sma_is_exit_featurizer = FuncFeaturizer(
        name="is_exit",
        func=is_exit,
        add_to_asset=True,
    )
    all_featurizers.append(sma_is_exit_featurizer)

    return all_featurizers


def catch_up(asset: CandlesAssetData, featurizers):
    for featurizer in featurizers:
        featurizer.get_features_batch(asset)


def preprocess_asset(asset: CandlesAssetData, dt: datetime, featurizers):
    for featurizer in featurizers:
        featurizer.get_features_iter(asset, dt)

    data = asset.get(dt)

    return data


def get_entrypoint():
    pass