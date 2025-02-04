from typing import Sequence

import catboost
import polars as pl
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
import holidays
from sklearn.linear_model import LinearRegression


def fill_gaps(df: pl.DataFrame, ts_col: str = "ts", period: str = "1m") -> pl.DataFrame:
    return (
        pl.datetime_range(df[ts_col].min(), df[ts_col].max(), interval=period, eager=True).alias(ts_col).to_frame()
        .join(df, on=ts_col, how="left")
        .select(pl.all().forward_fill())
    )


def add_hour_shift_features(df: pl.DataFrame, col: str, shift: int) -> pl.DataFrame:
    df.select("ts", "close").with_columns(
        pl.col("close") - pl.duration(hours=shift)
    )


def add_shift_features(df: pl.DataFrame, col: str, shifts: Sequence[int]) -> pl.DataFrame:
    return pl.concat([
        df, 
        (
            df
            .select(*[
                pl.col(col).shift(s).alias(f"{col}_s_{s}")
                for s in shifts
            ])
        )
    ], how="horizontal")


class PricePredictor:
    """
    Predict prices for previous data
    """
    def fit(self, df: pl.DataFrame):
        return self

    def predict(self, n: int) -> pl.DataFrame:
        raise NotImplementedError()


class CatBoostPricePredictor(PricePredictor):
    def __init__(
        self, 
        interval: str = "1h",
        avg_interval: str | None = None,
        random_state: int = 42, 
        shifts: tuple = (12, 24, 36, 48, 60, 72),
        cb_params: dict | None = None
    ):
        self.random_state = random_state

        self.cb_params = {} if cb_params is None else cb_params
        self.model = catboost.CatBoostRegressor(random_state=random_state, **self.cb_params)

        self.holidays = pl.from_pandas(make_holidays_df(list(range(2021, 2030)), country="RU")[["ds"]])
        self.avg_interval = avg_interval

        self.interval = interval
        self.shifts = shifts

        self.history_df = None
        self.feature_columns = None

    def prepare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        df = fill_gaps(df.select("ts", "close"), period=self.interval)

        if self.avg_interval is not None:
            df = df.with_columns(pl.col("close").ewm_mean_by("ts", half_life=self.avg_interval).alias("close_avg"))

        df = add_shift_features((
            df
            .with_columns(
                pl.col("close" if self.avg_interval is None else "close_avg").alias("close_features")
            )
        ), col="close_features", shifts=self.shifts).drop("close_features")
        
        if "close_avg" in df.columns:
            df = df.drop("close_avg")

        # add holidays
        df = (
            df
            .join(self.holidays, left_on=pl.col("ts").dt.date(), right_on=pl.col("ds").dt.date(), how="left")
            .with_columns(
                pl.when(pl.col("ds").is_null()).then(0).otherwise(1).alias("is_holiday")
            )
            .drop("ds")
        )

        return df.sort("ts")

    def fit(self, df: pl.DataFrame):
        self.history_df = (
            df.select("ts")
            .join(self.prepare_features(df), how="left", on="ts")
        )
        self.feature_columns = [c for c in self.history_df.columns if c not in ("ts", "close", "volume")]

        self.model.fit(
            self.history_df.to_pandas()[self.feature_columns],
            self.history_df["close"].to_numpy()
        )

        return self

    def _get_duration(self, n_units: int):
        if "h" in self.interval:
            base_units = int("".join([v for v in self.interval if v in "1234567890"]))
            units = base_units * n_units
            return pl.duration(hours=units)
        elif "m" in self.interval:
            base_units = int("".join([v for v in self.interval if v in "1234567890"]))
            units = base_units * n_units
            return pl.duration(minutes=units)
        else:
            raise TypeError(f"Cannot parse interval {self.interval}, or ont implemented")

    def predict(self, n: int) -> pl.DataFrame:
        assert self.history_df is not None

        max_future_dt = min(self.shifts)
        duration = self._get_duration(max_future_dt)

        predicted_intervals = 0
        predict = []
        prev_data = self.history_df.select("ts", "close")
        while predicted_intervals < n:
            future_ts = pl.datetime_range(
                prev_data["ts"].max(),  # type: ignore
                prev_data["ts"].max() + duration,  # type: ignore
                interval=self.interval, 
                eager=True
            ).alias("ts").to_frame()[1:] # type: ignore

            future_df = pl.concat([
                prev_data.select("ts", "close"),
                future_ts.with_columns(pl.lit(None).alias("close"))
            ])

            future_feature_df = self.prepare_features(future_df).tail(len(future_ts))

            predict_ = (
                future_feature_df
                .select(
                    "ts", 
                    pl.lit(self.model.predict(future_feature_df.select(*self.feature_columns).to_pandas())).alias("close")
                )
            )
            prev_data = pl.concat([prev_data, predict_])
            predicted_intervals += predict_.shape[0]
            predict.append(predict_)

        return pl.concat(predict)[:n]


class ProphetPricePredictor(PricePredictor):
    def __init__(self, interval: str = "1h"):
        holidays = make_holidays_df(list(range(2021, 2030)), country="RU")
        self.interval = interval
        self.model = Prophet()

    def fit(self, df: pl.DataFrame):
        self.model.fit(df.select(
            pl.col("ts").dt.replace_time_zone(None).alias("ds"),
            pl.col("close").alias("y")
        ).to_pandas())
        
        return self
    
    def predict(self, n: int) -> pl.DataFrame:
        future_df = self.model.make_future_dataframe(periods=n, freq=self.interval, include_history=False)
        return (
            pl.from_pandas(self.model.predict(future_df)[["ds", "yhat"]])
            .select(
                pl.col("ds").dt.replace_time_zone("UTC").alias("ts"),
                pl.col("yhat").alias("close"),
            )
        )
