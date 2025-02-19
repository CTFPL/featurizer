from datetime import datetime, timedelta

from tinkoff.invest import Client, CandleInterval
from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX
from tinkoff.invest.utils import now
from tqdm.auto import tqdm
import polars as pl
from ratelimit import limits, RateLimitException
from backoff import on_exception, expo

from src.settings import CandleParserSettings
from src.ops import get_lot_count, parse_quotation


def parse_candle_to_record(candle):
    return {
        "open": parse_quotation(candle.open),
        "high": parse_quotation(candle.high),
        "low": parse_quotation(candle.low),
        "close": parse_quotation(candle.close),
        "volume": candle.volume,
        "ts": candle.time
    }


@on_exception(expo, RateLimitException, max_tries=CandleParserSettings.max_tries)
@limits(calls=CandleParserSettings.calls, period=CandleParserSettings.period)
def fetch_candles_data_last(
    instrument_id,
    token: str,
    n_days: int = 365,
    interval = CandleInterval.CANDLE_INTERVAL_HOUR,
) -> pl.DataFrame:
    data = []
    with Client(token, target=INVEST_GRPC_API_SANDBOX) as client:
        lot_count = get_lot_count(instrument_id, client=client)
        for candle in tqdm(client.get_all_candles(
            from_=now() - timedelta(days=n_days), 
            to=now(), 
            interval=interval, 
            instrument_id=instrument_id,
        )):
            data.append(parse_candle_to_record(candle))
    
    return (
        pl.from_dicts(data)
        .with_columns(
            pl.lit(instrument_id).alias("instrument_id"),
            pl.lit(datetime.now()).alias("parse_ts"),
            pl.lit(lot_count).alias("lot")
        )
    )


@on_exception(expo, RateLimitException, max_tries=CandleParserSettings.max_tries)
@limits(calls=CandleParserSettings.calls, period=CandleParserSettings.period)
def fetch_candles_data(
    instrument_id,
    token: str,
    date_from: datetime,
    date_to: datetime,
    interval = CandleInterval.CANDLE_INTERVAL_HOUR,
) -> pl.DataFrame:
    data = []
    with Client(token, target=INVEST_GRPC_API_SANDBOX) as client:
        lot_count = get_lot_count(instrument_id, client=client)
        for candle in tqdm(client.get_all_candles(
            from_=date_from, 
            to=date_to, 
            interval=interval, 
            instrument_id=instrument_id,
        )):
            data.append(parse_candle_to_record(candle))
    
    if data:
        return (
            pl.from_dicts(data)
            .with_columns(
                pl.lit(instrument_id).alias("instrument_id"),
                pl.lit(datetime.now()).alias("parse_ts"),
                pl.lit(lot_count).alias("lot")
            )
        )

    return pl.DataFrame()
