from datetime import datetime
import os
from pathlib import Path

import click
import polars as pl
from dotenv import load_dotenv
from tinkoff.invest import InstrumentStatus, Client, CandleInterval
from tinkoff.invest.schemas import InstrumentExchangeType, Currency
from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX
from tqdm import tqdm

from scripts.fetch import fetch_candles_data, fetch_candles_data_last


def get_available_tickers(token: str) -> pl.DataFrame:
    with Client(token, target=INVEST_GRPC_API_SANDBOX) as client:
        shares = client.instruments.shares(instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE)

    print(f"Instruments before filter:", len(shares.instruments))

    return (
        pl.from_records([d.__dict__ for d in shares.instruments])
        .filter(pl.col("currency") == "rub") 
        # .filter(pl.col("trading_status") == 5)  # доступен для торгов
        .filter(pl.col("for_iis_flag"))  # доступен для ИИС
        .filter(~pl.col("for_qual_investor_flag"))  # доступен для неквал
        .filter(pl.col("liquidity_flag"))  # ликвидный
    )


def validate_candle_interval_input(interval):
    try:
        CandleInterval[interval]
        return True
    except KeyError as e:
        return False


@click.group()
@click.option("-d", "--dotenv-path", type=click.Path(exists=True, dir_okay=False), default=".env")
def cli(dotenv_path):
    print(load_dotenv(dotenv_path))


@cli.command()
@click.option("-d", "--n-days", default=365)
@click.option("-i", "--interval", default="CANDLE_INTERVAL_HOUR", help="Аналогичное название из либы")
@click.option("-s", "--save-dir", type=click.Path(file_okay=False), default="./data/candles/")
@click.option("-e", "--error-ticker-dir", type=click.Path(file_okay=False), default="./data/logs/error/")
def fetch_all(n_days, interval, save_dir, error_ticker_dir):
    """
    PYTHONPATH=. nohup python scripts/fetch_all_tickers.py -d .env fetch-all >logs/fetch_all_hour.log 2>&1 &
    PYTHONPATH=. nohup python scripts/fetch_bar_data.py -d .env fetch-all -i CANDLE_INTERVAL_1_MIN -d 90 -s data/candles_1_min/ >logs/fetch_all_minute.log 2>&1 &
    echo $! > logs/fetch_all_hour.pid
    """
    token = os.environ["SANDBOX_TOKEN"]
    save_dir = Path(save_dir)
    err_path = Path(error_ticker_dir) / f"{int(datetime.now().timestamp())}.txt"

    assert validate_candle_interval_input(interval)

    available_tickers = get_available_tickers(token=token)
    print(f"Number of available tickers: ", len(available_tickers))
    for t in tqdm(available_tickers.to_dicts()):
        try:
            fetch_candles_data_last(
                instrument_id=t["uid"], 
                token=token, 
                n_days=n_days, 
                interval=CandleInterval[interval]
            ).write_parquet(save_dir / f"{t['ticker']}.parquet")  # type:ignore
        except Exception as e:
            print(f"Failed to parse {t['ticker']}")
            with open(err_path, "a") as f:
                f.write(f"{t['ticker']}|{t['uid']}\n")


@cli.command()
@click.option("-i", "--interval", default="CANDLE_INTERVAL_HOUR", help="Аналогичное название из либы")
@click.option("-s", "--save-dir", type=click.Path(file_okay=False, exists=True), default="./data/candles/")
@click.option("-f", "--date-from", type=click.DateTime(formats=["%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"]))
@click.option("-e", "--date-to", type=click.DateTime(formats=["%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"]))
@click.option("-t", "--tickers", type=click.STRING)
def fetch_tickers_from_list(interval, save_dir, date_from, date_to, tickers):
    token = os.environ["SANDBOX_TOKEN"]
    save_dir = Path(save_dir)

    tickers = tickers.split(",")
    # date_from = datetime.fromisoformat(date_from)
    # date_to = datetime.fromisoformat(date_to)

    assert validate_candle_interval_input(interval)
    for t in tqdm(tickers):
        fetch_candles_data(
            instrument_id=t, 
            token=token, 
            date_from=date_from,
            date_to=date_to,
            interval=CandleInterval[interval]
        ).write_parquet(save_dir / f"{t['ticker']}.parquet")  # type:ignore


if __name__ == "__main__":
    cli()
