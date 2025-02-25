import logging
import asyncio
from functools import wraps
import os
from pathlib import Path
import pickle
from datetime import datetime, timedelta
import subprocess

import polars as pl
from dotenv import load_dotenv
import telegram
from tqdm import tqdm
import click

from src.data.index import CandlesAssetData
from src.db.clickhouse import get_connector, read_data, with_ssh_tunnel
from scripts.fetch_bar_data import get_available_tickers
from strategies.sma import catch_up, get_featurizers


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    handlers=[logging.StreamHandler()])


def async_decorator(f):
    """Decorator to allow calling an async function like a sync function"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        ret = asyncio.run(f(*args, **kwargs))
        return ret
    return wrapper


@async_decorator
async def send_message(entry_data):
    msg = "\n".join(" | ".join(e) for e in entry_data)

    bot = telegram.Bot(os.environ["TELEGRAM_TOKEN"])
    async with bot:
        await bot.send_message(text=msg, chat_id=os.environ["MY_TG_CHAT_ID"])


def get_path(save_dir: Path, uid: str) -> Path:
    return save_dir / f"{uid}.pickle"


def save_asset(asset: CandlesAssetData, save_dir: Path, uid: str):
    with open(get_path(save_dir, uid), "wb") as f:
        pickle.dump(asset, f)


def load_asset(save_dir: Path, uid: str) -> CandlesAssetData:
    with open(get_path(save_dir, uid), "rb") as f:
        asset = pickle.load(f)

    return asset


def format_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def catch_up_asset(save_dir, uid, featurizers):
    asset = CandlesAssetData.from_clickhouse(
        uid=uid,
        date_from='2024-12-01',
        date_to=format_datetime(datetime.now()),
        name=uid,
        filter_weekend=True,
        check_columns=True,
    )

    if len(asset) >= 1000:
        catch_up(asset, featurizers=featurizers)

    save_asset(asset, save_dir, uid)


def main(save_dir: Path):
    available_uids = get_available_tickers(token=os.environ["TOKEN"]).filter(pl.col("short_enabled_flag"))["uid"].to_list()
    print(len(available_uids))
    featurizers = get_featurizers()

    connector = get_connector()
    mapping = read_data("select ticker, uid from default.instruments", connector)
    mapping = mapping.assign(uid=lambda x: x.uid.astype(str)).set_index("uid")["ticker"].to_dict()

    entry_data = []
    for uid in tqdm(available_uids):
        try:
            if not get_path(save_dir, uid).exists():
                catch_up_asset(save_dir, uid, featurizers)

            asset = load_asset(save_dir, uid)
            if len(asset) < 1000:
                continue

            if asset.sorted_dates[-1] < datetime.now() - timedelta(days=1):
                continue

            new_asset = CandlesAssetData.from_clickhouse(
                uid=uid,
                date_from=format_datetime(asset.sorted_dates[-1]),
                date_to=format_datetime(datetime.now()),
                name=uid,
                filter_weekend=True,
                check_columns=True,
            )
            if len(new_asset) == 0:
                continue

            logger.info(f'Update ticker: {mapping.get(uid, "Not found")} with {len(new_asset)} new dates')
            asset = asset.merge(new_asset)
            for dt in tqdm(new_asset.sorted_dates):
                for featurizer in featurizers:
                    featurizer.get_features_iter(asset, dt)

                data = asset.get(dt)

                if (data is not None) and data["is_entry"]:
                    entry_data.append((data[asset.dt_col].isoformat(), uid, mapping.get(uid), "long"))

                if (data is not None) and data["is_exit"]:
                    entry_data.append((data[asset.dt_col].isoformat(), uid, mapping.get(uid), "short"))

            save_asset(asset, save_dir, uid)
        except Exception as e:
            print(uid)
            print(e)

    if len(entry_data) > 0:
        send_message(entry_data)


@click.command()
@click.option("-d", "--dotenv-path", type=click.Path(exists=True, dir_okay=False), default=".env")
@click.option("-s", "--save-dir", type=click.Path(file_okay=False), default="data/cache/sma_cb/")
def scan(dotenv_path, save_dir):
    load_dotenv(dotenv_path)
    with_ssh_tunnel(main)(Path(save_dir))


if __name__ == "__main__":
    scan()
