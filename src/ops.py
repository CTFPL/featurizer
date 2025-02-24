import os

from tinkoff.invest import Client
from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX, INVEST_GRPC_API


def get_lot_count(instrument_id, client):
    instruments = client.instruments.find_instrument(query=instrument_id).instruments

    assert len(instruments) == 1, f"Got {len(instruments)}, expected only 1"
    
    return instruments[0].lot
    
    
def parse_quotation(q):
    return q.units + q.nano / 1e9


def find_instrument_id(ticker):
    matching_instruments = []
    with Client(os.environ["TOKEN"], target=INVEST_GRPC_API) as client:
        for instrument in client.instruments.find_instrument(query=ticker).instruments:
            if not instrument.api_trade_available_flag:
                continue

            if instrument.isin == '':
                continue

            if instrument.isin.startswith("RU"):
                matching_instruments.append(instrument)

    if len(matching_instruments) > 1:
        print(matching_instruments)
    elif len(matching_instruments) == 1:
        return matching_instruments[0].uid

    return False