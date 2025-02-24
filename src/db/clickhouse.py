import os

import clickhouse_connect


def get_connector():
    return clickhouse_connect.get_client(
        host=os.environ["CH_HOST"], 
        username=os.environ["CH_USERNAME"], 
        password=os.environ["CH_PASSWORD"], 
        port=int(os.environ["CH_PORT"]), 
        database=os.environ["CH_DB"],
    )
