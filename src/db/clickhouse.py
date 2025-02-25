from functools import wraps
import os

import pandas as pd
import clickhouse_connect
from sshtunnel import SSHTunnelForwarder, open_tunnel, HandlerSSHTunnelForwarderError


def get_connector():
    return clickhouse_connect.get_client(
        host=os.environ["CH_HOST"], 
        username=os.environ["CH_USERNAME"], 
        password=os.environ["CH_PASSWORD"], 
        port=int(os.environ["CH_PORT"]), 
        database=os.environ["CH_DB"],
    )


def read_data(query: str, connector) -> pd.DataFrame:
    try:
        with open_tunnel(
            os.environ["SSH_BIND_IP"],
            ssh_username=os.environ["SSH_USERNAME"],
            ssh_password=os.environ["SSH_PKEY_PASSWORD"],
            ssh_pkey=os.environ["SSH_PKEY"],
            remote_bind_address=(os.environ["SSH_REMOTE_BIND_ADDRESS_IP"], int(os.environ["SSH_REMOTE_BIND_ADDRESS_HOST"])),
            local_bind_address=(os.environ["SSH_LOCAL_BIND_ADDRESS_IP"], int(os.environ["SSH_LOCAL_BIND_ADDRESS_HOST"])),
        ) as server:
            data = connector.query_df(query)
    except HandlerSSHTunnelForwarderError as e:
        data = connector.query_df(query)

    return data


def open_ssh_tunnel():
    server = SSHTunnelForwarder(
        os.environ["SSH_BIND_IP"],
        ssh_username=os.environ["SSH_USERNAME"],
        ssh_password=os.environ["SSH_PKEY_PASSWORD"],
        ssh_pkey=os.environ["SSH_PKEY"],
        remote_bind_address=(os.environ["SSH_REMOTE_BIND_ADDRESS_IP"], int(os.environ["SSH_REMOTE_BIND_ADDRESS_HOST"])),
        local_bind_address=(os.environ["SSH_LOCAL_BIND_ADDRESS_IP"], int(os.environ["SSH_LOCAL_BIND_ADDRESS_HOST"])),
    )

    server.start()

    return server


def with_ssh_tunnel(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            with open_tunnel(
                os.environ["SSH_BIND_IP"],
                ssh_username=os.environ["SSH_USERNAME"],
                # ssh_password=os.environ["SSH_PKEY_PASSWORD"],
                ssh_pkey=os.environ["SSH_PKEY"],
                remote_bind_address=(os.environ["SSH_REMOTE_BIND_ADDRESS_IP"], int(os.environ["SSH_REMOTE_BIND_ADDRESS_HOST"])),
                local_bind_address=(os.environ["SSH_LOCAL_BIND_ADDRESS_IP"], int(os.environ["SSH_LOCAL_BIND_ADDRESS_HOST"])),
            ) as server:
                return f(*args, **kwargs)
        except HandlerSSHTunnelForwarderError as e:
            return f(*args, **kwargs)
    return wrapper
