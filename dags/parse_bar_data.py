from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


project_dir = "/Users/aleksandryusov/code/algotrading"
python_exec_path = "/Users/aleksandryusov/.pyenv/versions/algotrading/bin/python"
default_tickers = ",".join([
    '278d9ccc-4dde-484e-bf79-49ce8f733470',
    'b993e814-9986-4434-ae88-b086066714a0',
    '21423d2d-9009-4d37-9325-883b368d13ae',
    '4d813ab1-8bc9-4670-89ea-12bfbab6017d',
    '8e2b0325-0292-4654-8a18-4f63ed3b0e09',
    'ca845f68-6c43-44bc-b584-330d2a1e5eb7',
    'efdb54d3-2f92-44da-b7a3-8849e96039f6',
    'b83ab195-dcd2-4d44-b9bf-27fa294f19a0',
    '509edd0c-129c-4ee2-934d-7f6246126da1',
    '53b67587-96eb-4b41-8e0c-d2e3c0bdd234'
])

with DAG(
    'parse_bar_data',
    default_args={
        'depends_on_past': False,
        'email': ['airflow@example.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(seconds=5),
    },
    description='Parse data from tinkoff API and save it to dir',
    schedule_interval="*/10 10-19 * * *",
    # schedule_interval=timedelta(minutes=10),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['data', 'candles', 'tinkoff'],
) as dag:
    t3 = BashOperator(
        task_id='templated',
        depends_on_past=False,
        bash_command=" ".join([
            python_exec_path,
            f"{project_dir}/scripts/fetch_bar_data.py",
            f"-d {project_dir}/.env",
            "fetch-tickers-from-list",
            "-i CANDLE_INTERVAL_1_MIN",
            f"-s {project_dir}/data/candles_1_min/",
            '-f "{{data_interval_start}}"',
            '-e "{{data_interval_end}}"',
            f'-t "{default_tickers}"'
        ]),
        env={
            "PYTHONPATH": project_dir,
        }
    )

t3  # type:ignore
