from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


project_dir = "/Users/aleksandryusov/code/algotrading"
python_exec_path = "/Users/aleksandryusov/.pyenv/versions/algotrading/bin/python"

with DAG(
    'sma_notify',
    default_args={
        'depends_on_past': False,
        'email': ['airflow@example.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(seconds=5),
    },
    description='Run sma strategy and notify tg',
    schedule_interval="*/5 10-19 * * *",
    # schedule_interval=timedelta(minutes=10),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['data', 'strategy', 'tg'],
) as dag:
    t3 = BashOperator(
        task_id='templated',
        depends_on_past=False,
        bash_command=" ".join([
            python_exec_path,
            f"{project_dir}/scripts/scan.py",
            f"-d {project_dir}/.env",
            f"-s {project_dir}/data/cache/sma_cb/",
        ]),
        env={
            "PYTHONPATH": project_dir,
        }
    )

t3  # type:ignore
