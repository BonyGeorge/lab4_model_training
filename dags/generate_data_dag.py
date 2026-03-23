from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts"))
from generate_data import generate_data

default_args = {"owner": "airflow"}

with DAG(
    dag_id="generate_data_only",
    default_args=default_args,
    description="Generate dataset only",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    generate_task = PythonOperator(
        task_id="generate_data",
        python_callable=generate_data,
        op_kwargs={"output_path": "data/breast_cancer.csv"},
    )