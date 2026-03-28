from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

# Add src to path so DAGs can import ml_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.data import generate_data, load_data
from ml_pipeline.model import train_model, evaluate_model

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="ml_pipeline",
    default_args=default_args,
    description="Pipeline: generate data -> train model",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    generate_task = PythonOperator(
        task_id="generate_data",
        python_callable=generate_data,
        op_kwargs={"output_path": "data/iris.csv"},
    )

    def train_model_wrapper(data_path: str, model_path: str):
        df = load_data(data_path)
        return train_model(df, model_path)

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model_wrapper,
        op_kwargs={
            "data_path": "data/iris.csv",
            "model_path": "models/breast_cancer_model.pkl",
        },
    )

    generate_task >> train_task