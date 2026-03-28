import os
import joblib
import json
import datetime
import pytz
import boto3

from zoneinfo import ZoneInfo

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_model(df: pd.DataFrame, model_path: str = "models/breast_cancer_model.pkl") -> float:
    """Train a logistic regression classifier and save it."""
    dtObject_cst = datetime.datetime.now(pytz.timezone('US/Central')).strftime("%Y%m%d_%H%M%S")
    metadata_path = "models/metadata.json"
    
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[ml_pipeline.model] Model accuracy: {acc:.4f}")
    
    versioned_model_path = f"models/{dtObject_cst}_breast_cancer_model.pkl"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, versioned_model_path)
    
    new_entry = {
        "model_version": dtObject_cst,
        "dataset": "breast_cancer",
        "model_type": "logistic_regression",
        "status": "stagging",
        "accuracy": round(acc, 2)
    }
    
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(new_entry)
    
    with open(metadata_path, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"[ml_pipeline.model] Saved model to {versioned_model_path}")

    return acc

def evaluate_model(df: pd.DataFrame, model_path: str = "models/breast_cancer_model.pkl"):
    """Evaluating the classifier"""
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = joblib.load(model_path)
    print(f"[ml_pipeline.model] model has been loaded from {model_path}")
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"[ml_pipeline.model] Model accuracy: {acc:.4f}")
    
    with open ("models/metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f, indent=4)
    
    return acc  
    


def promote_model(model_version: str, s3_uri: str, base_path: str = "models", threshold: float = 0.94):
    """Promote a model to S3 if it meets the accuracy threshold."""
    
    if not s3_uri.startswith("s3://"):
        raise ValueError("s3_uri must start with 's3://'")
    
    path = s3_uri[5:]
    parts = path.split("/", 1)
    bucket_name = parts[0]
    prefix_base = parts[1] if len(parts) > 1 else ""
    if prefix_base and not prefix_base.endswith("/"):
        prefix_base += "/"
    
    s3_prefix = f"{prefix_base}{model_version}/"
    
    metadata_path = os.path.join(base_path, "metadata.json")
    metrics_path = os.path.join(base_path, "metrics.json")
    model_path = os.path.join(base_path, f"{model_version}_breast_cancer_model.pkl")
    
    with open(metadata_path, "r") as f:
        data = json.load(f)
    
    model_entry = next((m for m in data if m["model_version"] == model_version), None)
    if not model_entry:
        raise ValueError(f"Model version {model_version} not found in metadata")
    
    accuracy = model_entry["accuracy"]
    if accuracy < threshold:
        raise ValueError(f"Model {model_version} failed accuracy threshold: {accuracy}")
    
    s3 = boto3.client("s3")
    s3.upload_file(model_path, bucket_name, s3_prefix + "model.pkl")
    s3.upload_file(metrics_path, bucket_name, s3_prefix + "metrics.json")
    s3.upload_file(metadata_path, bucket_name, s3_prefix + "metadata.json")
    

    for m in data:
        if m["model_version"] == model_version:
            m["status"] = "production"
        else:
            m["status"] = "archived"
    
    with open(metadata_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Model {model_version} promoted to S3 at {s3_prefix}")
    
 
    