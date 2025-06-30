"""
MLflow utilities for experiment tracking and model management
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os
import json
from datetime import datetime


class MLflowTracker:
    """MLflow experiment tracker for recommendation models"""
    
    def __init__(self, experiment_name: str, tracking_uri: str = "http://127.0.0.1:5000"):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Создание or get Эксперимент
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
            
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        """Start a new MLflow run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.run = mlflow.start_run(run_name=run_name)
        return self.run.info.run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, model_name: str, **kwargs):
        """Log model to MLflow"""
        if hasattr(model, 'fit'):  # sklearn-like model
            mlflow.sklearn.log_model(model, model_name, **kwargs)
        else:
            mlflow.log_artifact(model, model_name)
    
    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log artifact to MLflow"""
        mlflow.log_artifact(artifact_path, artifact_name)
    
    def log_dataframe(self, df: pd.DataFrame, name: str):
        """Log dataframe as CSV artifact"""
        csv_path = f"temp_{name}.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path, f"data/{name}.csv")
        os.remove(csv_path)
    
    def log_dict(self, data: Dict[str, Any], name: str):
        """Log dictionary as JSON artifact"""
        json_path = f"temp_{name}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        mlflow.log_artifact(json_path, f"data/{name}.json")
        os.remove(json_path)
    
    def end_run(self):
        """End current MLflow run"""
        mlflow.end_run()


def log_recommendation_metrics(
    tracker: MLflowTracker,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_values: list = [5, 10, 20],
    step: Optional[int] = None
):
    """
    Log recommendation-specific metrics
    
    Args:
        tracker: MLflowTracker instance
        y_true: True relevance scores
        y_pred: Predicted relevance scores
        k_values: List of k values for top-k metrics
        step: Step number for metric logging
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    metrics = {}
    
    # Basic Классификация metrics
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    metrics['precision'] = precision_score(y_true_binary, y_pred_binary, average='weighted')
    metrics['recall'] = recall_score(y_true_binary, y_pred_binary, average='weighted')
    metrics['f1_score'] = f1_score(y_true_binary, y_pred_binary, average='weighted')
    
    # Top-k metrics (simplified implementation)
    for k in k_values:
        # This is a simplified Версия - you'd implement proper ranking metrics
        top_k_indices = np.argsort(y_pred)[-k:]
        top_k_precision = np.mean(y_true[top_k_indices] > 0)
        metrics[f'precision_at_{k}'] = top_k_precision
    
    tracker.log_metrics(metrics, step=step)


def create_experiment_config(
    model_type: str,
    data_params: Dict[str, Any],
    model_params: Dict[str, Any],
    training_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create experiment configuration dictionary
    
    Args:
        model_type: Type of recommendation model
        data_params: Data preprocessing parameters
        model_params: Model hyperparameters
        training_params: Training configuration
        
    Returns:
        Complete experiment configuration
    """
    config = {
        'model_type': model_type,
        'timestamp': datetime.now().isoformat(),
        'data_params': data_params,
        'model_params': model_params,
        'training_params': training_params
    }
    
    return config


# Example usage functions
def example_collaborative_filtering_experiment():
    """Example of logging collaborative filtering experiment"""
    
    # Инициализация tracker
    tracker = MLflowTracker("collaborative_filtering")
    
    # Запуск run
    run_id = tracker.start_run("als_baseline")
    
    # Логирование Параметры
    params = {
        'model_type': 'ALS',
        'factors': 50,
        'regularization': 0.01,
        'iterations': 10,
        'alpha': 1.0
    }
    tracker.log_params(params)
    
    # Simulate training and Логирование metrics
    metrics = {
        'rmse': 0.85,
        'precision_at_5': 0.12,
        'precision_at_10': 0.08,
        'recall_at_5': 0.15,
        'recall_at_10': 0.22
    }
    tracker.log_metrics(metrics)
    
    # End run
    tracker.end_run()
    
    print(f"Experiment logged with run_id: {run_id}")


if __name__ == "__main__":
    # Run example
    example_collaborative_filtering_experiment() 