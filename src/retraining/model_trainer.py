"""
Automatic Model Training Pipeline

Handles automatic model retraining with validation, A/B testing, and deployment.
"""

import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import threading
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

# Импорт our Модель classes
from ..models.baseline import (
    RandomRecommender, PopularityRecommender, WeightedPopularityRecommender
)
from ..models.collaborative_filtering import ALSRecommender
from ..models.content_based import ContentBasedRecommender
from ..models.hybrid import HybridRecommender

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Automatic model training pipeline with validation and A/B testing.
    
    Features:
    - Automated data preparation
    - Model training with hyperparameter optimization
    - Validation and performance evaluation
    - A/B testing framework
    - Model versioning and artifact management
    - Integration with MLflow
    """
    
    def __init__(
        self,
        data_path: str = "data/",
        models_path: str = "models/",
        experiments_path: str = "experiments/",
        mlflow_tracking_uri: str = "http://localhost:5000"
    ):
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.experiments_path = Path(experiments_path)
        
        # Создание directories
        for path in [self.data_path, self.models_path, self.experiments_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # MLflow Настройка
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Training Состояние
        self.is_training = False
        self.current_models: Dict[str, Any] = {}
        self.training_history: List[Dict[str, Any]] = []
        
        # A/B Тестирование
        self.ab_test_active = False
        self.ab_test_config: Optional[Dict[str, Any]] = None
        
        logger.info("ModelTrainer initialized")
    
    def trigger_retraining(
        self,
        trigger_info: Dict[str, Any],
        models_to_retrain: List[str] = None,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Trigger automatic model retraining.
        
        Args:
            trigger_info: Information about what triggered retraining
            models_to_retrain: List of model names to retrain (None = all)
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training results and status
        """
        if self.is_training:
            logger.warning("Training already in progress, skipping trigger")
            return {'status': 'skipped', 'reason': 'training_in_progress'}
        
        logger.info(f"Starting retraining triggered by: {trigger_info['reason']}")
        
        training_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Запуск MLflow Эксперимент
            experiment_name = "automatic_retraining"
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                else:
                    experiment_id = experiment.experiment_id
            except Exception as e:
                logger.warning(f"MLflow experiment setup failed: {e}")
                experiment_id = None
            
            with mlflow.start_run(experiment_id=experiment_id, run_name=training_id):
                # Логирование trigger information
                mlflow.log_params({
                    'trigger_reason': trigger_info['reason'],
                    'trigger_metric': trigger_info.get('metric_name', 'unknown'),
                    'trigger_value': trigger_info.get('metric_value', 'unknown')
                })
                
                # Загрузка and prepare data
                train_data, val_data = self._prepare_training_data(validation_split)
                
                if train_data is None:
                    return {'status': 'failed', 'reason': 'data_preparation_failed'}
                
                # Determine models to Обучение
                if models_to_retrain is None:
                    models_to_retrain = ['popularity', 'als', 'hybrid']
                
                # Обучение models
                training_results = self._train_models(
                    models_to_retrain, train_data, val_data, training_id
                )
                
                # Оценка new models
                evaluation_results = self._evaluate_models(training_results, val_data)
                
                # Decide on Развертывание
                deployment_decision = self._make_deployment_decision(
                    evaluation_results, trigger_info
                )
                
                # Логирование results
                mlflow.log_metrics(evaluation_results.get('summary_metrics', {}))
                mlflow.log_params(deployment_decision)
                
                result = {
                    'status': 'completed',
                    'training_id': training_id,
                    'trigger_info': trigger_info,
                    'models_trained': models_to_retrain,
                    'training_results': training_results,
                    'evaluation_results': evaluation_results,
                    'deployment_decision': deployment_decision,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Сохранение training history
                self.training_history.append(result)
                self._save_training_history()
                
                logger.info(f"Retraining completed: {training_id}")
                return result
                
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'training_id': training_id,
                'timestamp': datetime.now().isoformat()
            }
        finally:
            self.is_training = False
    
    def _prepare_training_data(self, validation_split: float) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Prepare training and validation data."""
        try:
            # Try to Загрузка real data
            interactions_file = self.data_path / "interactions.parquet"
            items_file = self.data_path / "items.parquet"
            
            if interactions_file.exists() and items_file.exists():
                interactions_df = pd.read_parquet(interactions_file)
                items_df = pd.read_parquet(items_file)
                
                logger.info(f"Loaded real data: {len(interactions_df)} interactions, {len(items_df)} items")
            else:
                # Генерация synthetic data for training
                logger.info("Real data not found, generating synthetic training data")
                interactions_df, items_df = self._generate_training_data()
            
            # Разделение data temporally for realistic validation
            interactions_df = interactions_df.sort_values('timestamp' if 'timestamp' in interactions_df.columns else interactions_df.index)
            split_idx = int(len(interactions_df) * (1 - validation_split))
            
            train_interactions = interactions_df.iloc[:split_idx]
            val_interactions = interactions_df.iloc[split_idx:]
            
            # Создание Пользователь-Товар matrices
            train_matrix = self._create_user_item_matrix(train_interactions)
            val_matrix = self._create_user_item_matrix(val_interactions)
            
            train_data = {
                'interactions': train_interactions,
                'items': items_df,
                'user_item_matrix': train_matrix,
                'num_users': train_matrix.shape[0],
                'num_items': train_matrix.shape[1]
            }
            
            val_data = {
                'interactions': val_interactions,
                'items': items_df,
                'user_item_matrix': val_matrix,
                'ground_truth': self._prepare_ground_truth(val_interactions)
            }
            
            logger.info(f"Training data prepared: {len(train_interactions)} train, {len(val_interactions)} val")
            return train_data, val_data
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    def _generate_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic training data with more realistic patterns."""
        np.random.seed(42)
        
        # Параметры
        num_users = 1000
        num_items = 500
        num_interactions = 50000
        
        # Генерация users
        user_ids = np.arange(num_users)
        
        # Генерация items with categories
        categories = ['Electronics', 'Books', 'Clothing', 'Home', 'Sports']
        items_data = []
        
        for item_id in range(num_items):
            items_data.append({
                'item_id': item_id,
                'category': np.random.choice(categories),
                'price': np.random.uniform(10, 500),
                'popularity_score': np.random.exponential(1)
            })
        
        items_df = pd.DataFrame(items_data)
        
        # Генерация interactions with realistic patterns
        interactions_data = []
        
        for _ in range(num_interactions):
            user_id = np.random.choice(user_ids)
            
            # Users have preferences for certain categories
            user_category_preference = np.random.choice(categories)
            
            # Higher Вероятность for preferred Категория
            if np.random.random() < 0.6:
                item_candidates = items_df[items_df['category'] == user_category_preference]['item_id'].values
            else:
                item_candidates = items_df['item_id'].values
            
            if len(item_candidates) > 0:
                item_id = np.random.choice(item_candidates)
                
                # Событие types with different probabilities
                event_type = np.random.choice(
                    ['view', 'addtocart', 'transaction'],
                    p=[0.7, 0.2, 0.1]
                )
                
                interactions_data.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'event_type': event_type,
                    'timestamp': datetime.now() - timedelta(
                        days=np.random.uniform(0, 30)
                    ),
                    'rating': np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
                })
        
        interactions_df = pd.DataFrame(interactions_data)
        
        return interactions_df, items_df
    
    def _create_user_item_matrix(self, interactions_df: pd.DataFrame) -> csr_matrix:
        """Create sparse user-item interaction matrix."""
        # Map Пользователь and Товар IDs to indices
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['item_id'].unique()
        
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        
        # Создание Матрица
        num_users = len(unique_users)
        num_items = len(unique_items)
        
        # Weight events differently
        event_weights = {'view': 1.0, 'addtocart': 2.0, 'transaction': 3.0}
        
        row_indices = []
        col_indices = []
        data = []
        
        for _, interaction in interactions_df.iterrows():
            user_idx = user_to_idx[interaction['user_id']]
            item_idx = item_to_idx[interaction['item_id']]
            weight = event_weights.get(interaction.get('event_type', 'view'), 1.0)
            
            row_indices.append(user_idx)
            col_indices.append(item_idx)
            data.append(weight)
        
        matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(num_users, num_items)
        )
        
        return matrix
    
    def _prepare_ground_truth(self, val_interactions: pd.DataFrame) -> Dict[int, List[int]]:
        """Prepare ground truth for validation."""
        ground_truth = {}
        
        for user_id, user_data in val_interactions.groupby('user_id'):
            # Only consider positive interactions (Рейтинг >= 4 or Транзакция)
            positive_items = user_data[
                (user_data.get('rating', 5) >= 4) | 
                (user_data.get('event_type', 'view') == 'transaction')
            ]['item_id'].tolist()
            
            if positive_items:
                ground_truth[user_id] = positive_items
        
        return ground_truth
    
    def _train_models(
        self,
        model_names: List[str],
        train_data: Dict[str, Any],
        val_data: Dict[str, Any],
        training_id: str
    ) -> Dict[str, Any]:
        """Train specified models."""
        self.is_training = True
        training_results = {}
        
        try:
            for model_name in model_names:
                logger.info(f"Training model: {model_name}")
                start_time = time.time()
                
                try:
                    model, metrics = self._train_single_model(
                        model_name, train_data, val_data
                    )
                    
                    training_time = time.time() - start_time
                    
                    # Сохранение Модель
                    model_path = self.models_path / f"{model_name}_{training_id}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    
                    # Логирование to MLflow
                    mlflow.log_artifact(str(model_path), f"models/{model_name}")
                    mlflow.log_metrics({
                        f"{model_name}_training_time": training_time,
                        **{f"{model_name}_{k}": v for k, v in metrics.items()}
                    })
                    
                    training_results[model_name] = {
                        'model': model,
                        'model_path': str(model_path),
                        'metrics': metrics,
                        'training_time': training_time,
                        'status': 'success'
                    }
                    
                    logger.info(f"Model {model_name} trained successfully in {training_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error training model {model_name}: {e}")
                    training_results[model_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
        
        finally:
            self.is_training = False
        
        return training_results
    
    def _train_single_model(
        self,
        model_name: str,
        train_data: Dict[str, Any],
        val_data: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, float]]:
        """Train a single model and return it with validation metrics."""
        user_item_matrix = train_data['user_item_matrix']
        
        if model_name == 'popularity':
            model = PopularityRecommender()
            model.fit(user_item_matrix)
            
        elif model_name == 'weighted_popularity':
            model = WeightedPopularityRecommender()
            model.fit(user_item_matrix)
            
        elif model_name == 'als':
            model = ALSRecommender(factors=50, regularization=0.1, iterations=20)
            model.fit(user_item_matrix)
            
        elif model_name == 'content':
            model = ContentBasedRecommender()
            # For Контентная-based, we need Товар features
            item_features = self._create_item_features(train_data['items'])
            model.fit(user_item_matrix, item_features)
            
        elif model_name == 'hybrid':
            # Обучение Компонент models first
            base_models = {}
            
            # Популярность Модель
            pop_model = PopularityRecommender()
            pop_model.fit(user_item_matrix)
            base_models['popularity'] = pop_model
            
            # ALS Модель
            als_model = ALSRecommender(factors=50, regularization=0.1, iterations=20)
            als_model.fit(user_item_matrix)
            base_models['collaborative'] = als_model
            
            # Гибридная Модель
            model = HybridRecommender(
                models=base_models,
                weights={'collaborative': 0.7, 'popularity': 0.3}
            )
            model.fit(user_item_matrix)
            
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Оценка on validation set
        metrics = self._evaluate_single_model(model, val_data)
        
        return model, metrics
    
    def _create_item_features(self, items_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Create item features for content-based filtering."""
        item_features = {}
        
        for _, item in items_df.iterrows():
            features = {
                'category': item.get('category', 'unknown'),
                'price_range': 'low' if item.get('price', 0) < 50 else 'medium' if item.get('price', 0) < 200 else 'high',
                'popularity': item.get('popularity_score', 0)
            }
            item_features[item['item_id']] = features
        
        return item_features
    
    def _evaluate_single_model(self, model: Any, val_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single model on validation data."""
        ground_truth = val_data['ground_truth']
        metrics = {}
        
        if not ground_truth:
            return {'precision_at_10': 0.0, 'recall_at_10': 0.0, 'ndcg_at_10': 0.0}
        
        precisions = []
        recalls = []
        ndcgs = []
        
        for user_id, true_items in ground_truth.items():
            try:
                # Get recommendations
                recommendations = model.recommend_for_user(
                    user_idx=user_id, n_recommendations=10
                )
                
                if isinstance(recommendations, list) and recommendations:
                    if isinstance(recommendations[0], tuple):
                        # Форматирование: [(item_id, Оценка), ...]
                        pred_items = [item_id for item_id, _ in recommendations]
                    else:
                        # Форматирование: [item_id, ...]
                        pred_items = recommendations
                else:
                    pred_items = []
                
                # Расчет metrics
                if pred_items and true_items:
                    # Точность@10
                    precision = len(set(pred_items) & set(true_items)) / len(pred_items)
                    precisions.append(precision)
                    
                    # Полнота@10
                    recall = len(set(pred_items) & set(true_items)) / len(true_items)
                    recalls.append(recall)
                    
                    # NDCG@10 (simplified)
                    ndcg = self._calculate_ndcg(pred_items, true_items, k=10)
                    ndcgs.append(ndcg)
                
            except Exception as e:
                logger.warning(f"Error evaluating user {user_id}: {e}")
                continue
        
        # Aggregate metrics
        metrics['precision_at_10'] = np.mean(precisions) if precisions else 0.0
        metrics['recall_at_10'] = np.mean(recalls) if recalls else 0.0
        metrics['ndcg_at_10'] = np.mean(ndcgs) if ndcgs else 0.0
        
        return metrics
    
    def _calculate_ndcg(self, predictions: List[int], ground_truth: List[int], k: int = 10) -> float:
        """Calculate NDCG@k (simplified version)."""
        # DCG
        dcg = 0.0
        for i, item in enumerate(predictions[:k]):
            if item in ground_truth:
                dcg += 1.0 / np.log2(i + 2)
        
        # IDCG
        idcg = 0.0
        for i in range(min(len(ground_truth), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _evaluate_models(self, training_results: Dict[str, Any], val_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all trained models."""
        evaluation_results = {
            'model_comparisons': {},
            'summary_metrics': {},
            'best_model': None
        }
        
        best_score = 0.0
        best_model = None
        
        for model_name, result in training_results.items():
            if result['status'] == 'success':
                metrics = result['metrics']
                evaluation_results['model_comparisons'][model_name] = metrics
                
                # Расчет Компоновщик Оценка
                composite_score = (
                    metrics.get('precision_at_10', 0) * 0.4 +
                    metrics.get('recall_at_10', 0) * 0.3 +
                    metrics.get('ndcg_at_10', 0) * 0.3
                )
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_model = model_name
        
        evaluation_results['best_model'] = best_model
        evaluation_results['summary_metrics'] = {
            'best_composite_score': best_score,
            'models_evaluated': len([r for r in training_results.values() if r['status'] == 'success'])
        }
        
        return evaluation_results
    
    def _make_deployment_decision(
        self,
        evaluation_results: Dict[str, Any],
        trigger_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decide whether to deploy new models."""
        decision = {
            'deploy': False,
            'reason': '',
            'recommended_model': None,
            'ab_test_recommended': False
        }
        
        best_model = evaluation_results.get('best_model')
        if not best_model:
            decision['reason'] = 'No successful model training'
            return decision
        
        best_score = evaluation_results['summary_metrics']['best_composite_score']
        
        # Decision criteria
        if best_score > 0.3:  # Good performance threshold
            decision['deploy'] = True
            decision['recommended_model'] = best_model
            decision['reason'] = f'Model {best_model} shows good performance (score: {best_score:.3f})'
            
            # Рекомендация A/B Тестирование for significant changes
            if best_score > 0.4:
                decision['ab_test_recommended'] = True
        else:
            decision['reason'] = f'Best model score too low: {best_score:.3f}'
        
        return decision
    
    def _save_training_history(self):
        """Save training history to disk."""
        try:
            history_file = self.experiments_path / "training_history.json"
            
            # Преобразование to serializable Форматирование
            serializable_history = []
            for entry in self.training_history:
                serializable_entry = entry.copy()
                # Удаление non-serializable Модель objects
                if 'training_results' in serializable_entry:
                    for model_name, result in serializable_entry['training_results'].items():
                        if 'model' in result:
                            del result['model']
                
                serializable_history.append(serializable_entry)
            
            import json
            with open(history_file, 'w') as f:
                json.dump(serializable_history, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving training history: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            'is_training': self.is_training,
            'ab_test_active': self.ab_test_active,
            'ab_test_config': self.ab_test_config,
            'training_history_length': len(self.training_history),
            'last_training': self.training_history[-1] if self.training_history else None
        } 