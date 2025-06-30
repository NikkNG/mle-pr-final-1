"""
Model manager for loading and managing recommendation models.
"""

import os
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Импорт рекомендательных моделей
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.baseline import RandomRecommender, PopularityRecommender, WeightedPopularityRecommender
from src.models.collaborative_filtering import ALSRecommender, create_collaborative_models
from src.models.content_based import ContentBasedRecommender, TFIDFRecommender
from src.models.hybrid import HybridRecommender
from src.data.preprocessing import DataPreprocessor, load_and_preprocess_data

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Управляет загрузкой, кэшированием и обслуживанием рекомендательных моделей.
    """
    
    def __init__(self, data_path: str = "data/", models_path: str = "models/"):
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.preprocessor: Optional[DataPreprocessor] = None
        self.user_item_matrix = None
        self.item_features = None
        self.user_encoder = None
        self.item_encoder = None
        self.item_properties = None
        
        # Создание директорий, если они не существуют
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Инициализация логирования
        logging.basicConfig(level=logging.INFO)
        
    def load_data(self) -> bool:
        """Загрузка и предобработка данных для вывода модели."""
        try:
            logger.info("Loading and preprocessing data...")
            
            # Загрузка сырых данных
            events_path = self.data_path / "raw" / "events.csv"
            item_properties_path = self.data_path / "raw" / "item_properties.csv"
            category_tree_path = self.data_path / "raw" / "category_tree.csv"
            
            if not all(p.exists() for p in [events_path, item_properties_path, category_tree_path]):
                logger.warning("Data files not found, creating synthetic data for testing")
                self._create_synthetic_data()
                return True
            
            # Загрузка и предобработка
            result = load_and_preprocess_data(
                events_path=str(events_path),
                item_properties_path=str(item_properties_path),
                category_tree_path=str(category_tree_path),
                min_user_interactions=5,
                min_item_interactions=5
            )
            
            (self.user_item_matrix, self.item_features, 
             test_matrix, self.user_encoder, self.item_encoder, 
             self.preprocessor) = result
            
            # Загрузка свойств товаров для метаданных
            self.item_properties = pd.read_csv(item_properties_path)
            
            logger.info(f"Data loaded successfully. Matrix shape: {self.user_item_matrix.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _create_synthetic_data(self):
        """Создание синтетических данных для тестирования, когда реальные данные недоступны."""
        logger.info("Creating synthetic data for testing...")
        
        # Создание синтетической матрицы пользователь-товар
        n_users, n_items = 500, 200
        density = 0.02
        
        np.random.seed(42)
        user_item_data = np.random.rand(n_users, n_items)
        user_item_data[user_item_data > density] = 0
        user_item_data[user_item_data > 0] = np.random.choice([1, 2, 3], 
                                                              size=(user_item_data > 0).sum(),
                                                              p=[0.7, 0.2, 0.1])
        
        from scipy.sparse import csr_matrix
        self.user_item_matrix = csr_matrix(user_item_data)
        
        # Создание синтетических признаков товаров
        self.item_features = np.random.rand(n_items, 50)
        
        # Создание синтетических кодировщиков
        self.user_encoder = {i: i for i in range(n_users)}
        self.item_encoder = {i: i for i in range(n_items)}
        
        # Создание синтетических свойств товаров
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
        self.item_properties = pd.DataFrame({
            'item_id': range(n_items),
            'category': np.random.choice(categories, n_items),
            'name': [f'Item_{i}' for i in range(n_items)]
        })
        
        logger.info("Synthetic data created successfully")
    
    def load_models(self) -> bool:
        """Загрузка всех доступных моделей."""
        try:
            if self.user_item_matrix is None:
                if not self.load_data():
                    return False
            
            logger.info("Loading recommendation models...")
            
            # Загрузка базовых моделей
            self._load_baseline_models()
            
            # Загрузка моделей коллаборативной фильтрации
            self._load_collaborative_models()
            
            # Загрузка контентных моделей
            self._load_content_models()
            
            # Загрузка гибридных моделей
            self._load_hybrid_models()
            
            logger.info(f"Loaded {len(self.models)} models successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _load_baseline_models(self):
        """Загрузка базовых рекомендательных моделей."""
        try:
            # Случайный рекомендатор
            random_model = RandomRecommender()
            random_model.fit(self.user_item_matrix)
            self.models['random'] = random_model
            self.model_metadata['random'] = {
                'type': 'baseline',
                'name': 'Random Recommender',
                'loaded_at': datetime.now()
            }
            
            # Рекомендатор по популярности
            popularity_model = PopularityRecommender()
            popularity_model.fit(self.user_item_matrix)
            self.models['popularity'] = popularity_model
            self.model_metadata['popularity'] = {
                'type': 'baseline',
                'name': 'Popularity Recommender',
                'loaded_at': datetime.now()
            }
            
            # Взвешенный рекомендатор по популярности
            weighted_pop_model = WeightedPopularityRecommender()
            weighted_pop_model.fit(self.user_item_matrix)
            self.models['weighted_popularity'] = weighted_pop_model
            self.model_metadata['weighted_popularity'] = {
                'type': 'baseline',
                'name': 'Weighted Popularity Recommender',
                'loaded_at': datetime.now()
            }
            
            logger.info("Baseline models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading baseline models: {e}")
    
    def _load_collaborative_models(self):
        """Загрузка моделей коллаборативной фильтрации."""
        try:
            # Модель ALS
            als_model = ALSRecommender(factors=50, regularization=0.1, iterations=20)
            als_model.fit(self.user_item_matrix)
            self.models['als'] = als_model
            self.model_metadata['als'] = {
                'type': 'collaborative',
                'name': 'ALS Collaborative Filtering',
                'loaded_at': datetime.now()
            }
            
            logger.info("Collaborative filtering models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading collaborative models: {e}")
    
    def _load_content_models(self):
        """Load content-based models."""
        try:
            if self.item_features is not None:
                # Контентная-based Модель
                content_model = ContentBasedRecommender()
                content_model.fit(self.user_item_matrix, self.item_features)
                self.models['content'] = content_model
                self.model_metadata['content'] = {
                    'type': 'content',
                    'name': 'Content-Based Recommender',
                    'loaded_at': datetime.now()
                }
            
            logger.info("Content-based models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading content models: {e}")
    
    def _load_hybrid_models(self):
        """Load hybrid models."""
        try:
            # Создание Гибридная Модель if we have multiple base models
            if len(self.models) >= 2:
                base_models = {}
                if 'als' in self.models:
                    base_models['collaborative'] = self.models['als']
                if 'content' in self.models:
                    base_models['content'] = self.models['content']
                if 'popularity' in self.models:
                    base_models['popularity'] = self.models['popularity']
                
                if len(base_models) >= 2:
                    hybrid_model = HybridRecommender(
                        models=base_models,
                        weights={'collaborative': 0.5, 'content': 0.3, 'popularity': 0.2}
                    )
                    hybrid_model.fit(self.user_item_matrix, self.item_features)
                    self.models['hybrid'] = hybrid_model
                    self.model_metadata['hybrid'] = {
                        'type': 'hybrid',
                        'name': 'Hybrid Recommender',
                        'loaded_at': datetime.now()
                    }
            
            logger.info("Hybrid models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading hybrid models: {e}")
    
    def get_recommendations(self, user_id: int, model_name: str = 'hybrid', 
                          num_recommendations: int = 10, 
                          exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """Get recommendations for a user."""
        try:
            # Use Гибридная Модель as default, fallback to available models
            if model_name not in self.models:
                available_models = ['hybrid', 'als', 'content', 'popularity', 'random']
                for fallback_model in available_models:
                    if fallback_model in self.models:
                        model_name = fallback_model
                        logger.warning(f"Requested model not available, using {model_name}")
                        break
                else:
                    raise ValueError("No models available")
            
            model = self.models[model_name]
            
            # Проверка if Пользователь exists in our data
            if user_id >= self.user_item_matrix.shape[0]:
                logger.warning(f"User {user_id} not in training data, using popularity model")
                model = self.models.get('popularity', self.models.get('random'))
            
            # Get recommendations
            recommendations = model.recommend_for_user(
                user_idx=user_id,
                n_recommendations=num_recommendations,
                filter_already_liked=exclude_seen
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            # Fallback to Случайная recommendations
            if 'random' in self.models:
                return self.models['random'].recommend_for_user(
                    user_idx=0, n_recommendations=num_recommendations
                )
            return []
    
    def get_similar_items(self, item_id: int, model_name: str = 'als', 
                         num_items: int = 10) -> List[Tuple[int, float]]:
        """Get similar items for a given item."""
        try:
            # Use Коллаборативная Модель for Товар Схожесть
            available_models = ['als', 'content', 'hybrid']
            for model_name in available_models:
                if model_name in self.models:
                    model = self.models[model_name]
                    if hasattr(model, 'get_similar_items'):
                        return model.get_similar_items(item_id, num_items)
            
            # Fallback: return Случайная items
            if item_id < self.user_item_matrix.shape[1]:
                random_items = np.random.choice(
                    self.user_item_matrix.shape[1], 
                    size=min(num_items, self.user_item_matrix.shape[1]), 
                    replace=False
                )
                return [(int(item), np.random.random()) for item in random_items if item != item_id]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting similar items: {e}")
            return []
    
    def get_popular_items(self, num_items: int = 10, 
                         category: Optional[str] = None) -> List[Tuple[int, float]]:
        """Get popular items, optionally filtered by category."""
        try:
            if 'popularity' in self.models:
                recommendations = self.models['popularity'].recommend_for_user(
                    user_idx=0, n_recommendations=num_items
                )
                
                # Фильтрация by Категория if specified
                if category and self.item_properties is not None:
                    category_items = self.item_properties[
                        self.item_properties['category'] == category
                    ]['item_id'].values
                    
                    filtered_recs = []
                    for item_id, score in recommendations:
                        if item_id in category_items:
                            filtered_recs.append((item_id, score))
                        if len(filtered_recs) >= num_items:
                            break
                    
                    return filtered_recs
                
                return recommendations
            
            # Fallback to Случайная items
            random_items = np.random.choice(
                self.user_item_matrix.shape[1], 
                size=min(num_items, self.user_item_matrix.shape[1]), 
                replace=False
            )
            return [(int(item), np.random.random()) for item in random_items]
            
        except Exception as e:
            logger.error(f"Error getting popular items: {e}")
            return []
    
    def get_item_info(self, item_id: int) -> Dict[str, Any]:
        """Get item information."""
        try:
            if self.item_properties is not None:
                item_info = self.item_properties[
                    self.item_properties['item_id'] == item_id
                ]
                if not item_info.empty:
                    return item_info.iloc[0].to_dict()
            
            # Fallback Информация
            return {
                'item_id': item_id,
                'name': f'Item_{item_id}',
                'category': 'Unknown'
            }
            
        except Exception as e:
            logger.error(f"Error getting item info: {e}")
            return {'item_id': item_id, 'name': f'Item_{item_id}', 'category': 'Unknown'}
    
    def get_models_status(self) -> Dict[str, Any]:
        """Get status of all loaded models."""
        return {
            'loaded_models': list(self.models.keys()),
            'model_metadata': self.model_metadata,
            'data_loaded': self.user_item_matrix is not None,
            'matrix_shape': self.user_item_matrix.shape if self.user_item_matrix is not None else None
        }
    
    def is_healthy(self) -> bool:
        """Check if the model manager is in a healthy state."""
        return (
            len(self.models) > 0 and 
            self.user_item_matrix is not None and
            any(model_name in self.models for model_name in ['hybrid', 'als', 'popularity'])
        ) 