"""
Collaborative Filtering models for recommendation system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.sparse import csr_matrix
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
import warnings
warnings.filterwarnings('ignore')


class ALSRecommender:
    """
    Alternating Least Squares (ALS) Collaborative Filtering Recommender
    """
    
    def __init__(self, 
                 factors: int = 50,
                 regularization: float = 0.01,
                 iterations: int = 15,
                 alpha: float = 1.0,
                 random_state: int = 42):
        """
        Initialize ALS model
        
        Args:
            factors: Number of latent factors
            regularization: Regularization parameter
            iterations: Number of training iterations
            alpha: Confidence parameter for implicit feedback
            random_state: Random seed for reproducibility
        """
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        
        # Инициализация Модель
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            alpha=alpha,
            random_state=random_state,
            use_gpu=False  # Set to True if GPU available
        )
        
        self.is_fitted = False
        self.user_item_matrix = None
        self.item_user_matrix = None
        
    def fit(self, user_item_matrix: csr_matrix) -> 'ALSRecommender':
        """
        Fit ALS model
        
        Args:
            user_item_matrix: User-item interaction matrix
            
        Returns:
            Self for method chaining
        """
        print(f"🤖 Обучение ALS модели...")
        print(f"   Факторы: {self.factors}")
        print(f"   Регуляризация: {self.regularization}")
        print(f"   Итерации: {self.iterations}")
        print(f"   Альфа: {self.alpha}")
        
        # Сохранение matrices
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = user_item_matrix.T.tocsr()
        
        # Обучение Модель
        self.model.fit(self.item_user_matrix)
        self.is_fitted = True
        
        print("✅ Модель ALS обучена")
        return self
    
    def recommend_for_user(self, 
                          user_idx: int, 
                          n_recommendations: int = 10,
                          filter_already_liked: bool = True) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user
        
        Args:
            user_idx: User index
            n_recommendations: Number of recommendations
            filter_already_liked: Whether to filter items user already interacted with
            
        Returns:
            List of (item_idx, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get Пользователь Вектор
        if user_idx >= self.user_item_matrix.shape[0]:
            raise ValueError(f"User index {user_idx} out of range")
        
        # Get recommendations
        item_ids, scores = self.model.recommend(
            userid=user_idx,
            user_items=self.user_item_matrix[user_idx],
            N=n_recommendations,
            filter_already_liked_items=filter_already_liked
        )
        
        return list(zip(item_ids, scores))
    
    def recommend_batch(self, 
                       user_indices: List[int],
                       n_recommendations: int = 10,
                       filter_already_liked: bool = True) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations for multiple users
        
        Args:
            user_indices: List of user indices
            n_recommendations: Number of recommendations per user
            filter_already_liked: Whether to filter items user already interacted with
            
        Returns:
            Dictionary mapping user_idx to list of (item_idx, score) tuples
        """
        recommendations = {}
        
        for user_idx in user_indices:
            try:
                recommendations[user_idx] = self.recommend_for_user(
                    user_idx, n_recommendations, filter_already_liked
                )
            except Exception as e:
                print(f"⚠️ Ошибка для пользователя {user_idx}: {e}")
                recommendations[user_idx] = []
        
        return recommendations
    
    def get_similar_items(self, 
                         item_idx: int, 
                         n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Find similar items
        
        Args:
            item_idx: Item index
            n_similar: Number of similar items
            
        Returns:
            List of (item_idx, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar items")
        
        if item_idx >= self.item_user_matrix.shape[0]:
            raise ValueError(f"Item index {item_idx} out of range")
        
        # Get similar items
        item_ids, scores = self.model.similar_items(
            itemid=item_idx,
            N=n_similar + 1  # +1 because it includes the item itself
        )
        
        # Удаление the Товар itself from results
        similar_items = []
        for item_id, score in zip(item_ids, scores):
            if item_id != item_idx:
                similar_items.append((item_id, score))
        
        return similar_items[:n_similar]
    
    def get_user_factors(self, user_idx: int) -> np.ndarray:
        """Get user latent factors"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting factors")
        
        return self.model.user_factors[user_idx]
    
    def get_item_factors(self, item_idx: int) -> np.ndarray:
        """Get item latent factors"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting factors")
        
        return self.model.item_factors[item_idx]
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """
        Predict rating for user-item pair
        
        Args:
            user_idx: User index
            item_idx: Item index
            
        Returns:
            Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        user_factors = self.get_user_factors(user_idx)
        item_factors = self.get_item_factors(item_idx)
        
        return float(np.dot(user_factors, item_factors))
    
    def get_model_params(self) -> Dict:
        """Get model parameters"""
        return {
            'factors': self.factors,
            'regularization': self.regularization,
            'iterations': self.iterations,
            'alpha': self.alpha,
            'random_state': self.random_state
        }


class BPRRecommender:
    """
    Bayesian Personalized Ranking (BPR) Collaborative Filtering Recommender
    """
    
    def __init__(self,
                 factors: int = 50,
                 learning_rate: float = 0.01,
                 regularization: float = 0.01,
                 iterations: int = 100,
                 random_state: int = 42):
        """
        Initialize BPR model
        
        Args:
            factors: Number of latent factors
            learning_rate: Learning rate
            regularization: Regularization parameter
            iterations: Number of training iterations
            random_state: Random seed
        """
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        
        # Инициализация Модель
        self.model = BayesianPersonalizedRanking(
            factors=factors,
            learning_rate=learning_rate,
            regularization=regularization,
            iterations=iterations,
            random_state=random_state,
            use_gpu=False
        )
        
        self.is_fitted = False
        self.user_item_matrix = None
        self.item_user_matrix = None
    
    def fit(self, user_item_matrix: csr_matrix) -> 'BPRRecommender':
        """Fit BPR model"""
        print(f"🤖 Обучение BPR модели...")
        print(f"   Факторы: {self.factors}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Регуляризация: {self.regularization}")
        print(f"   Итерации: {self.iterations}")
        
        # Сохранение matrices
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = user_item_matrix.T.tocsr()
        
        # Обучение Модель
        self.model.fit(self.item_user_matrix)
        self.is_fitted = True
        
        print("✅ Модель BPR обучена")
        return self
    
    def recommend_for_user(self, 
                          user_idx: int, 
                          n_recommendations: int = 10,
                          filter_already_liked: bool = True) -> List[Tuple[int, float]]:
        """Generate recommendations for a user"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_idx >= self.user_item_matrix.shape[0]:
            raise ValueError(f"User index {user_idx} out of range")
        
        # Get recommendations
        item_ids, scores = self.model.recommend(
            userid=user_idx,
            user_items=self.user_item_matrix[user_idx],
            N=n_recommendations,
            filter_already_liked_items=filter_already_liked
        )
        
        return list(zip(item_ids, scores))
    
    def get_model_params(self) -> Dict:
        """Get model parameters"""
        return {
            'factors': self.factors,
            'learning_rate': self.learning_rate,
            'regularization': self.regularization,
            'iterations': self.iterations,
            'random_state': self.random_state
        }


class CollaborativeFilteringEvaluator:
    """
    Evaluator for collaborative filtering models
    """
    
    def __init__(self, train_matrix: csr_matrix, test_matrix: csr_matrix):
        """
        Initialize evaluator
        
        Args:
            train_matrix: Training user-item matrix
            test_matrix: Test user-item matrix
        """
        self.train_matrix = train_matrix
        self.test_matrix = test_matrix
    
    def evaluate_model(self, 
                      model: Union[ALSRecommender, BPRRecommender],
                      n_recommendations: int = 10,
                      k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate collaborative filtering model
        
        Args:
            model: Trained model
            n_recommendations: Number of recommendations to generate
            k_values: List of k values for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        from .metrics import RecommendationMetrics
        
        print(f"📊 Оценка модели {model.__class__.__name__}...")
        
        # Get Тестирование users (users who have interactions in Тестирование set)
        test_users = []
        test_items_per_user = {}
        
        for user_idx in range(self.test_matrix.shape[0]):
            user_test_items = self.test_matrix[user_idx].nonzero()[1]
            if len(user_test_items) > 0:
                test_users.append(user_idx)
                test_items_per_user[user_idx] = user_test_items.tolist()
        
        print(f"📊 Тестирование на {len(test_users)} пользователях...")
        
        # Генерация recommendations
        recommendations = model.recommend_batch(
            test_users, 
            n_recommendations=max(k_values),
            filter_already_liked=True
        )
        
        # Prepare data for evaluation
        y_true = []
        y_pred = []
        
        for user_idx in test_users:
            if user_idx in recommendations and user_idx in test_items_per_user:
                # Get recommended items (only Товар indices)
                rec_items = [item_idx for item_idx, score in recommendations[user_idx]]
                
                y_true.append(test_items_per_user[user_idx])
                y_pred.append(rec_items)
        
        # Расчет metrics
        metrics_calculator = RecommendationMetrics()
        metrics = metrics_calculator.calculate_all_metrics(
            y_true=y_true,
            y_pred=y_pred,
            k_values=k_values,
            total_items=self.train_matrix.shape[1]
        )
        
        return metrics


def create_collaborative_models() -> Dict[str, Union[ALSRecommender, BPRRecommender]]:
    """
    Create collaborative filtering models with different configurations
    
    Returns:
        Dictionary of models
    """
    models = {
        'als_default': ALSRecommender(),
        'als_high_factors': ALSRecommender(factors=100),
        'als_high_reg': ALSRecommender(regularization=0.1),
        'als_more_iter': ALSRecommender(iterations=30),
        'als_high_alpha': ALSRecommender(alpha=10.0),
        'bpr_default': BPRRecommender(),
        'bpr_high_factors': BPRRecommender(factors=100),
        'bpr_low_lr': BPRRecommender(learning_rate=0.001)
    }
    
    return models


def hyperparameter_search(train_matrix: csr_matrix,
                         test_matrix: csr_matrix,
                         param_grid: Dict[str, List],
                         model_type: str = 'als') -> Tuple[Dict, float]:
    """
    Perform hyperparameter search
    
    Args:
        train_matrix: Training matrix
        test_matrix: Test matrix
        param_grid: Parameter grid for search
        model_type: Type of model ('als' or 'bpr')
        
    Returns:
        Tuple of (best_params, best_score)
    """
    from itertools import product
    
    print(f"🔍 Поиск гиперпараметров для {model_type.upper()}...")
    
    # Генерация parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    best_score = 0
    best_params = {}
    
    evaluator = CollaborativeFilteringEvaluator(train_matrix, test_matrix)
    
    for i, param_combo in enumerate(param_combinations):
        params = dict(zip(param_names, param_combo))
        
        print(f"🧪 Тестирование параметров {i+1}/{len(param_combinations)}: {params}")
        
        # Создание and Обучение Модель
        if model_type == 'als':
            model = ALSRecommender(**params)
        elif model_type == 'bpr':
            model = BPRRecommender(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        try:
            model.fit(train_matrix)
            
            # Оценка Модель
            metrics = evaluator.evaluate_model(model, k_values=[10])
            score = metrics.get('ndcg_at_10', 0)
            
            print(f"   NDCG@10: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = params
                print(f"   🎉 Новый лучший результат!")
        
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            continue
    
    print(f"\n🏆 Лучшие параметры: {best_params}")
    print(f"🏆 Лучший NDCG@10: {best_score:.4f}")
    
    return best_params, best_score


# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    
    # Создание synthetic Пользователь-Товар Матрица
    n_users, n_items = 1000, 500
    density = 0.01
    
    # Генерация Случайная interactions
    n_interactions = int(n_users * n_items * density)
    user_indices = np.random.randint(0, n_users, n_interactions)
    item_indices = np.random.randint(0, n_items, n_interactions)
    ratings = np.random.exponential(1.0, n_interactions)
    
    # Создание sparse Матрица
    user_item_matrix = csr_matrix(
        (ratings, (user_indices, item_indices)),
        shape=(n_users, n_items)
    )
    
    print(f"📊 Создана синтетическая матрица: {user_item_matrix.shape}")
    print(f"📊 Плотность: {user_item_matrix.nnz / (n_users * n_items):.4f}")
    
    # Тестирование ALS Модель
    als_model = ALSRecommender(factors=20, iterations=5)
    als_model.fit(user_item_matrix)
    
    # Get recommendations for first Пользователь
    recommendations = als_model.recommend_for_user(0, n_recommendations=5)
    print(f"\n🎯 Рекомендации для пользователя 0:")
    for item_idx, score in recommendations:
        print(f"   Товар {item_idx}: {score:.4f}")
    
    # Get similar items
    similar_items = als_model.get_similar_items(0, n_similar=3)
    print(f"\n🔗 Похожие товары на товар 0:")
    for item_idx, score in similar_items:
        print(f"   Товар {item_idx}: {score:.4f}")
    
    print("\n✅ Тест коллаборативной фильтрации завершен")
