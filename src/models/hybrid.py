"""
Гибридные рекомендательные модели, объединяющие коллаборативную и контентную фильтрацию
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

from .collaborative_filtering import ALSRecommender, BPRRecommender
from .content_based import ContentBasedRecommender, TFIDFRecommender
from .baseline import PopularityRecommender


class HybridRecommender:
    """
    Гибридный рекомендатор, объединяющий несколько рекомендательных подходов
    """
    
    def __init__(self, 
                 models: Dict[str, Any],
                 weights: Optional[Dict[str, float]] = None,
                 combination_method: str = 'weighted_average',
                 fallback_strategy: str = 'popularity'):
        """
        Инициализация гибридного рекомендатора
        
        Args:
            models: Словарь рекомендательных моделей
            weights: Веса для каждой модели (если None, равные веса)
            combination_method: Способ объединения предсказаний ('weighted_average', 'rank_fusion')
            fallback_strategy: Стратегия для холодного старта ('popularity', 'random')
        """
        self.models = models
        self.combination_method = combination_method
        self.fallback_strategy = fallback_strategy
        
        # Установка весов
        if weights is None:
            self.weights = {name: 1.0 / len(models) for name in models.keys()}
        else:
            # Нормализация весов
            total_weight = sum(weights.values())
            self.weights = {name: weight / total_weight for name, weight in weights.items()}
        
        # Резервная модель
        self.fallback_model = None
        self.is_fitted = False
        
        # Отслеживание производительности моделей
        self.model_performance = {}
    
    def fit(self, 
            user_item_matrix: csr_matrix,
            item_features: Optional[Dict] = None,
            item_properties_df: Optional[pd.DataFrame] = None,
            metadata: Optional[Dict] = None) -> 'HybridRecommender':
        """
        Обучение всех моделей в гибридной системе
        
        Args:
            user_item_matrix: Матрица взаимодействий пользователь-товар
            item_features: Словарь признаков товаров
            item_properties_df: DataFrame свойств товаров
            metadata: Словарь метаданных
            
        Returns:
            Self для цепочки методов
        """
        print("🤖 Обучение гибридной модели...")
        print(f"📊 Модели: {list(self.models.keys())}")
        print(f"⚖️ Веса: {self.weights}")
        
        # Обучение каждой модели
        for model_name, model in self.models.items():
            print(f"\n🔧 Обучение модели '{model_name}'...")
            
            try:
                if isinstance(model, (ALSRecommender, BPRRecommender)):
                    # Модели коллаборативной фильтрации
                    model.fit(user_item_matrix)
                    
                elif isinstance(model, ContentBasedRecommender):
                    # Контентные модели
                    if item_features and metadata:
                        model.fit(user_item_matrix, item_features, metadata)
                    else:
                        print(f"⚠️ Пропуск {model_name}: отсутствуют item_features или metadata")
                        continue
                        
                elif isinstance(model, TFIDFRecommender):
                    # TF-IDF модели
                    if item_properties_df is not None and metadata:
                        model.fit(item_properties_df, metadata)
                    else:
                        print(f"⚠️ Пропуск {model_name}: отсутствует item_properties_df или metadata")
                        continue
                        
                elif hasattr(model, 'fit'):
                    # Другие модели с методом fit
                    model.fit(user_item_matrix)
                    
                else:
                    print(f"⚠️ Неизвестный тип модели: {type(model)}")
                    continue
                
                print(f"✅ Модель '{model_name}' обучена")
                
            except Exception as e:
                print(f"❌ Ошибка при обучении модели '{model_name}': {e}")
                # Удаление неудачной модели
                if model_name in self.weights:
                    del self.weights[model_name]
        
        # Перенормализация весов после возможного удаления моделей
        if self.weights:
            total_weight = sum(self.weights.values())
            self.weights = {name: weight / total_weight for name, weight in self.weights.items()}
        
        # Инициализация резервной модели
        if self.fallback_strategy == 'popularity':
            self.fallback_model = PopularityRecommender()
            self.fallback_model.fit(user_item_matrix)
        
        self.is_fitted = True
        print(f"\n✅ Гибридная модель обучена с {len(self.weights)} активными моделями")
        
        return self
    
    def recommend_for_user(self, 
                          user_idx: int,
                          n_recommendations: int = 10,
                          filter_already_liked: bool = True,
                          user_item_matrix: Optional[csr_matrix] = None) -> List[Tuple[int, float]]:
        """
        Генерация рекомендаций для пользователя с использованием гибридного подхода
        
        Args:
            user_idx: Индекс пользователя
            n_recommendations: Количество рекомендаций
            filter_already_liked: Фильтровать ли товары, с которыми пользователь уже взаимодействовал
            user_item_matrix: Матрица пользователь-товар (для фильтрации)
            
        Returns:
            Список кортежей (item_idx, score)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Получение рекомендаций от каждой модели
        model_recommendations = {}
        
        for model_name, model in self.models.items():
            if model_name not in self.weights:
                continue
                
            try:
                if hasattr(model, 'recommend_for_user'):
                    # Получение большего количества рекомендаций для лучшего слияния
                    # Проверка, принимает ли модель параметр user_item_matrix
                    import inspect
                    sig = inspect.signature(model.recommend_for_user)
                    
                    if 'user_item_matrix' in sig.parameters:
                        recs = model.recommend_for_user(
                            user_idx, 
                            n_recommendations=n_recommendations * 2,
                            filter_already_liked=filter_already_liked,
                            user_item_matrix=user_item_matrix
                        )
                    else:
                        recs = model.recommend_for_user(
                            user_idx, 
                            n_recommendations=n_recommendations * 2,
                            filter_already_liked=filter_already_liked
                        )
                    model_recommendations[model_name] = recs
                else:
                    print(f"⚠️ Модель '{model_name}' не поддерживает recommend_for_user")
                    
            except Exception as e:
                print(f"⚠️ Ошибка в модели '{model_name}': {e}")
                continue
        
        # Обработка случая, когда ни одна модель не смогла сгенерировать рекомендации
        if not model_recommendations:
            if self.fallback_model:
                print("🔄 Использование fallback модели")
                return self.fallback_model.recommend_for_user(
                    user_idx, n_recommendations, filter_already_liked, user_item_matrix
                )
            else:
                return []
        
        # Объединение рекомендаций
        if self.combination_method == 'weighted_average':
            final_recommendations = self._weighted_average_combination(
                model_recommendations, n_recommendations
            )
        elif self.combination_method == 'rank_fusion':
            final_recommendations = self._rank_fusion_combination(
                model_recommendations, n_recommendations
            )
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        return final_recommendations
    
    def _weighted_average_combination(self, 
                                    model_recommendations: Dict[str, List[Tuple[int, float]]],
                                    n_recommendations: int) -> List[Tuple[int, float]]:
        """Combine recommendations using weighted average of scores"""
        
        # Collect all Товар scores
        item_scores = {}
        item_weights = {}
        
        for model_name, recommendations in model_recommendations.items():
            weight = self.weights.get(model_name, 0)
            
            for item_idx, score in recommendations:
                if item_idx not in item_scores:
                    item_scores[item_idx] = 0
                    item_weights[item_idx] = 0
                
                item_scores[item_idx] += score * weight
                item_weights[item_idx] += weight
        
        # Normalize scores
        final_scores = {}
        for item_idx in item_scores:
            if item_weights[item_idx] > 0:
                final_scores[item_idx] = item_scores[item_idx] / item_weights[item_idx]
            else:
                final_scores[item_idx] = 0
        
        # Сортировка and return top recommendations
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items[:n_recommendations]
    
    def _rank_fusion_combination(self, 
                               model_recommendations: Dict[str, List[Tuple[int, float]]],
                               n_recommendations: int) -> List[Tuple[int, float]]:
        """Combine recommendations using rank fusion (Borda count)"""
        
        # Расчет rank scores
        item_rank_scores = {}
        
        for model_name, recommendations in model_recommendations.items():
            weight = self.weights.get(model_name, 0)
            
            for rank, (item_idx, score) in enumerate(recommendations):
                # Higher rank = lower Индекс (better position)
                rank_score = (len(recommendations) - rank) * weight
                
                if item_idx not in item_rank_scores:
                    item_rank_scores[item_idx] = 0
                
                item_rank_scores[item_idx] += rank_score
        
        # Сортировка by rank Оценка
        sorted_items = sorted(item_rank_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Преобразование to (item_idx, normalized_score) Форматирование
        max_score = max(item_rank_scores.values()) if item_rank_scores else 1
        final_recommendations = [
            (item_idx, score / max_score) 
            for item_idx, score in sorted_items[:n_recommendations]
        ]
        
        return final_recommendations
    
    def get_similar_items(self, 
                         item_idx: int,
                         n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Find similar items using hybrid approach
        
        Args:
            item_idx: Item index
            n_similar: Number of similar items
            
        Returns:
            List of (item_idx, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar items")
        
        # Get similar items from each Модель that supports it
        model_similarities = {}
        
        for model_name, model in self.models.items():
            if model_name not in self.weights:
                continue
                
            try:
                if hasattr(model, 'get_similar_items'):
                    similar_items = model.get_similar_items(item_idx, n_similar * 2)
                    model_similarities[model_name] = similar_items
                    
            except Exception as e:
                print(f"⚠️ Ошибка в модели '{model_name}': {e}")
                continue
        
        if not model_similarities:
            return []
        
        # Объединение using weighted average
        return self._weighted_average_combination(model_similarities, n_similar)
    
    def update_weights(self, 
                      performance_metrics: Dict[str, float],
                      adaptation_rate: float = 0.1):
        """
        Update model weights based on performance metrics
        
        Args:
            performance_metrics: Dictionary with model performance scores
            adaptation_rate: Rate of weight adaptation
        """
        print("📊 Обновление весов моделей на основе производительности...")
        
        # Расчет new weights based on Производительность
        total_performance = sum(performance_metrics.values())
        
        if total_performance > 0:
            new_weights = {}
            for model_name in self.weights:
                if model_name in performance_metrics:
                    # Adaptive weight Обновление
                    performance_weight = performance_metrics[model_name] / total_performance
                    current_weight = self.weights[model_name]
                    
                    new_weight = (1 - adaptation_rate) * current_weight + adaptation_rate * performance_weight
                    new_weights[model_name] = new_weight
                else:
                    new_weights[model_name] = self.weights[model_name]
            
            # Normalize weights
            total_weight = sum(new_weights.values())
            self.weights = {name: weight / total_weight for name, weight in new_weights.items()}
            
            print(f"✅ Обновленные веса: {self.weights}")
    
    def get_model_contributions(self, 
                               user_idx: int,
                               n_recommendations: int = 10) -> Dict[str, List[Tuple[int, float]]]:
        """
        Get individual model contributions for analysis
        
        Args:
            user_idx: User index
            n_recommendations: Number of recommendations
            
        Returns:
            Dictionary with recommendations from each model
        """
        model_contributions = {}
        
        for model_name, model in self.models.items():
            if model_name not in self.weights:
                continue
                
            try:
                if hasattr(model, 'recommend_for_user'):
                    recs = model.recommend_for_user(user_idx, n_recommendations)
                    model_contributions[model_name] = recs
                    
            except Exception as e:
                print(f"⚠️ Ошибка в модели '{model_name}': {e}")
                model_contributions[model_name] = []
        
        return model_contributions
    
    def get_model_params(self) -> Dict:
        """Get hybrid model parameters"""
        return {
            'models': list(self.models.keys()),
            'weights': self.weights,
            'combination_method': self.combination_method,
            'fallback_strategy': self.fallback_strategy
        }


class AdaptiveHybridRecommender(HybridRecommender):
    """
    Adaptive Hybrid Recommender that learns optimal weights over time
    """
    
    def __init__(self, 
                 models: Dict[str, Any],
                 initial_weights: Optional[Dict[str, float]] = None,
                 learning_rate: float = 0.01,
                 exploration_rate: float = 0.1):
        """
        Initialize Adaptive Hybrid Recommender
        
        Args:
            models: Dictionary of recommendation models
            initial_weights: Initial weights for each model
            learning_rate: Learning rate for weight updates
            exploration_rate: Rate of exploration for weight adaptation
        """
        super().__init__(models, initial_weights, 'weighted_average', 'popularity')
        
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.model_rewards = {name: [] for name in models.keys()}
        self.total_interactions = 0
    
    def update_from_feedback(self, 
                           user_idx: int,
                           item_idx: int,
                           feedback: float,
                           model_contributions: Dict[str, float]):
        """
        Update model weights based on user feedback
        
        Args:
            user_idx: User index
            item_idx: Item index that received feedback
            feedback: User feedback (1 for positive, 0 for negative)
            model_contributions: How much each model contributed to this recommendation
        """
        self.total_interactions += 1
        
        # Обновление rewards for each Модель based on their contribution
        for model_name, contribution in model_contributions.items():
            if model_name in self.model_rewards:
                reward = feedback * contribution
                self.model_rewards[model_name].append(reward)
        
        # Обновление weights periodically
        if self.total_interactions % 100 == 0:  # Update every 100 interactions
            self._update_weights_from_rewards()
    
    def _update_weights_from_rewards(self):
        """Update weights based on accumulated rewards"""
        
        # Расчет average rewards
        avg_rewards = {}
        for model_name, rewards in self.model_rewards.items():
            if rewards:
                avg_rewards[model_name] = np.mean(rewards[-100:])  # Use last 100 rewards
            else:
                avg_rewards[model_name] = 0
        
        # Обновление weights using gradient ascent
        total_reward = sum(avg_rewards.values())
        
        if total_reward > 0:
            new_weights = {}
            for model_name in self.weights:
                if model_name in avg_rewards:
                    # Gradient Обновление
                    gradient = avg_rewards[model_name] / total_reward
                    new_weight = self.weights[model_name] + self.learning_rate * gradient
                    
                    # Добавление exploration noise
                    exploration_noise = np.random.normal(0, self.exploration_rate)
                    new_weight += exploration_noise
                    
                    # Ensure positive weights
                    new_weights[model_name] = max(0.01, new_weight)
                else:
                    new_weights[model_name] = self.weights[model_name]
            
            # Normalize weights
            total_weight = sum(new_weights.values())
            self.weights = {name: weight / total_weight for name, weight in new_weights.items()}
            
            print(f"🔄 Адаптивное обновление весов: {self.weights}")


class HybridEvaluator:
    """
    Evaluator for hybrid models
    """
    
    def __init__(self, 
                 train_matrix: csr_matrix,
                 test_matrix: csr_matrix):
        """
        Initialize evaluator
        
        Args:
            train_matrix: Training user-item matrix
            test_matrix: Test user-item matrix
        """
        self.train_matrix = train_matrix
        self.test_matrix = test_matrix
    
    def evaluate_model(self,
                      model: Union[HybridRecommender, AdaptiveHybridRecommender],
                      n_recommendations: int = 10,
                      k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate hybrid model
        
        Args:
            model: Trained hybrid model
            n_recommendations: Number of recommendations to generate
            k_values: List of k values for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        from .metrics import RecommendationMetrics
        
        print(f"📊 Оценка гибридной модели...")
        
        # Get Тестирование users
        test_users = []
        test_items_per_user = {}
        
        for user_idx in range(self.test_matrix.shape[0]):
            user_test_items = self.test_matrix[user_idx].nonzero()[1]
            if len(user_test_items) > 0:
                test_users.append(user_idx)
                test_items_per_user[user_idx] = user_test_items.tolist()
        
        print(f"📊 Тестирование на {len(test_users)} пользователях...")
        
        # Генерация recommendations
        y_true = []
        y_pred = []
        
        for user_idx in test_users:
            try:
                recommendations = model.recommend_for_user(
                    user_idx,
                    n_recommendations=max(k_values),
                    filter_already_liked=True,
                    user_item_matrix=self.train_matrix
                )
                rec_items = [item_idx for item_idx, score in recommendations]
                
                y_true.append(test_items_per_user[user_idx])
                y_pred.append(rec_items)
                
            except Exception as e:
                print(f"⚠️ Ошибка для пользователя {user_idx}: {e}")
                y_true.append(test_items_per_user[user_idx])
                y_pred.append([])
        
        # Расчет metrics
        metrics_calculator = RecommendationMetrics()
        metrics = metrics_calculator.calculate_all_metrics(
            y_true=y_true,
            y_pred=y_pred,
            k_values=k_values,
            total_items=self.train_matrix.shape[1]
        )
        
        return metrics


def create_hybrid_models() -> Dict[str, HybridRecommender]:
    """
    Create hybrid models with different configurations
    
    Returns:
        Dictionary of hybrid models
    """
    from .collaborative_filtering import ALSRecommender, BPRRecommender
    from .content_based import ContentBasedRecommender
    from .baseline import PopularityRecommender
    
    # Define base models
    als_model = ALSRecommender(factors=50, iterations=15)
    content_model = ContentBasedRecommender(similarity_metric='cosine')
    popularity_model = PopularityRecommender()
    
    # Создание Гибридная configurations
    hybrid_models = {
        'hybrid_equal': HybridRecommender(
            models={
                'als': als_model,
                'content': content_model,
                'popularity': popularity_model
            },
            weights={'als': 0.33, 'content': 0.33, 'popularity': 0.34}
        ),
        
        'hybrid_cf_heavy': HybridRecommender(
            models={
                'als': als_model,
                'content': content_model,
                'popularity': popularity_model
            },
            weights={'als': 0.6, 'content': 0.3, 'popularity': 0.1}
        ),
        
        'hybrid_content_heavy': HybridRecommender(
            models={
                'als': als_model,
                'content': content_model,
                'popularity': popularity_model
            },
            weights={'als': 0.2, 'content': 0.7, 'popularity': 0.1}
        ),
        
        'hybrid_rank_fusion': HybridRecommender(
            models={
                'als': als_model,
                'content': content_model
            },
            combination_method='rank_fusion'
        ),
        
        'adaptive_hybrid': AdaptiveHybridRecommender(
            models={
                'als': als_model,
                'content': content_model,
                'popularity': popularity_model
            },
            learning_rate=0.01,
            exploration_rate=0.05
        )
    }
    
    return hybrid_models


# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    
    # Создание synthetic data
    n_users, n_items = 500, 200
    user_item_matrix = csr_matrix(np.random.rand(n_users, n_items) > 0.95).astype(float)
    
    print(f"📊 Создана синтетическая матрица: {user_item_matrix.shape}")
    
    # Создание models
    from .collaborative_filtering import ALSRecommender
    from .baseline import PopularityRecommender
    
    models = {
        'als': ALSRecommender(factors=20, iterations=5),
        'popularity': PopularityRecommender()
    }
    
    # Создание Гибридная Модель
    hybrid_model = HybridRecommender(
        models=models,
        weights={'als': 0.7, 'popularity': 0.3}
    )
    
    # Обучение Модель
    hybrid_model.fit(user_item_matrix)
    
    # Get recommendations
    recommendations = hybrid_model.recommend_for_user(0, n_recommendations=5)
    print(f"\n🎯 Гибридные рекомендации для пользователя 0:")
    for item_idx, score in recommendations:
        print(f"   Товар {item_idx}: {score:.4f}")
    
    # Get Модель contributions
    contributions = hybrid_model.get_model_contributions(0, n_recommendations=5)
    print(f"\n📊 Вклад каждой модели:")
    for model_name, recs in contributions.items():
        print(f"   {model_name}: {len(recs)} рекомендаций")
    
    print("\n✅ Тест гибридной модели завершен")
