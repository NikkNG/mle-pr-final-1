"""
Базовые рекомендательные модели
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from scipy.sparse import csr_matrix
import random
from collections import defaultdict, Counter


class BaselineRecommender:
    """Базовый класс для базовых рекомендаторов"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    def fit(self, interactions_df: pd.DataFrame):
        """Обучение модели"""
        raise NotImplementedError
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Генерация рекомендаций для пользователя"""
        raise NotImplementedError
    
    def recommend_batch(self, user_ids: List[int], n_recommendations: int = 10) -> Dict[int, List[int]]:
        """Генерация рекомендаций для нескольких пользователей"""
        recommendations = {}
        for user_id in user_ids:
            recommendations[user_id] = self.recommend(user_id, n_recommendations)
        return recommendations


class RandomRecommender(BaselineRecommender):
    """Базовая модель случайных рекомендаций"""
    
    def __init__(self, random_state: int = 42):
        super().__init__("Random")
        self.random_state = random_state
        self.items = []
    
    def fit(self, user_item_matrix: Union[csr_matrix, pd.DataFrame]):
        """Обучение случайного рекомендатора"""
        if isinstance(user_item_matrix, csr_matrix):
            # Получение всех товаров из размерности матрицы
            self.items = list(range(user_item_matrix.shape[1]))
        else:
            # Получение всех уникальных товаров из DataFrame
            self.items = list(user_item_matrix['itemid'].unique())
        
        self.is_fitted = True
        
        # Установка случайного семени для воспроизводимости
        random.seed(self.random_state)
        np.random.seed(self.random_state)
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Генерация случайных рекомендаций"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Выборка случайных товаров
        n_rec = min(n_recommendations, len(self.items))
        return random.sample(self.items, n_rec)
    
    def recommend_for_user(self, user_idx: int, n_recommendations: int = 10, 
                          filter_already_liked: bool = True, 
                          user_item_matrix=None) -> List[Tuple[int, float]]:
        """Генерация рекомендаций с оценками"""
        items = self.recommend(user_idx, n_recommendations)
        # Возврат со случайными оценками
        scores = [random.random() for _ in items]
        return list(zip(items, scores))


class PopularityRecommender(BaselineRecommender):
    """Базовая модель самых популярных товаров"""
    
    def __init__(self):
        super().__init__("Popularity")
        self.popular_items = []
        self.item_scores = {}
    
    def fit(self, user_item_matrix: Union[csr_matrix, pd.DataFrame]):
        """Обучение рекомендатора по популярности"""
        if isinstance(user_item_matrix, csr_matrix):
            # Расчет популярности товаров из разреженной матрицы
            item_scores = np.array(user_item_matrix.sum(axis=0)).flatten()
            # Создание отсортированного списка товаров по популярности
            item_indices = np.argsort(item_scores)[::-1]
            self.popular_items = item_indices.tolist()
            self.item_scores = {i: float(score) for i, score in enumerate(item_scores)}
        else:
            # Расчет оценок популярности товаров из DataFrame
            item_counts = user_item_matrix['itemid'].value_counts()
            self.popular_items = item_counts.index.tolist()
            self.item_scores = item_counts.to_dict()
        
        self.is_fitted = True
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Генерация рекомендаций на основе популярности"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Возврат топ N популярных товаров
        n_rec = min(n_recommendations, len(self.popular_items))
        return self.popular_items[:n_rec]
    
    def recommend_for_user(self, user_idx: int, n_recommendations: int = 10, 
                          filter_already_liked: bool = True, 
                          user_item_matrix=None) -> List[Tuple[int, float]]:
        """Генерация рекомендаций с оценками"""
        items = self.recommend(user_idx, n_recommendations)
        # Возврат с оценками популярности
        scores = [self.item_scores.get(item, 0.0) for item in items]
        return list(zip(items, scores))


class WeightedPopularityRecommender(BaselineRecommender):
    """Взвешенный рекомендатор по популярности (учитывает типы событий)"""
    
    def __init__(self, event_weights: Optional[Dict[str, float]] = None):
        super().__init__("WeightedPopularity")
        
        # Веса по умолчанию для различных типов событий
        if event_weights is None:
            self.event_weights = {
                'view': 1.0,
                'addtocart': 2.0,
                'transaction': 3.0
            }
        else:
            self.event_weights = event_weights
        
        self.popular_items = []
        self.item_scores = {}
    
    def fit(self, user_item_matrix: Union[csr_matrix, pd.DataFrame]):
        """Обучение взвешенного рекомендатора по популярности"""
        if isinstance(user_item_matrix, csr_matrix):
            # Для разреженной матрицы использование простой популярности (веса уже применены при предобработке)
            item_scores = np.array(user_item_matrix.sum(axis=0)).flatten()
            item_indices = np.argsort(item_scores)[::-1]
            self.popular_items = item_indices.tolist()
            self.item_scores = {i: float(score) for i, score in enumerate(item_scores)}
        else:
            # Применение весов к событиям для DataFrame
            weighted_df = user_item_matrix.copy()
            weighted_df['weight'] = weighted_df['event'].map(self.event_weights).fillna(1.0)
            
            # Расчет взвешенных оценок популярности
            item_scores = weighted_df.groupby('itemid')['weight'].sum()
            item_scores = item_scores.sort_values(ascending=False)
            
            # Сохранение популярных товаров в порядке
            self.popular_items = item_scores.index.tolist()
            self.item_scores = item_scores.to_dict()
        
        self.is_fitted = True
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Генерация рекомендаций на основе взвешенной популярности"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Возврат топ N популярных товаров
        n_rec = min(n_recommendations, len(self.popular_items))
        return self.popular_items[:n_rec]
    
    def recommend_for_user(self, user_idx: int, n_recommendations: int = 10, 
                          filter_already_liked: bool = True, 
                          user_item_matrix=None) -> List[Tuple[int, float]]:
        """Генерация рекомендаций с оценками"""
        items = self.recommend(user_idx, n_recommendations)
        # Возврат с взвешенными оценками популярности
        scores = [self.item_scores.get(item, 0.0) for item in items]
        return list(zip(items, scores))


class RecentlyViewedRecommender(BaselineRecommender):
    """Базовая модель недавно просмотренных товаров"""
    
    def __init__(self, max_items_per_user: int = 50):
        super().__init__("RecentlyViewed")
        self.max_items_per_user = max_items_per_user
        self.user_items = {}
        self.fallback_items = []
    
    def fit(self, interactions_df: pd.DataFrame):
        """Обучение рекомендатора недавно просмотренных товаров"""
        # Сортировка по временной метке, если доступна
        if 'timestamp' in interactions_df.columns:
            interactions_df = interactions_df.sort_values('timestamp', ascending=False)
        
        # Группировка по пользователю и получение недавних товаров
        self.user_items = {}
        for user_id, user_data in interactions_df.groupby('visitorid'):
            recent_items = user_data['itemid'].drop_duplicates().tolist()
            self.user_items[user_id] = recent_items[:self.max_items_per_user]
        
        # Fallback to popular items for new users
        item_counts = interactions_df['itemid'].value_counts()
        self.fallback_items = item_counts.index.tolist()
        
        self.is_fitted = True
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Generate recently viewed recommendations"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get Пользователь's recent items
        if user_id in self.user_items:
            user_recent = self.user_items[user_id]
            n_rec = min(n_recommendations, len(user_recent))
            recommendations = user_recent[:n_rec]
            
            # Fill with popular items if needed
            if len(recommendations) < n_recommendations:
                remaining = n_recommendations - len(recommendations)
                for item in self.fallback_items:
                    if item not in recommendations and len(recommendations) < n_recommendations:
                        recommendations.append(item)
            
            return recommendations[:n_recommendations]
        else:
            # New Пользователь - return popular items
            n_rec = min(n_recommendations, len(self.fallback_items))
            return self.fallback_items[:n_rec]


class CategoryBasedRecommender(BaselineRecommender):
    """Category-based recommendations"""
    
    def __init__(self):
        super().__init__("CategoryBased")
        self.user_categories = {}
        self.category_items = defaultdict(list)
        self.fallback_items = []
    
    def fit(self, interactions_df: pd.DataFrame, item_properties_df: pd.DataFrame):
        """Fit category-based recommender"""
        # Создание Товар to Категория mapping
        category_data = item_properties_df[
            item_properties_df['property'].str.contains('category', case=False, na=False)
        ]
        
        item_category_map = {}
        for _, row in category_data.iterrows():
            try:
                item_category_map[row['itemid']] = str(row['value'])
            except (ValueError, TypeError):
                continue
        
        # Добавление categories to interactions
        interactions_with_cat = interactions_df.copy()
        interactions_with_cat['category'] = interactions_with_cat['itemid'].map(item_category_map)
        interactions_with_cat = interactions_with_cat.dropna(subset=['category'])
        
        # Расчет Пользователь Категория preferences
        self.user_categories = {}
        for user_id, user_data in interactions_with_cat.groupby('visitorid'):
            category_counts = user_data['category'].value_counts()
            self.user_categories[user_id] = category_counts.index.tolist()
        
        # Группировка items by Категория
        self.category_items = defaultdict(list)
        for _, row in interactions_with_cat.iterrows():
            self.category_items[row['category']].append(row['itemid'])
        
        # Удаление duplicates and Сортировка by Популярность
        for category in self.category_items:
            item_counts = Counter(self.category_items[category])
            self.category_items[category] = [item for item, _ in item_counts.most_common()]
        
        # Fallback items
        item_counts = interactions_df['itemid'].value_counts()
        self.fallback_items = item_counts.index.tolist()
        
        self.is_fitted = True
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[int]:
        """Generate category-based recommendations"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        recommendations = []
        
        # Get Пользователь's preferred categories
        if user_id in self.user_categories:
            preferred_categories = self.user_categories[user_id]
            
            # Рекомендация items from preferred categories
            for category in preferred_categories:
                if category in self.category_items:
                    category_items = self.category_items[category]
                    for item in category_items:
                        if item not in recommendations and len(recommendations) < n_recommendations:
                            recommendations.append(item)
                        if len(recommendations) >= n_recommendations:
                            break
                if len(recommendations) >= n_recommendations:
                    break
        
        # Fill with popular items if needed
        if len(recommendations) < n_recommendations:
            for item in self.fallback_items:
                if item not in recommendations and len(recommendations) < n_recommendations:
                    recommendations.append(item)
        
        return recommendations[:n_recommendations]


class CoOccurrenceRecommender(BaselineRecommender):
    """Co-occurrence based recommendations (items frequently viewed together)"""
    
    def __init__(self, min_cooccurrence: int = 2):
        super().__init__("CoOccurrence")
        self.min_cooccurrence = min_cooccurrence
        self.cooccurrence_matrix = {}
        self.item_popularity = {}
    
    def fit(self, interactions_df: pd.DataFrame):
        """Fit co-occurrence recommender"""
        # Расчет Товар Популярность
        self.item_popularity = interactions_df['itemid'].value_counts().to_dict()
        
        # Сборка co-occurrence Матрица
        self.cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
        
        # Группировка by Пользователь sessions
        for user_id, user_data in interactions_df.groupby('visitorid'):
            user_items = user_data['itemid'].unique().tolist()
            
            # Count co-occurrences
            for i, item1 in enumerate(user_items):
                for item2 in user_items[i+1:]:
                    self.cooccurrence_matrix[item1][item2] += 1
                    self.cooccurrence_matrix[item2][item1] += 1
        
        # Фильтрация by minimum co-occurrence
        filtered_matrix = defaultdict(dict)
        for item1 in self.cooccurrence_matrix:
            for item2, count in self.cooccurrence_matrix[item1].items():
                if count >= self.min_cooccurrence:
                    filtered_matrix[item1][item2] = count
        
        self.cooccurrence_matrix = filtered_matrix
        self.is_fitted = True
    
    def recommend(self, user_id: int, n_recommendations: int = 10, user_items: Optional[List[int]] = None) -> List[int]:
        """Generate co-occurrence based recommendations"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_items is None:
            # Fallback to popular items if no Пользователь history provided
            popular_items = sorted(self.item_popularity.items(), key=lambda x: x[1], reverse=True)
            return [item for item, _ in popular_items[:n_recommendations]]
        
        # Расчет Рекомендация scores
        item_scores = defaultdict(float)
        
        for user_item in user_items:
            if user_item in self.cooccurrence_matrix:
                for related_item, score in self.cooccurrence_matrix[user_item].items():
                    if related_item not in user_items:  # Don't recommend already interacted items
                        item_scores[related_item] += score
        
        # Сортировка by Оценка and return top N
        if item_scores:
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = [item for item, _ in sorted_items[:n_recommendations]]
        else:
            # Fallback to popular items
            popular_items = sorted(self.item_popularity.items(), key=lambda x: x[1], reverse=True)
            recommendations = [item for item, _ in popular_items[:n_recommendations]]
        
        return recommendations


def create_baseline_models() -> Dict[str, BaselineRecommender]:
    """Create all baseline models"""
    models = {
        'random': RandomRecommender(),
        'popularity': PopularityRecommender(),
        'weighted_popularity': WeightedPopularityRecommender(),
        'recently_viewed': RecentlyViewedRecommender(),
        'category_based': CategoryBasedRecommender(),
        'cooccurrence': CoOccurrenceRecommender()
    }
    
    return models


def evaluate_baseline_models(
    interactions_df: pd.DataFrame,
    item_properties_df: pd.DataFrame,
    test_interactions: Dict[int, List[int]],
    n_recommendations: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all baseline models
    
    Args:
        interactions_df: Training interactions
        item_properties_df: Item properties for category-based model
        test_interactions: Test interactions for evaluation
        n_recommendations: Number of recommendations to generate
        
    Returns:
        Dictionary with evaluation results for each model
    """
    from .metrics import RecommendationMetrics
    
    # Создание Базовая models
    models = create_baseline_models()
    
    # Обучение models
    for name, model in models.items():
        print(f"Fitting {name} model...")
        if name == 'category_based':
            model.fit(interactions_df, item_properties_df)
        else:
            model.fit(interactions_df)
    
    # Оценка models
    metrics_calculator = RecommendationMetrics()
    results = {}
    
    test_user_ids = list(test_interactions.keys())
    
    for name, model in models.items():
        print(f"Evaluating {name} model...")
        
        # Генерация recommendations
        if name == 'cooccurrence':
            # For co-occurrence, we need Пользователь history
            user_history = {}
            for user_id in test_user_ids:
                user_data = interactions_df[interactions_df['visitorid'] == user_id]
                user_history[user_id] = user_data['itemid'].unique().tolist()
            
            recommendations = {}
            for user_id in test_user_ids:
                recommendations[user_id] = model.recommend(
                    user_id, n_recommendations, user_history.get(user_id, [])
                )
        else:
            recommendations = model.recommend_batch(test_user_ids, n_recommendations)
        
        # Prepare data for evaluation
        y_true = []
        y_pred = []
        
        for user_id in test_user_ids:
            if user_id in recommendations and user_id in test_interactions:
                y_true.append(test_interactions[user_id])
                y_pred.append(recommendations[user_id])
        
        # Расчет metrics
        model_metrics = metrics_calculator.calculate_all_metrics(
            y_true=y_true,
            y_pred=y_pred,
            k_values=[5, 10, 20],
            total_items=interactions_df['itemid'].nunique()
        )
        
        results[name] = model_metrics
    
    return results


# Example usage
if __name__ == "__main__":
    # Example data
    interactions_data = {
        'visitorid': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'itemid': [101, 102, 103, 101, 104, 102, 103, 105, 106],
        'event': ['view', 'view', 'addtocart', 'view', 'transaction', 'view', 'view', 'view', 'addtocart']
    }
    
    interactions_df = pd.DataFrame(interactions_data)
    
    # Тестирование Популярность recommender
    pop_model = PopularityRecommender()
    pop_model.fit(interactions_df)
    
    print("Popularity-based recommendations:")
    print(pop_model.recommend(user_id=1, n_recommendations=5))
    
    # Тестирование Случайная recommender
    random_model = RandomRecommender()
    random_model.fit(interactions_df)
    
    print("\nRandom recommendations:")
    print(random_model.recommend(user_id=1, n_recommendations=5)) 