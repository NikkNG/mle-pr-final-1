"""
Content-Based Filtering models for recommendation system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')


class ContentBasedRecommender:
    """
    Content-Based Recommender using item features
    """
    
    def __init__(self, 
                 similarity_metric: str = 'cosine',
                 n_components: Optional[int] = None,
                 random_state: int = 42):
        """
        Initialize Content-Based Recommender
        
        Args:
            similarity_metric: Similarity metric ('cosine', 'euclidean')
            n_components: Number of components for dimensionality reduction (optional)
            random_state: Random seed
        """
        self.similarity_metric = similarity_metric
        self.n_components = n_components
        self.random_state = random_state
        
        # Модель components
        self.item_features = None
        self.item_similarity_matrix = None
        self.user_profiles = None
        self.item_encoder = None
        self.feature_scaler = StandardScaler()
        self.svd = None
        
        # Fitted flag
        self.is_fitted = False
    
    def fit(self, 
            user_item_matrix: csr_matrix,
            item_features: Dict[str, Union[np.ndarray, Dict]],
            metadata: Dict) -> 'ContentBasedRecommender':
        """
        Fit Content-Based model
        
        Args:
            user_item_matrix: User-item interaction matrix
            item_features: Dictionary with item features
            metadata: Metadata dictionary
            
        Returns:
            Self for method chaining
        """
        print("🤖 Обучение Content-Based модели...")
        
        # Сохранение metadata
        self.item_encoder = metadata['item_encoder']
        
        # Обработка Товар features
        self.item_features = self._process_features(item_features)
        print(f"📊 Обработано {self.item_features.shape[0]} товаров с {self.item_features.shape[1]} признаками")
        
        # Apply dimensionality reduction if specified
        if self.n_components and self.n_components < self.item_features.shape[1]:
            print(f"📉 Применение SVD: {self.item_features.shape[1]} -> {self.n_components} компонент")
            self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
            self.item_features = self.svd.fit_transform(self.item_features)
        
        # Расчет Товар Схожесть Матрица
        self.item_similarity_matrix = self._calculate_similarity_matrix()
        print(f"📊 Создана матрица схожести товаров: {self.item_similarity_matrix.shape}")
        
        # Сборка Пользователь profiles
        self.user_profiles = self._build_user_profiles(user_item_matrix)
        print(f"📊 Создано {len(self.user_profiles)} профилей пользователей")
        
        self.is_fitted = True
        print("✅ Content-Based модель обучена")
        
        return self
    
    def _process_features(self, item_features: Dict) -> np.ndarray:
        """Process item features into numerical matrix"""
        print("🔧 Обработка признаков товаров...")
        
        # Get Товар IDs
        item_ids = self.item_encoder.classes_
        n_items = len(item_ids)
        
        feature_vectors = []
        
        # Обработка TF-IDF features if available
        if 'tfidf' in item_features and item_features['tfidf']:
            print("📝 Обработка TF-IDF признаков...")
            tfidf_features = item_features['tfidf']
            
            # Создание TF-IDF Матрица
            tfidf_matrix = []
            for item_id in item_ids:
                if str(item_id) in tfidf_features:
                    tfidf_matrix.append(tfidf_features[str(item_id)])
                else:
                    # Zero Вектор for items without features
                    feature_dim = len(next(iter(tfidf_features.values())))
                    tfidf_matrix.append(np.zeros(feature_dim))
            
            tfidf_matrix = np.array(tfidf_matrix)
            feature_vectors.append(tfidf_matrix)
            print(f"   TF-IDF: {tfidf_matrix.shape[1]} признаков")
        
        # Обработка Категория features if available
        if 'categories' in item_features and item_features['categories']:
            print("🏷️ Обработка категорий...")
            categories = item_features['categories']
            
            # Get unique categories
            unique_categories = list(set(categories.values()))
            category_encoder = LabelEncoder()
            category_encoder.fit(unique_categories)
            
            # Создание one-hot encoded Категория features
            category_matrix = np.zeros((n_items, len(unique_categories)))
            for i, item_id in enumerate(item_ids):
                if str(item_id) in categories:
                    category_idx = category_encoder.transform([categories[str(item_id)]])[0]
                    category_matrix[i, category_idx] = 1
            
            feature_vectors.append(category_matrix)
            print(f"   Категории: {category_matrix.shape[1]} признаков")
        
        # Объединение all features
        if feature_vectors:
            combined_features = np.hstack(feature_vectors)
        else:
            # Создание Случайная features if no features available
            print("⚠️ Признаки товаров недоступны, создаем случайные признаки")
            combined_features = np.random.rand(n_items, 50)
        
        # Масштабирование признаков
        combined_features = self.feature_scaler.fit_transform(combined_features)
        
        return combined_features
    
    def _calculate_similarity_matrix(self) -> np.ndarray:
        """Calculate item similarity matrix"""
        print(f"📊 Вычисление матрицы схожести методом '{self.similarity_metric}'...")
        
        if self.similarity_metric == 'cosine':
            similarity_matrix = cosine_similarity(self.item_features)
        elif self.similarity_metric == 'euclidean':
            # Преобразование distances to similarities
            distances = euclidean_distances(self.item_features)
            max_distance = np.max(distances)
            similarity_matrix = 1 - (distances / max_distance)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return similarity_matrix
    
    def _build_user_profiles(self, user_item_matrix: csr_matrix) -> Dict[int, np.ndarray]:
        """Build user profiles based on item interactions"""
        print("👤 Создание профилей пользователей...")
        
        user_profiles = {}
        
        for user_idx in range(user_item_matrix.shape[0]):
            # Get Пользователь interactions
            user_items = user_item_matrix[user_idx].nonzero()[1]
            user_ratings = user_item_matrix[user_idx].data
            
            if len(user_items) > 0:
                # Weight Товар features by interaction strength
                weighted_features = []
                total_weight = 0
                
                for item_idx, rating in zip(user_items, user_ratings):
                    if item_idx < len(self.item_features):
                        weighted_features.append(self.item_features[item_idx] * rating)
                        total_weight += rating
                
                if weighted_features and total_weight > 0:
                    # Average weighted features
                    user_profile = np.sum(weighted_features, axis=0) / total_weight
                    user_profiles[user_idx] = user_profile
        
        return user_profiles
    
    def recommend_for_user(self, 
                          user_idx: int,
                          n_recommendations: int = 10,
                          filter_already_liked: bool = True,
                          user_item_matrix: Optional[csr_matrix] = None) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user
        
        Args:
            user_idx: User index
            n_recommendations: Number of recommendations
            filter_already_liked: Whether to filter items user already interacted with
            user_item_matrix: User-item matrix (for filtering)
            
        Returns:
            List of (item_idx, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_idx not in self.user_profiles:
            # Cold Запуск: Рекомендация popular items or Случайная items
            return self._handle_cold_start_user(n_recommendations)
        
        # Get Пользователь profile
        user_profile = self.user_profiles[user_idx]
        
        # Расчет similarities with all items
        item_scores = cosine_similarity([user_profile], self.item_features)[0]
        
        # Get items to Фильтрация out
        items_to_filter = set()
        if filter_already_liked and user_item_matrix is not None:
            items_to_filter = set(user_item_matrix[user_idx].nonzero()[1])
        
        # Сортировка items by Оценка
        item_indices = np.argsort(item_scores)[::-1]
        
        # Фильтрация and collect recommendations
        recommendations = []
        for item_idx in item_indices:
            if item_idx not in items_to_filter:
                recommendations.append((item_idx, float(item_scores[item_idx])))
                if len(recommendations) >= n_recommendations:
                    break
        
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
        
        if item_idx >= len(self.item_similarity_matrix):
            raise ValueError(f"Item index {item_idx} out of range")
        
        # Get Схожесть scores
        similarities = self.item_similarity_matrix[item_idx]
        
        # Сортировка by Схожесть (excluding the Товар itself)
        similar_indices = np.argsort(similarities)[::-1]
        
        # Collect similar items
        similar_items = []
        for idx in similar_indices:
            if idx != item_idx:  # Exclude the item itself
                similar_items.append((idx, float(similarities[idx])))
                if len(similar_items) >= n_similar:
                    break
        
        return similar_items
    
    def _handle_cold_start_user(self, n_recommendations: int) -> List[Tuple[int, float]]:
        """Handle cold start users"""
        # Return items with highest average Признак values (Прокси for Популярность)
        item_scores = np.mean(self.item_features, axis=1)
        top_items = np.argsort(item_scores)[::-1][:n_recommendations]
        
        return [(int(item_idx), float(item_scores[item_idx])) for item_idx in top_items]
    
    def get_model_params(self) -> Dict:
        """Get model parameters"""
        return {
            'similarity_metric': self.similarity_metric,
            'n_components': self.n_components,
            'random_state': self.random_state
        }


class TFIDFRecommender:
    """
    TF-IDF based Content Recommender
    """
    
    def __init__(self,
                 max_features: int = 1000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.8):
        """
        Initialize TF-IDF Recommender
        
        Args:
            max_features: Maximum number of features
            ngram_range: N-gram range for TF-IDF
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        # Модель components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True
        )
        
        self.item_features = None
        self.item_similarity_matrix = None
        self.item_encoder = None
        self.is_fitted = False
    
    def fit(self, 
            item_properties_df: pd.DataFrame,
            metadata: Dict) -> 'TFIDFRecommender':
        """
        Fit TF-IDF model
        
        Args:
            item_properties_df: Item properties DataFrame
            metadata: Metadata dictionary
            
        Returns:
            Self for method chaining
        """
        print("🤖 Обучение TF-IDF модели...")
        
        # Сохранение metadata
        self.item_encoder = metadata['item_encoder']
        item_ids = self.item_encoder.classes_
        
        # Создание Товар text descriptions
        item_texts = self._create_item_texts(item_properties_df, item_ids)
        
        # Обучение TF-IDF vectorizer
        self.item_features = self.tfidf_vectorizer.fit_transform(item_texts)
        print(f"📊 TF-IDF матрица: {self.item_features.shape}")
        
        # Расчет Схожесть Матрица
        self.item_similarity_matrix = cosine_similarity(self.item_features)
        print(f"📊 Матрица схожести: {self.item_similarity_matrix.shape}")
        
        self.is_fitted = True
        print("✅ TF-IDF модель обучена")
        
        return self
    
    def _create_item_texts(self, 
                          item_properties_df: pd.DataFrame,
                          item_ids: np.ndarray) -> List[str]:
        """Create text descriptions for items"""
        print("📝 Создание текстовых описаний товаров...")
        
        # Группировка Свойства by Товар
        item_texts = []
        
        for item_id in item_ids:
            item_props = item_properties_df[
                item_properties_df['itemid'].astype(str) == str(item_id)
            ]
            
            if not item_props.empty:
                # Объединение all Свойства and values
                text_parts = []
                for _, row in item_props.iterrows():
                    prop_text = f"{row['property']} {row['value']}"
                    text_parts.append(prop_text)
                
                item_text = ' '.join(text_parts)
            else:
                # Empty text for items without Свойства
                item_text = ""
            
            item_texts.append(item_text)
        
        return item_texts
    
    def get_similar_items(self, 
                         item_idx: int,
                         n_similar: int = 10) -> List[Tuple[int, float]]:
        """Find similar items using TF-IDF"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar items")
        
        if item_idx >= len(self.item_similarity_matrix):
            raise ValueError(f"Item index {item_idx} out of range")
        
        # Get Схожесть scores
        similarities = self.item_similarity_matrix[item_idx]
        
        # Сортировка by Схожесть (excluding the Товар itself)
        similar_indices = np.argsort(similarities)[::-1]
        
        # Collect similar items
        similar_items = []
        for idx in similar_indices:
            if idx != item_idx:  # Exclude the item itself
                similar_items.append((idx, float(similarities[idx])))
                if len(similar_items) >= n_similar:
                    break
        
        return similar_items


class ContentBasedEvaluator:
    """
    Evaluator for content-based models
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
                      model: Union[ContentBasedRecommender, TFIDFRecommender],
                      n_recommendations: int = 10,
                      k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate content-based model
        
        Args:
            model: Trained model
            n_recommendations: Number of recommendations to generate
            k_values: List of k values for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        from .metrics import RecommendationMetrics
        
        print(f"📊 Оценка модели {model.__class__.__name__}...")
        
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
                if hasattr(model, 'recommend_for_user'):
                    recommendations = model.recommend_for_user(
                        user_idx,
                        n_recommendations=max(k_values),
                        filter_already_liked=True,
                        user_item_matrix=self.train_matrix
                    )
                    rec_items = [item_idx for item_idx, score in recommendations]
                else:
                    # For models that don't have Пользователь recommendations
                    rec_items = []
                
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


def create_content_based_models() -> Dict[str, Union[ContentBasedRecommender, TFIDFRecommender]]:
    """
    Create content-based models with different configurations
    
    Returns:
        Dictionary of models
    """
    models = {
        'content_cosine': ContentBasedRecommender(similarity_metric='cosine'),
        'content_euclidean': ContentBasedRecommender(similarity_metric='euclidean'),
        'content_reduced': ContentBasedRecommender(similarity_metric='cosine', n_components=50),
        'tfidf_default': TFIDFRecommender(),
        'tfidf_bigrams': TFIDFRecommender(ngram_range=(1, 3)),
        'tfidf_large': TFIDFRecommender(max_features=2000)
    }
    
    return models


# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    
    # Создание synthetic Товар features
    n_items = 100
    feature_dim = 50
    
    item_features = {
        'tfidf': {
            str(i): np.random.rand(feature_dim) 
            for i in range(n_items)
        },
        'categories': {
            str(i): np.random.choice(['electronics', 'clothing', 'books'])
            for i in range(n_items)
        }
    }
    
    # Создание synthetic Пользователь-Товар Матрица
    n_users = 200
    user_item_matrix = csr_matrix(
        np.random.rand(n_users, n_items) > 0.95
    ).astype(float)
    
    # Создание metadata
    metadata = {
        'item_encoder': LabelEncoder().fit(range(n_items))
    }
    
    print(f"📊 Создана синтетическая матрица: {user_item_matrix.shape}")
    print(f"📊 Признаки товаров: {len(item_features)}")
    
    # Тестирование Контентная-Based Модель
    cb_model = ContentBasedRecommender()
    cb_model.fit(user_item_matrix, item_features, metadata)
    
    # Get recommendations for first Пользователь
    recommendations = cb_model.recommend_for_user(
        0, n_recommendations=5, 
        user_item_matrix=user_item_matrix
    )
    print(f"\n🎯 Рекомендации для пользователя 0:")
    for item_idx, score in recommendations:
        print(f"   Товар {item_idx}: {score:.4f}")
    
    # Get similar items
    similar_items = cb_model.get_similar_items(0, n_similar=3)
    print(f"\n🔗 Похожие товары на товар 0:")
    for item_idx, score in similar_items:
        print(f"   Товар {item_idx}: {score:.4f}")
    
    print("\n✅ Тест контентной фильтрации завершен")
