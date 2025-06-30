"""
Data preprocessing pipeline for recommendation system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Data preprocessing pipeline for recommendation system
    """
    
    def __init__(self, 
                 min_user_interactions: int = 5,
                 min_item_interactions: int = 5,
                 event_weights: Optional[Dict[str, float]] = None):
        """
        Initialize preprocessor
        
        Args:
            min_user_interactions: Minimum interactions per user
            min_item_interactions: Minimum interactions per item
            event_weights: Weights for different event types
        """
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        
        # Default Событие weights
        if event_weights is None:
            self.event_weights = {
                'view': 1.0,
                'addtocart': 2.0,
                'transaction': 3.0
            }
        else:
            self.event_weights = event_weights
        
        # Fitted encoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Data statistics
        self.stats = {}
    
    def fit_transform(self, 
                     events_df: pd.DataFrame,
                     item_properties_df: Optional[pd.DataFrame] = None) -> Tuple[csr_matrix, Dict]:
        """
        Fit preprocessor and transform data
        
        Args:
            events_df: Events DataFrame
            item_properties_df: Item properties DataFrame (optional)
            
        Returns:
            Tuple of (user_item_matrix, metadata)
        """
        print("🔧 Начинаем предобработку данных...")
        
        # Step 1: Очистка and Фильтрация events
        events_clean = self._clean_events(events_df)
        print(f"📊 После очистки: {len(events_clean):,} событий")
        
        # Step 2: Apply Событие weights
        events_weighted = self._apply_event_weights(events_clean)
        
        # Step 3: Фильтрация users and items by minimum interactions
        events_filtered = self._filter_by_interactions(events_weighted)
        print(f"📊 После фильтрации: {len(events_filtered):,} событий")
        
        # Step 4: Создание Пользователь-Товар Матрица
        user_item_matrix = self._create_user_item_matrix(events_filtered)
        print(f"📊 Матрица: {user_item_matrix.shape[0]:,} пользователей x {user_item_matrix.shape[1]:,} товаров")
        
        # Step 5: Создание metadata
        metadata = self._create_metadata(events_filtered, item_properties_df)
        
        self.is_fitted = True
        
        return user_item_matrix, metadata
    
    def transform(self, events_df: pd.DataFrame) -> csr_matrix:
        """
        Transform new data using fitted preprocessor
        
        Args:
            events_df: New events DataFrame
            
        Returns:
            User-item matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Очистка and weight events
        events_clean = self._clean_events(events_df)
        events_weighted = self._apply_event_weights(events_clean)
        
        # Фильтрация to only include known users and items
        known_users = set(self.user_encoder.classes_)
        known_items = set(self.item_encoder.classes_)
        
        events_filtered = events_weighted[
            (events_weighted['visitorid'].isin(known_users)) &
            (events_weighted['itemid'].isin(known_items))
        ]
        
        # Создание Матрица using fitted encoders
        return self._create_user_item_matrix(events_filtered, fit_encoders=False)
    
    def _clean_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Clean events data"""
        print("🧹 Очистка данных...")
        
        # Удаление duplicates
        events_clean = events_df.drop_duplicates()
        
        # Удаление null values
        events_clean = events_clean.dropna(subset=['visitorid', 'itemid', 'event'])
        
        # Фильтрация valid Событие types
        valid_events = set(self.event_weights.keys())
        events_clean = events_clean[events_clean['event'].isin(valid_events)]
        
        # Преобразование IDs to string for consistency
        events_clean['visitorid'] = events_clean['visitorid'].astype(str)
        events_clean['itemid'] = events_clean['itemid'].astype(str)
        
        # Сохранение statistics
        self.stats['original_events'] = len(events_df)
        self.stats['clean_events'] = len(events_clean)
        self.stats['unique_users'] = events_clean['visitorid'].nunique()
        self.stats['unique_items'] = events_clean['itemid'].nunique()
        
        return events_clean
    
    def _apply_event_weights(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Apply weights to different event types"""
        print("⚖️ Применение весов событий...")
        
        events_weighted = events_df.copy()
        events_weighted['weight'] = events_weighted['event'].map(self.event_weights)
        
        return events_weighted
    
    def _filter_by_interactions(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Filter users and items by minimum interactions"""
        print("🔍 Фильтрация по минимальному количеству взаимодействий...")
        
        # Count interactions per Пользователь and Товар
        user_counts = events_df['visitorid'].value_counts()
        item_counts = events_df['itemid'].value_counts()
        
        # Фильтрация users and items
        active_users = user_counts[user_counts >= self.min_user_interactions].index
        active_items = item_counts[item_counts >= self.min_item_interactions].index
        
        # Apply filters
        events_filtered = events_df[
            (events_df['visitorid'].isin(active_users)) &
            (events_df['itemid'].isin(active_items))
        ]
        
        # Сохранение statistics
        self.stats['filtered_events'] = len(events_filtered)
        self.stats['active_users'] = len(active_users)
        self.stats['active_items'] = len(active_items)
        
        return events_filtered
    
    def _create_user_item_matrix(self, events_df: pd.DataFrame, fit_encoders: bool = True) -> csr_matrix:
        """Create user-item interaction matrix"""
        print("🏗️ Создание user-item матрицы...")
        
        # Aggregate interactions by Пользователь-Товар pairs
        interactions = events_df.groupby(['visitorid', 'itemid'])['weight'].sum().reset_index()
        
        if fit_encoders:
            # Обучение encoders
            user_ids = interactions['visitorid'].unique()
            item_ids = interactions['itemid'].unique()
            
            self.user_encoder.fit(user_ids)
            self.item_encoder.fit(item_ids)
        
        # Кодирование Пользователь and Товар IDs
        interactions['user_idx'] = self.user_encoder.transform(interactions['visitorid'])
        interactions['item_idx'] = self.item_encoder.transform(interactions['itemid'])
        
        # Создание sparse Матрица
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        user_item_matrix = csr_matrix(
            (interactions['weight'], (interactions['user_idx'], interactions['item_idx'])),
            shape=(n_users, n_items)
        )
        
        return user_item_matrix
    
    def _create_metadata(self, events_df: pd.DataFrame, item_properties_df: Optional[pd.DataFrame] = None) -> Dict:
        """Create metadata dictionary"""
        print("📋 Создание метаданных...")
        
        metadata = {
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'n_users': len(self.user_encoder.classes_),
            'n_items': len(self.item_encoder.classes_),
            'stats': self.stats
        }
        
        # Добавление Товар features if available
        if item_properties_df is not None:
            item_features = self._process_item_features(item_properties_df)
            metadata['item_features'] = item_features
        
        # Добавление Пользователь statistics
        user_stats = self._calculate_user_stats(events_df)
        metadata['user_stats'] = user_stats
        
        # Добавление Товар statistics
        item_stats = self._calculate_item_stats(events_df)
        metadata['item_stats'] = item_stats
        
        return metadata
    
    def _process_item_features(self, item_properties_df: pd.DataFrame) -> Dict:
        """Process item properties into features"""
        print("🏷️ Обработка свойств товаров...")
        
        # Фильтрация items that are in our active set
        active_items = set(self.item_encoder.classes_)
        item_props_filtered = item_properties_df[
            item_properties_df['itemid'].astype(str).isin(active_items)
        ]
        
        features = {}
        
        # Создание Категория mapping
        category_data = item_props_filtered[
            item_props_filtered['property'].str.contains('category', case=False, na=False)
        ]
        
        if not category_data.empty:
            category_map = {}
            for _, row in category_data.iterrows():
                item_id = str(row['itemid'])
                if item_id in active_items:
                    try:
                        category_map[item_id] = str(row['value'])
                    except:
                        continue
            features['categories'] = category_map
        
        # Создание text features (for Контентная-based filtering)
        text_data = item_props_filtered.groupby('itemid').agg({
            'property': lambda x: ' '.join(x.astype(str)),
            'value': lambda x: ' '.join(x.astype(str))
        }).reset_index()
        
        if not text_data.empty:
            # Объединение property and value text
            text_data['combined_text'] = text_data['property'] + ' ' + text_data['value']
            
            # Создание TF-IDF features
            tfidf = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2)
            )
            
            try:
                tfidf_matrix = tfidf.fit_transform(text_data['combined_text'])
                
                # Map Товар IDs to TF-IDF vectors
                item_text_features = {}
                for idx, item_id in enumerate(text_data['itemid']):
                    item_id_str = str(item_id)
                    if item_id_str in active_items:
                        item_text_features[item_id_str] = tfidf_matrix[idx].toarray().flatten()
                
                features['tfidf'] = item_text_features
                features['tfidf_vectorizer'] = tfidf
                
            except Exception as e:
                print(f"⚠️ Ошибка при создании TF-IDF признаков: {e}")
        
        return features
    
    def _calculate_user_stats(self, events_df: pd.DataFrame) -> Dict:
        """Calculate user statistics"""
        user_stats = {}
        
        # Interactions per Пользователь
        user_interactions = events_df.groupby('visitorid').agg({
            'weight': 'sum',
            'event': 'count',
            'itemid': 'nunique'
        }).rename(columns={
            'weight': 'total_weight',
            'event': 'total_events',
            'itemid': 'unique_items'
        })
        
        # Преобразование to dictionary
        for user_id in user_interactions.index:
            user_stats[str(user_id)] = user_interactions.loc[user_id].to_dict()
        
        return user_stats
    
    def _calculate_item_stats(self, events_df: pd.DataFrame) -> Dict:
        """Calculate item statistics"""
        item_stats = {}
        
        # Interactions per Товар
        item_interactions = events_df.groupby('itemid').agg({
            'weight': 'sum',
            'event': 'count',
            'visitorid': 'nunique'
        }).rename(columns={
            'weight': 'total_weight',
            'event': 'total_events',
            'visitorid': 'unique_users'
        })
        
        # Расчет Популярность scores
        total_weight = item_interactions['total_weight'].sum()
        item_interactions['popularity'] = item_interactions['total_weight'] / total_weight
        
        # Преобразование to dictionary
        for item_id in item_interactions.index:
            item_stats[str(item_id)] = item_interactions.loc[item_id].to_dict()
        
        return item_stats
    
    def get_user_id(self, user_idx: int) -> str:
        """Get original user ID from encoded index"""
        return self.user_encoder.classes_[user_idx]
    
    def get_item_id(self, item_idx: int) -> str:
        """Get original item ID from encoded index"""
        return self.item_encoder.classes_[item_idx]
    
    def get_user_idx(self, user_id: str) -> int:
        """Get encoded index from original user ID"""
        try:
            return self.user_encoder.transform([str(user_id)])[0]
        except ValueError:
            return -1
    
    def get_item_idx(self, item_id: str) -> int:
        """Get encoded index from original item ID"""
        try:
            return self.item_encoder.transform([str(item_id)])[0]
        except ValueError:
            return -1
    
    def create_train_test_split(self, 
                               events_df: pd.DataFrame,
                               test_size: float = 0.2,
                               method: str = 'temporal') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split
        
        Args:
            events_df: Events DataFrame
            test_size: Fraction for test set
            method: Split method ('temporal', 'random', 'user_based')
            
        Returns:
            Tuple of (train_df, test_df)
        """
        print(f"📊 Создание train/test разделения методом '{method}'...")
        
        if method == 'temporal':
            # Сортировка by timestamp if available
            if 'timestamp' in events_df.columns:
                events_sorted = events_df.sort_values('timestamp')
            else:
                events_sorted = events_df.copy()
            
            # Разделение by time
            split_idx = int(len(events_sorted) * (1 - test_size))
            train_df = events_sorted.iloc[:split_idx]
            test_df = events_sorted.iloc[split_idx:]
            
        elif method == 'random':
            # Случайная Разделение
            test_df = events_df.sample(frac=test_size, random_state=42)
            train_df = events_df.drop(test_df.index)
            
        elif method == 'user_based':
            # Разделение users
            unique_users = events_df['visitorid'].unique()
            test_users = np.random.choice(
                unique_users, 
                size=int(len(unique_users) * test_size), 
                replace=False
            )
            
            test_df = events_df[events_df['visitorid'].isin(test_users)]
            train_df = events_df[~events_df['visitorid'].isin(test_users)]
            
        else:
            raise ValueError(f"Unknown split method: {method}")
        
        print(f"📊 Train: {len(train_df):,} событий, Test: {len(test_df):,} событий")
        
        return train_df, test_df
    
    def print_stats(self):
        """Print preprocessing statistics"""
        if not self.stats:
            print("⚠️ Статистика недоступна. Запустите fit_transform сначала.")
            return
        
        print("\n📊 Статистика предобработки:")
        print(f"  Исходные события: {self.stats.get('original_events', 0):,}")
        print(f"  После очистки: {self.stats.get('clean_events', 0):,}")
        print(f"  После фильтрации: {self.stats.get('filtered_events', 0):,}")
        print(f"  Уникальные пользователи: {self.stats.get('unique_users', 0):,}")
        print(f"  Активные пользователи: {self.stats.get('active_users', 0):,}")
        print(f"  Уникальные товары: {self.stats.get('unique_items', 0):,}")
        print(f"  Активные товары: {self.stats.get('active_items', 0):,}")
        
        if self.stats.get('original_events', 0) > 0:
            retention_rate = self.stats.get('filtered_events', 0) / self.stats.get('original_events', 1)
            print(f"  Коэффициент сохранения данных: {retention_rate:.2%}")


# Utility functions
def load_and_preprocess_data(
    events_path: str,
    item_properties_path: str,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    test_size: float = 0.2
) -> Tuple[csr_matrix, csr_matrix, Dict, Dict]:
    """
    Load and preprocess data for recommendation system
    
    Args:
        events_path: Path to events CSV
        item_properties_path: Path to item properties CSV
        min_user_interactions: Minimum interactions per user
        min_item_interactions: Minimum interactions per item
        test_size: Test set size
        
    Returns:
        Tuple of (train_matrix, test_matrix, train_metadata, test_metadata)
    """
    print("📁 Загрузка данных...")
    
    # Загрузка data
    events_df = pd.read_csv(events_path)
    item_properties_df = pd.read_csv(item_properties_path)
    
    print(f"📊 Загружено {len(events_df):,} событий и {len(item_properties_df):,} свойств товаров")
    
    # Инициализация preprocessor
    preprocessor = DataPreprocessor(
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions
    )
    
    # Создание Обучение/Тестирование Разделение
    train_events, test_events = preprocessor.create_train_test_split(
        events_df, test_size=test_size, method='temporal'
    )
    
    # Обучение on Обучение data
    train_matrix, train_metadata = preprocessor.fit_transform(train_events, item_properties_df)
    
    # Преобразование Тестирование data
    test_matrix = preprocessor.transform(test_events)
    test_metadata = train_metadata.copy()  # Use same metadata
    
    # Print statistics
    preprocessor.print_stats()
    
    return train_matrix, test_matrix, train_metadata, test_metadata


# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    
    # Создание synthetic events data
    n_events = 10000
    events_data = {
        'visitorid': np.random.randint(1, 1000, n_events),
        'itemid': np.random.randint(1, 500, n_events),
        'event': np.random.choice(['view', 'addtocart', 'transaction'], n_events, p=[0.8, 0.15, 0.05])
    }
    events_df = pd.DataFrame(events_data)
    
    # Создание synthetic Товар Свойства
    n_items = 500
    properties_data = {
        'itemid': np.repeat(range(1, n_items + 1), 3),
        'property': ['category', 'brand', 'price'] * n_items,
        'value': np.random.choice(['electronics', 'clothing', 'books'], n_items * 3)
    }
    item_properties_df = pd.DataFrame(properties_data)
    
    # Тестирование preprocessor
    preprocessor = DataPreprocessor()
    user_item_matrix, metadata = preprocessor.fit_transform(events_df, item_properties_df)
    
    print(f"\n✅ Создана матрица размером: {user_item_matrix.shape}")
    print(f"✅ Метаданные содержат: {list(metadata.keys())}")
    
    preprocessor.print_stats()
