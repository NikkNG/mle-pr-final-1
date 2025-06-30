"""
Metrics for recommendation systems evaluation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')


class RecommendationMetrics:
    """
    Class for calculating recommendation system metrics
    """
    
    def __init__(self):
        self.supported_metrics = [
            'precision_at_k', 'recall_at_k', 'ndcg_at_k', 'map_at_k',
            'coverage', 'diversity', 'novelty', 'auc_roc'
        ]
    
    def precision_at_k(self, y_true: List[List[int]], y_pred: List[List[int]], k: int = 10) -> float:
        """
        Calculate Precision@K
        
        Args:
            y_true: List of lists with true relevant items for each user
            y_pred: List of lists with predicted items for each user
            k: Number of top recommendations to consider
            
        Returns:
            Average Precision@K across all users
        """
        precisions = []
        
        for true_items, pred_items in zip(y_true, y_pred):
            if len(pred_items) == 0:
                precisions.append(0.0)
                continue
                
            # Take top-k predictions
            top_k_pred = pred_items[:k]
            
            # Расчет Точность
            relevant_in_top_k = len(set(true_items) & set(top_k_pred))
            precision = relevant_in_top_k / min(len(top_k_pred), k)
            precisions.append(precision)
        
        return np.mean(precisions)
    
    def recall_at_k(self, y_true: List[List[int]], y_pred: List[List[int]], k: int = 10) -> float:
        """
        Calculate Recall@K
        
        Args:
            y_true: List of lists with true relevant items for each user
            y_pred: List of lists with predicted items for each user
            k: Number of top recommendations to consider
            
        Returns:
            Average Recall@K across all users
        """
        recalls = []
        
        for true_items, pred_items in zip(y_true, y_pred):
            if len(true_items) == 0:
                continue
                
            # Take top-k predictions
            top_k_pred = pred_items[:k]
            
            # Расчет Полнота
            relevant_in_top_k = len(set(true_items) & set(top_k_pred))
            recall = relevant_in_top_k / len(true_items)
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def ndcg_at_k(self, y_true: List[List[int]], y_pred: List[List[int]], k: int = 10) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain)
        
        Args:
            y_true: List of lists with true relevant items for each user
            y_pred: List of lists with predicted items for each user
            k: Number of top recommendations to consider
            
        Returns:
            Average NDCG@K across all users
        """
        ndcgs = []
        
        for true_items, pred_items in zip(y_true, y_pred):
            if len(true_items) == 0:
                continue
                
            # Take top-k predictions
            top_k_pred = pred_items[:k]
            
            # Расчет DCG
            dcg = 0.0
            for i, item in enumerate(top_k_pred):
                if item in true_items:
                    dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
            
            # Расчет IDCG (ideal DCG)
            idcg = 0.0
            for i in range(min(len(true_items), k)):
                idcg += 1.0 / np.log2(i + 2)
            
            # Расчет NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0
    
    def map_at_k(self, y_true: List[List[int]], y_pred: List[List[int]], k: int = 10) -> float:
        """
        Calculate MAP@K (Mean Average Precision)
        
        Args:
            y_true: List of lists with true relevant items for each user
            y_pred: List of lists with predicted items for each user
            k: Number of top recommendations to consider
            
        Returns:
            Mean Average Precision@K
        """
        aps = []
        
        for true_items, pred_items in zip(y_true, y_pred):
            if len(true_items) == 0:
                continue
                
            # Take top-k predictions
            top_k_pred = pred_items[:k]
            
            # Расчет Average Точность
            ap = 0.0
            relevant_count = 0
            
            for i, item in enumerate(top_k_pred):
                if item in true_items:
                    relevant_count += 1
                    precision_at_i = relevant_count / (i + 1)
                    ap += precision_at_i
            
            if relevant_count > 0:
                ap /= min(len(true_items), k)
            
            aps.append(ap)
        
        return np.mean(aps) if aps else 0.0
    
    def coverage(self, y_pred: List[List[int]], total_items: int) -> float:
        """
        Calculate catalog coverage
        
        Args:
            y_pred: List of lists with predicted items for each user
            total_items: Total number of items in catalog
            
        Returns:
            Coverage percentage
        """
        recommended_items = set()
        for pred_items in y_pred:
            recommended_items.update(pred_items)
        
        return len(recommended_items) / total_items
    
    def diversity(self, y_pred: List[List[int]], item_features: Optional[np.ndarray] = None) -> float:
        """
        Calculate intra-list diversity
        
        Args:
            y_pred: List of lists with predicted items for each user
            item_features: Feature matrix for items (optional)
            
        Returns:
            Average intra-list diversity
        """
        if item_features is None:
            # Simple diversity based on unique items in recommendations
            diversities = []
            for pred_items in y_pred:
                if len(pred_items) > 1:
                    diversity = len(set(pred_items)) / len(pred_items)
                    diversities.append(diversity)
            return np.mean(diversities) if diversities else 0.0
        
        # Признак-based diversity
        from sklearn.metrics.pairwise import cosine_similarity
        
        diversities = []
        for pred_items in y_pred:
            if len(pred_items) > 1:
                # Get features for recommended items
                item_features_subset = item_features[pred_items]
                
                # Расчет pairwise similarities
                similarities = cosine_similarity(item_features_subset)
                
                # Расчет average diversity (1 - Схожесть)
                n_items = len(pred_items)
                total_similarity = 0
                count = 0
                
                for i in range(n_items):
                    for j in range(i + 1, n_items):
                        total_similarity += similarities[i, j]
                        count += 1
                
                avg_similarity = total_similarity / count if count > 0 else 0
                diversity = 1 - avg_similarity
                diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def category_diversity(self, y_pred: List[List[int]], item_categories: Dict[int, int]) -> float:
        """
        Calculate category diversity in recommendations
        
        Args:
            y_pred: List of lists with predicted items for each user
            item_categories: Dictionary mapping item_id to category_id
            
        Returns:
            Average number of unique categories per recommendation list
        """
        category_counts = []
        
        for pred_items in y_pred:
            categories = set()
            for item in pred_items:
                if item in item_categories:
                    categories.add(item_categories[item])
            category_counts.append(len(categories))
        
        return np.mean(category_counts) if category_counts else 0.0
    
    def novelty(self, y_pred: List[List[int]], item_popularity: Dict[int, float]) -> float:
        """
        Calculate novelty of recommendations
        
        Args:
            y_pred: List of lists with predicted items for each user
            item_popularity: Dictionary mapping item_id to popularity score
            
        Returns:
            Average novelty score (higher = more novel)
        """
        novelties = []
        
        for pred_items in y_pred:
            item_novelties = []
            for item in pred_items:
                if item in item_popularity:
                    # Novelty = -Логирование(Популярность)
                    novelty = -np.log(item_popularity[item] + 1e-10)
                    item_novelties.append(novelty)
            
            if item_novelties:
                novelties.append(np.mean(item_novelties))
        
        return np.mean(novelties) if novelties else 0.0
    
    def calculate_all_metrics(
        self, 
        y_true: List[List[int]], 
        y_pred: List[List[int]], 
        k_values: List[int] = [5, 10, 20],
        total_items: Optional[int] = None,
        item_categories: Optional[Dict[int, int]] = None,
        item_popularity: Optional[Dict[int, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate all available metrics
        
        Args:
            y_true: List of lists with true relevant items for each user
            y_pred: List of lists with predicted items for each user
            k_values: List of k values to calculate metrics for
            total_items: Total number of items in catalog
            item_categories: Dictionary mapping item_id to category_id
            item_popularity: Dictionary mapping item_id to popularity score
            
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}
        
        # Расчет ranking metrics for different k values
        for k in k_values:
            metrics[f'precision_at_{k}'] = self.precision_at_k(y_true, y_pred, k)
            metrics[f'recall_at_{k}'] = self.recall_at_k(y_true, y_pred, k)
            metrics[f'ndcg_at_{k}'] = self.ndcg_at_k(y_true, y_pred, k)
            metrics[f'map_at_{k}'] = self.map_at_k(y_true, y_pred, k)
        
        # Расчет coverage if total_items provided
        if total_items is not None:
            metrics['coverage'] = self.coverage(y_pred, total_items)
        
        # Расчет diversity metrics
        metrics['diversity'] = self.diversity(y_pred)
        
        if item_categories is not None:
            metrics['category_diversity'] = self.category_diversity(y_pred, item_categories)
        
        if item_popularity is not None:
            metrics['novelty'] = self.novelty(y_pred, item_popularity)
        
        return metrics


def evaluate_recommendations(
    user_item_matrix: csr_matrix,
    recommendations: Dict[int, List[int]],
    test_interactions: Dict[int, List[int]],
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Evaluate recommendation system performance
    
    Args:
        user_item_matrix: Sparse matrix with user-item interactions
        recommendations: Dictionary mapping user_id to list of recommended items
        test_interactions: Dictionary mapping user_id to list of test items
        k_values: List of k values for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics_calculator = RecommendationMetrics()
    
    # Prepare data for evaluation
    y_true = []
    y_pred = []
    
    for user_id in recommendations.keys():
        if user_id in test_interactions:
            y_true.append(test_interactions[user_id])
            y_pred.append(recommendations[user_id])
    
    # Расчет metrics
    metrics = metrics_calculator.calculate_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        k_values=k_values,
        total_items=user_item_matrix.shape[1]
    )
    
    return metrics


def create_popularity_dict(interactions_df: pd.DataFrame) -> Dict[int, float]:
    """
    Create item popularity dictionary
    
    Args:
        interactions_df: DataFrame with user-item interactions
        
    Returns:
        Dictionary mapping item_id to popularity score
    """
    item_counts = interactions_df['itemid'].value_counts()
    total_interactions = len(interactions_df)
    
    popularity = {}
    for item_id, count in item_counts.items():
        popularity[item_id] = count / total_interactions
    
    return popularity


def create_category_dict(item_properties_df: pd.DataFrame) -> Dict[int, int]:
    """
    Create item to category mapping
    
    Args:
        item_properties_df: DataFrame with item properties
        
    Returns:
        Dictionary mapping item_id to category_id
    """
    # Find Категория information in Свойства
    category_data = item_properties_df[
        item_properties_df['property'].str.contains('category', case=False, na=False)
    ]
    
    category_dict = {}
    for _, row in category_data.iterrows():
        try:
            category_dict[row['itemid']] = int(row['value'])
        except (ValueError, TypeError):
            continue
    
    return category_dict


# Example usage and Тестирование
if __name__ == "__main__":
    # Example data for Тестирование
    y_true_example = [
        [1, 2, 3],      # User 1 relevant items
        [4, 5],         # User 2 relevant items
        [6, 7, 8, 9]    # User 3 relevant items
    ]
    
    y_pred_example = [
        [1, 10, 2, 11, 12],  # User 1 recommendations
        [4, 13, 14, 5, 15],  # User 2 recommendations
        [6, 16, 7, 17, 8]    # User 3 recommendations
    ]
    
    # Инициализация metrics calculator
    metrics_calc = RecommendationMetrics()
    
    # Расчет metrics
    results = metrics_calc.calculate_all_metrics(
        y_true=y_true_example,
        y_pred=y_pred_example,
        k_values=[3, 5],
        total_items=20
    )
    
    print("Example metrics calculation:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}") 