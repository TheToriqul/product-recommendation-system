"""
Evaluation Metrics for Recommendation System

Implements standard recommendation system evaluation metrics:
- Precision@K
- Recall@K
- NDCG (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def precision_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """
    Calculate Precision@K.
    
    Precision@K = (Number of relevant items in top K) / K
    
    Args:
        recommended_items: List of recommended item indices (ordered by relevance)
        relevant_items: Set of relevant item indices (ground truth)
        k: Number of top recommendations to consider
        
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    
    top_k = recommended_items[:k]
    relevant_in_top_k = sum(1 for item in top_k if item in relevant_items)
    return relevant_in_top_k / k


def recall_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """
    Calculate Recall@K.
    
    Recall@K = (Number of relevant items in top K) / (Total number of relevant items)
    
    Args:
        recommended_items: List of recommended item indices (ordered by relevance)
        relevant_items: Set of relevant item indices (ground truth)
        k: Number of top recommendations to consider
        
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if len(relevant_items) == 0:
        return 0.0
    
    top_k = recommended_items[:k]
    relevant_in_top_k = sum(1 for item in top_k if item in relevant_items)
    return relevant_in_top_k / len(relevant_items)


def dcg_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at K.
    
    DCG@K = sum(relevance_i / log2(i + 1)) for i in [1, k]
    where relevance_i = 1 if item is relevant, 0 otherwise
    
    Args:
        recommended_items: List of recommended item indices (ordered by relevance)
        relevant_items: Set of relevant item indices (ground truth)
        k: Number of top recommendations to consider
        
    Returns:
        DCG@K score
    """
    top_k = recommended_items[:k]
    dcg = 0.0
    for i, item in enumerate(top_k, start=1):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 1)
    return dcg


def ndcg_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.
    
    NDCG@K = DCG@K / IDCG@K
    where IDCG@K is the ideal DCG (all relevant items ranked first)
    
    Args:
        recommended_items: List of recommended item indices (ordered by relevance)
        relevant_items: Set of relevant item indices (ground truth)
        k: Number of top recommendations to consider
        
    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    dcg = dcg_at_k(recommended_items, relevant_items, k)
    
    # Calculate IDCG (Ideal DCG)
    num_relevant = min(len(relevant_items), k)
    if num_relevant == 0:
        return 0.0
    
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, num_relevant + 1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mean_average_precision(recommended_items: List[int], relevant_items: Set[int]) -> float:
    """
    Calculate Mean Average Precision (MAP).
    
    AP = (1 / |R|) * sum(Precision@i for each relevant item i)
    MAP = mean(AP) across all queries
    
    For a single query:
    MAP = Average of Precision@i for each position i where a relevant item appears
    
    Args:
        recommended_items: List of recommended item indices (ordered by relevance)
        relevant_items: Set of relevant item indices (ground truth)
        
    Returns:
        MAP score (0.0 to 1.0)
    """
    if len(relevant_items) == 0:
        return 0.0
    
    relevant_count = 0
    precision_sum = 0.0
    
    for i, item in enumerate(recommended_items, start=1):
        if item in relevant_items:
            relevant_count += 1
            precision_sum += relevant_count / i
    
    if relevant_count == 0:
        return 0.0
    
    return precision_sum / len(relevant_items)


def rmse(predicted_ratings: List[float], actual_ratings: List[float]) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    RMSE = sqrt(mean((predicted - actual)^2))
    
    Args:
        predicted_ratings: List of predicted ratings/scores
        actual_ratings: List of actual ratings/scores
        
    Returns:
        RMSE value (lower is better)
    """
    if len(predicted_ratings) != len(actual_ratings):
        raise ValueError("predicted_ratings and actual_ratings must have same length")
    
    if len(predicted_ratings) == 0:
        return 0.0
    
    squared_errors = [(p - a) ** 2 for p, a in zip(predicted_ratings, actual_ratings)]
    mse = np.mean(squared_errors)
    return np.sqrt(mse)


def mae(predicted_ratings: List[float], actual_ratings: List[float]) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    MAE = mean(|predicted - actual|)
    
    Args:
        predicted_ratings: List of predicted ratings/scores
        actual_ratings: List of actual ratings/scores
        
    Returns:
        MAE value (lower is better)
    """
    if len(predicted_ratings) != len(actual_ratings):
        raise ValueError("predicted_ratings and actual_ratings must have same length")
    
    if len(predicted_ratings) == 0:
        return 0.0
    
    absolute_errors = [abs(p - a) for p, a in zip(predicted_ratings, actual_ratings)]
    return np.mean(absolute_errors)


def evaluate_recommendations(
    recommended_items: List[int],
    relevant_items: Set[int],
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Evaluate recommendations using multiple metrics.
    
    Args:
        recommended_items: List of recommended item indices (ordered by relevance)
        relevant_items: Set of relevant item indices (ground truth)
        k_values: List of K values to evaluate (e.g., [5, 10, 20])
        
    Returns:
        Dictionary of metric scores
    """
    results = {}
    
    for k in k_values:
        results[f'Precision@{k}'] = precision_at_k(recommended_items, relevant_items, k)
        results[f'Recall@{k}'] = recall_at_k(recommended_items, relevant_items, k)
        results[f'NDCG@{k}'] = ndcg_at_k(recommended_items, relevant_items, k)
    
    results['MAP'] = mean_average_precision(recommended_items, relevant_items)
    
    return results


def evaluate_rating_predictions(
    predicted_ratings: List[float],
    actual_ratings: List[float]
) -> Dict[str, float]:
    """
    Evaluate rating predictions using RMSE and MAE.
    
    Args:
        predicted_ratings: List of predicted ratings/scores
        actual_ratings: List of actual ratings/scores
        
    Returns:
        Dictionary with RMSE and MAE scores
    """
    return {
        'RMSE': rmse(predicted_ratings, actual_ratings),
        'MAE': mae(predicted_ratings, actual_ratings)
    }

