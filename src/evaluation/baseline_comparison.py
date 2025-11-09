"""
Baseline Comparison Module

Compares the hybrid recommendation system against baseline methods:
- Random baseline
- TF-IDF only
- BM25 only
- Hybrid (BM25 + Semantic)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
import logging
import random
import os
import tempfile

from src.core.recommender_engine import ProductRecommender

logger = logging.getLogger(__name__)


class BaselineRecommender:
    """Base class for baseline recommendation methods."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def recommend(self, product_query: str, top_k: int = 10) -> List[int]:
        """
        Get recommendations.
        
        Args:
            product_query: Product query string
            top_k: Number of recommendations
            
        Returns:
            List of product indices
        """
        raise NotImplementedError


class RandomBaseline(BaselineRecommender):
    """Random baseline - returns random products."""
    
    def recommend(self, product_query: str, top_k: int = 10) -> List[int]:
        """Return random products."""
        all_indices = list(self.df.index)
        return random.sample(all_indices, min(top_k, len(all_indices)))


class TFIDFBaseline(BaselineRecommender):
    """TF-IDF only baseline."""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        # Create temporary CSV file for initialization
        temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_csv.name, index=False)
        temp_csv.close()
        
        try:
            self.recommender = ProductRecommender(
                csv_path=temp_csv.name,
                use_ngrams=False,
                use_advanced_preprocessing=False,
                use_genai=False,  # No GenAI
                load_models_immediately=True
            )
            # Disable BM25
            self.recommender.bm25 = None
            self.recommender.use_genai = False
        finally:
            # Clean up temp file
            if os.path.exists(temp_csv.name):
                os.unlink(temp_csv.name)
    
    def recommend(self, product_query: str, top_k: int = 10) -> List[int]:
        """Get recommendations using TF-IDF only."""
        results = self.recommender.recommend(product_query, top_k=top_k)
        if isinstance(results, list) and results and isinstance(results[0], dict):
            # Extract indices from results
            indices = []
            for item in results[:top_k]:
                # Find index in dataframe
                mask = (
                    (self.df['product_name'].astype(str) == item.get('name', '')) &
                    (self.df['brand'].astype(str) == item.get('brand', ''))
                )
                matches = self.df[mask].index.tolist()
                if matches:
                    indices.append(matches[0])
            return indices[:top_k]
        return []


class BM25Baseline(BaselineRecommender):
    """BM25 only baseline."""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        # Create temporary CSV file for initialization
        temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_csv.name, index=False)
        temp_csv.close()
        
        try:
            self.recommender = ProductRecommender(
                csv_path=temp_csv.name,
                use_ngrams=False,
                use_advanced_preprocessing=False,
                use_genai=False,  # No GenAI
                load_models_immediately=True
            )
            # Disable TF-IDF
            self.recommender.vectorizer = None
            self.recommender.tfidf_matrix = None
            self.recommender.use_genai = False
        finally:
            # Clean up temp file
            if os.path.exists(temp_csv.name):
                os.unlink(temp_csv.name)
    
    def recommend(self, product_query: str, top_k: int = 10) -> List[int]:
        """Get recommendations using BM25 only."""
        results = self.recommender.recommend(product_query, top_k=top_k)
        if isinstance(results, list) and results and isinstance(results[0], dict):
            # Extract indices from results
            indices = []
            for item in results[:top_k]:
                # Find index in dataframe
                mask = (
                    (self.df['product_name'].astype(str) == item.get('name', '')) &
                    (self.df['brand'].astype(str) == item.get('brand', ''))
                )
                matches = self.df[mask].index.tolist()
                if matches:
                    indices.append(matches[0])
            return indices[:top_k]
        return []


class HybridRecommender(BaselineRecommender):
    """Hybrid (BM25 + Semantic) recommender."""
    
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        # Create temporary CSV file for initialization
        temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_csv.name, index=False)
        temp_csv.close()
        
        try:
            self.recommender = ProductRecommender(
                csv_path=temp_csv.name,
                use_ngrams=True,
                use_advanced_preprocessing=True,
                use_genai=True,  # Enable GenAI
                load_models_immediately=True
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_csv.name):
                os.unlink(temp_csv.name)
    
    def recommend(self, product_query: str, top_k: int = 10) -> List[int]:
        """Get recommendations using hybrid approach."""
        results = self.recommender.recommend(product_query, top_k=top_k)
        if isinstance(results, list) and results and isinstance(results[0], dict):
            # Extract indices from results
            indices = []
            for item in results[:top_k]:
                # Find index in dataframe
                mask = (
                    (self.df['product_name'].astype(str) == item.get('name', '')) &
                    (self.df['brand'].astype(str) == item.get('brand', ''))
                )
                matches = self.df[mask].index.tolist()
                if matches:
                    indices.append(matches[0])
            return indices[:top_k]
        return []


def compare_baselines(
    test_queries: List[str],
    df: pd.DataFrame,
    ground_truth: Optional[Dict[str, Set[int]]] = None,
    top_k: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Compare different baseline methods.
    
    Args:
        test_queries: List of test queries
        df: Product dataframe
        ground_truth: Dictionary mapping queries to sets of relevant item indices
        top_k: Number of recommendations to generate
        
    Returns:
        Dictionary of results for each baseline method
    """
    results = {}
    
    # Initialize baseline recommenders
    baselines = {
        'Random': RandomBaseline(df),
        'TF-IDF Only': TFIDFBaseline(df),
        'BM25 Only': BM25Baseline(df),
        'Hybrid (BM25 + Semantic)': HybridRecommender(df)
    }
    
    for baseline_name, recommender in baselines.items():
        logger.info(f"Evaluating {baseline_name}...")
        baseline_results = []
        
        for query in test_queries:
            try:
                recommendations = recommender.recommend(query, top_k=top_k)
                baseline_results.append({
                    'query': query,
                    'recommendations': recommendations
                })
            except Exception as e:
                logger.warning(f"Error with {baseline_name} for query '{query}': {e}")
                baseline_results.append({
                    'query': query,
                    'recommendations': []
                })
        
        results[baseline_name] = {
            'recommendations': baseline_results
        }
    
    return results


def evaluate_baselines_with_metrics(
    test_queries: List[str],
    df: pd.DataFrame,
    ground_truth: Dict[str, Set[int]],
    top_k: int = 10,
    k_values: List[int] = [5, 10]
) -> Dict[str, Dict[str, float]]:
    """
    Compare baselines with evaluation metrics.
    
    Args:
        test_queries: List of test queries
        df: Product dataframe
        ground_truth: Dictionary mapping queries to sets of relevant item indices
        top_k: Number of recommendations to generate
        k_values: List of K values for evaluation
        
    Returns:
        Dictionary of metric results for each baseline
    """
    from src.evaluation.evaluation_metrics import evaluate_recommendations
    
    # Get recommendations from all baselines
    baseline_results = compare_baselines(test_queries, df, ground_truth, top_k)
    
    # Evaluate each baseline
    evaluation_results = {}
    
    for baseline_name, baseline_data in baseline_results.items():
        logger.info(f"Evaluating metrics for {baseline_name}...")
        
        all_metrics = []
        for rec_data in baseline_data['recommendations']:
            query = rec_data['query']
            recommendations = rec_data['recommendations']
            relevant_items = ground_truth.get(query, set())
            
            if recommendations:
                metrics = evaluate_recommendations(recommendations, relevant_items, k_values)
                all_metrics.append(metrics)
        
        # Average metrics across all queries
        if all_metrics:
            avg_metrics = {}
            for k in k_values:
                avg_metrics[f'Precision@{k}'] = np.mean([m[f'Precision@{k}'] for m in all_metrics])
                avg_metrics[f'Recall@{k}'] = np.mean([m[f'Recall@{k}'] for m in all_metrics])
                avg_metrics[f'NDCG@{k}'] = np.mean([m[f'NDCG@{k}'] for m in all_metrics])
            avg_metrics['MAP'] = np.mean([m['MAP'] for m in all_metrics])
            
            evaluation_results[baseline_name] = avg_metrics
        else:
            evaluation_results[baseline_name] = {}
    
    return evaluation_results

