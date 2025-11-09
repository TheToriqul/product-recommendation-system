"""
Parameter Tuning Module

Demonstrates hyperparameter tuning for:
- BM25 parameters (k1, b)
- Hybrid search weights (BM25 vs Semantic)
- Feature weights (product_name vs brand)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
import logging
from itertools import product

from src.core.recommender_engine import ProductRecommender, BM25
from src.evaluation.evaluation_metrics import evaluate_recommendations

logger = logging.getLogger(__name__)


def tune_bm25_parameters(
    corpus: List[str],
    test_queries: List[str],
    ground_truth: Dict[str, Set[int]],
    k1_range: List[float] = [1.0, 1.2, 1.5, 2.0],
    b_range: List[float] = [0.5, 0.75, 1.0]
) -> Dict[Tuple[float, float], Dict[str, float]]:
    """
    Tune BM25 parameters (k1 and b).
    
    Args:
        corpus: List of document strings
        test_queries: List of test queries
        ground_truth: Dictionary mapping queries to sets of relevant item indices
        k1_range: Range of k1 values to test
        b_range: Range of b values to test
        
    Returns:
        Dictionary mapping (k1, b) tuples to evaluation metrics
    """
    results = {}
    
    for k1, b in product(k1_range, b_range):
        logger.info(f"Testing BM25 parameters: k1={k1}, b={b}")
        
        # Initialize BM25 with these parameters
        bm25 = BM25(corpus, k1=k1, b=b)
        
        # Evaluate on test queries
        all_metrics = []
        for query in test_queries:
            # Get BM25 scores
            scores = bm25.get_scores(query)
            # Get top indices
            top_indices = scores.argsort()[-10:][::-1].tolist()
            
            # Get relevant items
            relevant_items = ground_truth.get(query, set())
            
            if relevant_items:
                metrics = evaluate_recommendations(top_indices, relevant_items, k_values=[5, 10])
                all_metrics.append(metrics)
        
        if all_metrics:
            # Average metrics
            avg_metrics = {
                'Precision@5': np.mean([m['Precision@5'] for m in all_metrics]),
                'Precision@10': np.mean([m['Precision@10'] for m in all_metrics]),
                'Recall@5': np.mean([m['Recall@5'] for m in all_metrics]),
                'Recall@10': np.mean([m['Recall@10'] for m in all_metrics]),
                'NDCG@5': np.mean([m['NDCG@5'] for m in all_metrics]),
                'NDCG@10': np.mean([m['NDCG@10'] for m in all_metrics]),
                'MAP': np.mean([m['MAP'] for m in all_metrics])
            }
            results[(k1, b)] = avg_metrics
    
    return results


def tune_hybrid_weights(
    recommender: ProductRecommender,
    test_queries: List[str],
    ground_truth: Dict[str, Set[int]],
    bm25_weight_range: List[float] = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    top_k: int = 10
) -> Dict[float, Dict[str, float]]:
    """
    Tune hybrid search weights (BM25 vs Semantic).
    
    Args:
        recommender: ProductRecommender instance (must have GenAI enabled)
        test_queries: List of test queries
        ground_truth: Dictionary mapping queries to sets of relevant item indices
        bm25_weight_range: Range of BM25 weights to test (semantic weight = 1 - bm25_weight)
        top_k: Number of recommendations
        
    Returns:
        Dictionary mapping BM25 weight to evaluation metrics
    """
    if not recommender.use_genai or recommender.genai_embeddings is None:
        logger.warning("GenAI not available, cannot tune hybrid weights")
        return {}
    
    results = {}
    original_bm25_weight = recommender.hybrid_weight_bm25
    original_semantic_weight = recommender.hybrid_weight_semantic
    
    for bm25_weight in bm25_weight_range:
        semantic_weight = 1.0 - bm25_weight
        
        logger.info(f"Testing hybrid weights: BM25={bm25_weight:.2f}, Semantic={semantic_weight:.2f}")
        
        # Set weights
        recommender.hybrid_weight_bm25 = bm25_weight
        recommender.hybrid_weight_semantic = semantic_weight
        
        # Evaluate on test queries
        all_metrics = []
        for query in test_queries:
            try:
                # Get recommendations
                rec_results = recommender.recommend(query, top_k=top_k)
                
                if isinstance(rec_results, list) and rec_results and isinstance(rec_results[0], dict):
                    # Extract indices
                    recommendations = []
                    for item in rec_results[:top_k]:
                        mask = (
                            (recommender.df['product_name'].astype(str) == item.get('name', '')) &
                            (recommender.df['brand'].astype(str) == item.get('brand', ''))
                        )
                        matches = recommender.df[mask].index.tolist()
                        if matches:
                            recommendations.append(matches[0])
                    
                    # Get relevant items
                    relevant_items = ground_truth.get(query, set())
                    
                    if relevant_items and recommendations:
                        metrics = evaluate_recommendations(recommendations, relevant_items, k_values=[5, 10])
                        all_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Error evaluating query '{query}': {e}")
        
        if all_metrics:
            # Average metrics
            avg_metrics = {
                'Precision@5': np.mean([m['Precision@5'] for m in all_metrics]),
                'Precision@10': np.mean([m['Precision@10'] for m in all_metrics]),
                'Recall@5': np.mean([m['Recall@5'] for m in all_metrics]),
                'Recall@10': np.mean([m['Recall@10'] for m in all_metrics]),
                'NDCG@5': np.mean([m['NDCG@5'] for m in all_metrics]),
                'NDCG@10': np.mean([m['NDCG@10'] for m in all_metrics]),
                'MAP': np.mean([m['MAP'] for m in all_metrics])
            }
            results[bm25_weight] = avg_metrics
    
    # Restore original weights
    recommender.hybrid_weight_bm25 = original_bm25_weight
    recommender.hybrid_weight_semantic = original_semantic_weight
    
    return results


def tune_feature_weights(
    recommender: ProductRecommender,
    test_queries: List[str],
    ground_truth: Dict[str, Set[int]],
    name_weight_range: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
    top_k: int = 10
) -> Dict[float, Dict[str, float]]:
    """
    Tune feature weights (product_name vs brand).
    
    Args:
        recommender: ProductRecommender instance
        test_queries: List of test queries
        ground_truth: Dictionary mapping queries to sets of relevant item indices
        name_weight_range: Range of product_name weights to test (brand weight = 1 - name_weight)
        top_k: Number of recommendations
        
    Returns:
        Dictionary mapping product_name weight to evaluation metrics
    """
    results = {}
    original_weights = recommender.feature_weights.copy()
    
    for name_weight in name_weight_range:
        brand_weight = 1.0 - name_weight
        
        logger.info(f"Testing feature weights: product_name={name_weight:.2f}, brand={brand_weight:.2f}")
        
        # Set weights
        recommender.feature_weights = {
            'product_name': name_weight,
            'brand': brand_weight
        }
        
        # Reinitialize vectorizer with new weights
        recommender._preprocess_text_data()
        recommender._initialize_vectorizer()
        
        # Evaluate on test queries
        all_metrics = []
        for query in test_queries:
            try:
                # Get recommendations
                rec_results = recommender.recommend(query, top_k=top_k)
                
                if isinstance(rec_results, list) and rec_results and isinstance(rec_results[0], dict):
                    # Extract indices
                    recommendations = []
                    for item in rec_results[:top_k]:
                        mask = (
                            (recommender.df['product_name'].astype(str) == item.get('name', '')) &
                            (recommender.df['brand'].astype(str) == item.get('brand', ''))
                        )
                        matches = recommender.df[mask].index.tolist()
                        if matches:
                            recommendations.append(matches[0])
                    
                    # Get relevant items
                    relevant_items = ground_truth.get(query, set())
                    
                    if relevant_items and recommendations:
                        metrics = evaluate_recommendations(recommendations, relevant_items, k_values=[5, 10])
                        all_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Error evaluating query '{query}': {e}")
        
        if all_metrics:
            # Average metrics
            avg_metrics = {
                'Precision@5': np.mean([m['Precision@5'] for m in all_metrics]),
                'Precision@10': np.mean([m['Precision@10'] for m in all_metrics]),
                'Recall@5': np.mean([m['Recall@5'] for m in all_metrics]),
                'Recall@10': np.mean([m['Recall@10'] for m in all_metrics]),
                'NDCG@5': np.mean([m['NDCG@5'] for m in all_metrics]),
                'NDCG@10': np.mean([m['NDCG@10'] for m in all_metrics]),
                'MAP': np.mean([m['MAP'] for m in all_metrics])
            }
            results[name_weight] = avg_metrics
    
    # Restore original weights
    recommender.feature_weights = original_weights
    recommender._preprocess_text_data()
    recommender._initialize_vectorizer()
    
    return results


def find_best_parameters(
    bm25_results: Optional[Dict[Tuple[float, float], Dict[str, float]]] = None,
    hybrid_results: Optional[Dict[float, Dict[str, float]]] = None,
    feature_results: Optional[Dict[float, Dict[str, float]]] = None,
    metric: str = 'NDCG@10'
) -> Dict[str, any]:
    """
    Find best parameters based on a metric.
    
    Args:
        bm25_results: Results from tune_bm25_parameters
        hybrid_results: Results from tune_hybrid_weights
        feature_results: Results from tune_feature_weights
        metric: Metric to optimize (e.g., 'NDCG@10', 'MAP', 'Precision@10')
        
    Returns:
        Dictionary with best parameters
    """
    best = {}
    
    if bm25_results:
        best_bm25 = max(bm25_results.items(), key=lambda x: x[1].get(metric, 0))
        best['bm25'] = {
            'k1': best_bm25[0][0],
            'b': best_bm25[0][1],
            'score': best_bm25[1].get(metric, 0)
        }
    
    if hybrid_results:
        best_hybrid = max(hybrid_results.items(), key=lambda x: x[1].get(metric, 0))
        best['hybrid'] = {
            'bm25_weight': best_hybrid[0],
            'semantic_weight': 1.0 - best_hybrid[0],
            'score': best_hybrid[1].get(metric, 0)
        }
    
    if feature_results:
        best_feature = max(feature_results.items(), key=lambda x: x[1].get(metric, 0))
        best['feature'] = {
            'product_name_weight': best_feature[0],
            'brand_weight': 1.0 - best_feature[0],
            'score': best_feature[1].get(metric, 0)
        }
    
    return best


def generate_tuning_report(
    bm25_results: Optional[Dict[Tuple[float, float], Dict[str, float]]] = None,
    hybrid_results: Optional[Dict[float, Dict[str, float]]] = None,
    feature_results: Optional[Dict[float, Dict[str, float]]] = None
) -> str:
    """
    Generate human-readable parameter tuning report.
    
    Args:
        bm25_results: Results from tune_bm25_parameters
        hybrid_results: Results from tune_hybrid_weights
        feature_results: Results from tune_feature_weights
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("PARAMETER TUNING REPORT")
    report.append("=" * 60)
    report.append("")
    
    if bm25_results:
        report.append("BM25 PARAMETER TUNING:")
        report.append("-" * 60)
        # Sort by NDCG@10
        sorted_results = sorted(
            bm25_results.items(),
            key=lambda x: x[1].get('NDCG@10', 0),
            reverse=True
        )
        for (k1, b), metrics in sorted_results[:5]:  # Top 5
            report.append(f"k1={k1:.2f}, b={b:.2f}:")
            report.append(f"  NDCG@10: {metrics.get('NDCG@10', 0):.4f}")
            report.append(f"  MAP: {metrics.get('MAP', 0):.4f}")
            report.append(f"  Precision@10: {metrics.get('Precision@10', 0):.4f}")
            report.append("")
    
    if hybrid_results:
        report.append("HYBRID WEIGHT TUNING:")
        report.append("-" * 60)
        # Sort by NDCG@10
        sorted_results = sorted(
            hybrid_results.items(),
            key=lambda x: x[1].get('NDCG@10', 0),
            reverse=True
        )
        for bm25_weight, metrics in sorted_results[:5]:  # Top 5
            semantic_weight = 1.0 - bm25_weight
            report.append(f"BM25={bm25_weight:.2f}, Semantic={semantic_weight:.2f}:")
            report.append(f"  NDCG@10: {metrics.get('NDCG@10', 0):.4f}")
            report.append(f"  MAP: {metrics.get('MAP', 0):.4f}")
            report.append(f"  Precision@10: {metrics.get('Precision@10', 0):.4f}")
            report.append("")
    
    if feature_results:
        report.append("FEATURE WEIGHT TUNING:")
        report.append("-" * 60)
        # Sort by NDCG@10
        sorted_results = sorted(
            feature_results.items(),
            key=lambda x: x[1].get('NDCG@10', 0),
            reverse=True
        )
        for name_weight, metrics in sorted_results[:5]:  # Top 5
            brand_weight = 1.0 - name_weight
            report.append(f"product_name={name_weight:.2f}, brand={brand_weight:.2f}:")
            report.append(f"  NDCG@10: {metrics.get('NDCG@10', 0):.4f}")
            report.append(f"  MAP: {metrics.get('MAP', 0):.4f}")
            report.append(f"  Precision@10: {metrics.get('Precision@10', 0):.4f}")
            report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)

