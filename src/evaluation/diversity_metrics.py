"""
Diversity, Novelty, and Coverage Metrics for Recommendation System

Implements metrics to analyze:
- Diversity: How different are recommended items from each other
- Novelty: How unexpected/surprising are the recommendations
- Coverage: What percentage of the catalog can be recommended
"""

import numpy as np
from typing import List, Dict, Set, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def intra_list_diversity(
    recommended_items: List[int],
    item_features: Dict[int, np.ndarray],
    similarity_func=None
) -> float:
    """
    Calculate Intra-List Diversity (ILD).
    
    Measures how different items in the recommendation list are from each other.
    ILD = (1 / (|R| * (|R| - 1))) * sum(1 - similarity(i, j)) for all pairs i != j
    
    Args:
        recommended_items: List of recommended item indices
        item_features: Dictionary mapping item indices to feature vectors
        similarity_func: Function to calculate similarity (default: cosine similarity)
        
    Returns:
        ILD score (0.0 to 1.0, higher = more diverse)
    """
    if len(recommended_items) < 2:
        return 0.0
    
    if similarity_func is None:
        from sklearn.metrics.pairwise import cosine_similarity
        def similarity_func(vec1, vec2):
            return cosine_similarity([vec1], [vec2])[0][0]
    
    total_dissimilarity = 0.0
    pair_count = 0
    
    for i, item1 in enumerate(recommended_items):
        if item1 not in item_features:
            continue
        vec1 = item_features[item1]
        
        for j, item2 in enumerate(recommended_items):
            if i >= j or item2 not in item_features:
                continue
            vec2 = item_features[item2]
            
            similarity = similarity_func(vec1, vec2)
            dissimilarity = 1.0 - similarity
            total_dissimilarity += dissimilarity
            pair_count += 1
    
    if pair_count == 0:
        return 0.0
    
    return total_dissimilarity / pair_count


def category_diversity(recommended_items: List[int], item_categories: Dict[int, str]) -> float:
    """
    Calculate Category Diversity.
    
    Measures diversity based on item categories.
    Uses Shannon entropy: -sum(p(c) * log2(p(c))) where p(c) is proportion of category c
    
    Args:
        recommended_items: List of recommended item indices
        item_categories: Dictionary mapping item indices to category strings
        
    Returns:
        Category diversity score (higher = more diverse categories)
    """
    if len(recommended_items) == 0:
        return 0.0
    
    # Count categories in recommendations
    categories = [item_categories.get(item, 'unknown') for item in recommended_items]
    category_counts = Counter(categories)
    
    # Calculate proportions
    total = len(recommended_items)
    proportions = [count / total for count in category_counts.values()]
    
    # Calculate Shannon entropy
    entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
    
    # Normalize by maximum possible entropy (log2 of number of unique categories)
    max_entropy = np.log2(len(category_counts)) if len(category_counts) > 1 else 1.0
    
    if max_entropy == 0:
        return 0.0
    
    return entropy / max_entropy


def novelty(
    recommended_items: List[int],
    item_popularity: Dict[int, float]
) -> float:
    """
    Calculate Novelty (unexpectedness) of recommendations.
    
    Novelty = -mean(log2(popularity(item))) for all recommended items
    Lower popularity = higher novelty
    
    Args:
        recommended_items: List of recommended item indices
        item_popularity: Dictionary mapping item indices to popularity scores (0.0 to 1.0)
        
    Returns:
        Novelty score (higher = more novel/unexpected)
    """
    if len(recommended_items) == 0:
        return 0.0
    
    novelty_scores = []
    for item in recommended_items:
        popularity = item_popularity.get(item, 0.5)  # Default to medium popularity if unknown
        # Avoid log(0) by adding small epsilon
        popularity = max(popularity, 1e-10)
        novelty = -np.log2(popularity)
        novelty_scores.append(novelty)
    
    return np.mean(novelty_scores)


def catalog_coverage(
    all_recommendations: List[List[int]],
    total_catalog_size: int
) -> float:
    """
    Calculate Catalog Coverage.
    
    Coverage = (Number of unique items recommended) / (Total catalog size)
    
    Measures what percentage of the catalog can be recommended by the system.
    
    Args:
        all_recommendations: List of recommendation lists (one per query/user)
        total_catalog_size: Total number of items in the catalog
        
    Returns:
        Coverage score (0.0 to 1.0, higher = better coverage)
    """
    if total_catalog_size == 0:
        return 0.0
    
    unique_items = set()
    for recommendations in all_recommendations:
        unique_items.update(recommendations)
    
    return len(unique_items) / total_catalog_size


def brand_diversity(recommended_items: List[int], item_brands: Dict[int, str]) -> float:
    """
    Calculate Brand Diversity.
    
    Measures diversity based on item brands using Shannon entropy.
    
    Args:
        recommended_items: List of recommended item indices
        item_brands: Dictionary mapping item indices to brand strings
        
    Returns:
        Brand diversity score (higher = more diverse brands)
    """
    if len(recommended_items) == 0:
        return 0.0
    
    # Count brands in recommendations
    brands = [item_brands.get(item, 'unknown') for item in recommended_items]
    brand_counts = Counter(brands)
    
    # Calculate proportions
    total = len(recommended_items)
    proportions = [count / total for count in brand_counts.values()]
    
    # Calculate Shannon entropy
    entropy = -sum(p * np.log2(p) for p in proportions if p > 0)
    
    # Normalize by maximum possible entropy
    max_entropy = np.log2(len(brand_counts)) if len(brand_counts) > 1 else 1.0
    
    if max_entropy == 0:
        return 0.0
    
    return entropy / max_entropy


def evaluate_diversity_novelty_coverage(
    all_recommendations: List[List[int]],
    item_features: Optional[Dict[int, np.ndarray]] = None,
    item_categories: Optional[Dict[int, str]] = None,
    item_brands: Optional[Dict[int, str]] = None,
    item_popularity: Optional[Dict[int, float]] = None,
    total_catalog_size: Optional[int] = None
) -> Dict[str, float]:
    """
    Comprehensive evaluation of diversity, novelty, and coverage.
    
    Args:
        all_recommendations: List of recommendation lists (one per query)
        item_features: Dictionary mapping item indices to feature vectors (for ILD)
        item_categories: Dictionary mapping item indices to categories
        item_brands: Dictionary mapping item indices to brands
        item_popularity: Dictionary mapping item indices to popularity scores
        total_catalog_size: Total number of items in catalog (for coverage)
        
    Returns:
        Dictionary of diversity, novelty, and coverage metrics
    """
    results = {}
    
    # Calculate average diversity across all recommendation lists
    if item_features:
        ild_scores = []
        for recommendations in all_recommendations:
            if len(recommendations) >= 2:
                ild = intra_list_diversity(recommendations, item_features)
                ild_scores.append(ild)
        if ild_scores:
            results['Intra-List Diversity (ILD)'] = np.mean(ild_scores)
    
    if item_categories:
        cat_diversity_scores = []
        for recommendations in all_recommendations:
            if recommendations:
                cat_div = category_diversity(recommendations, item_categories)
                cat_diversity_scores.append(cat_div)
        if cat_diversity_scores:
            results['Category Diversity'] = np.mean(cat_diversity_scores)
    
    if item_brands:
        brand_div_scores = []
        for recommendations in all_recommendations:
            if recommendations:
                brand_div = brand_diversity(recommendations, item_brands)
                brand_div_scores.append(brand_div)
        if brand_div_scores:
            results['Brand Diversity'] = np.mean(brand_div_scores)
    
    if item_popularity:
        novelty_scores = []
        for recommendations in all_recommendations:
            if recommendations:
                nov = novelty(recommendations, item_popularity)
                novelty_scores.append(nov)
        if novelty_scores:
            results['Novelty'] = np.mean(novelty_scores)
    
    if total_catalog_size is not None:
        results['Catalog Coverage'] = catalog_coverage(all_recommendations, total_catalog_size)
    
    return results

