"""
Cold Start Problem Handling

Addresses the cold start problem in recommendation systems:
- New User Cold Start: Users with no interaction history
- New Item Cold Start: Products with no interaction history

Strategies implemented:
1. Content-based approach (works for both new users and new items)
2. Popular items fallback
3. Category-based recommendations
4. Hybrid strategies
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Set, Tuple
import logging

from src.core.recommender_engine import ProductRecommender

logger = logging.getLogger(__name__)


class ColdStartHandler:
    """Handles cold start problems for new users and new items."""
    
    def __init__(self, recommender: ProductRecommender):
        self.recommender = recommender
        self.df = recommender.df
    
    def handle_new_user(
        self,
        product_query: str = "",
        brand_preference: Optional[str] = None,
        budget_max: Optional[float] = None,
        top_k: int = 10
    ) -> List[Dict[str, str]]:
        """
        Handle new user cold start.
        
        Strategy: Use content-based filtering with query or popular items fallback.
        New users can provide:
        - Product query (what they're looking for)
        - Brand preference (if any)
        - Budget constraint
        
        Args:
            product_query: Product type or name user is looking for
            brand_preference: Preferred brand (optional)
            budget_max: Maximum budget (optional)
            top_k: Number of recommendations
            
        Returns:
            List of recommended products
        """
        # Strategy 1: If user provides query, use content-based filtering
        if product_query and product_query.strip():
            logger.info(f"New user with query: {product_query}")
            return self.recommender.recommend(
                product_query=product_query,
                brand_query=brand_preference or "",
                budget_max=budget_max,
                top_k=top_k
            )
        
        # Strategy 2: Fallback to popular items
        logger.info("New user without query - using popular items fallback")
        return self._get_popular_items(brand_preference, budget_max, top_k)
    
    def handle_new_item(
        self,
        new_item_name: str,
        new_item_brand: str,
        new_item_features: Optional[Dict[str, str]] = None,
        top_k: int = 10
    ) -> List[Dict[str, str]]:
        """
        Handle new item cold start.
        
        Strategy: Use content-based similarity to find similar existing items,
        then recommend those items to users who might like the new item.
        
        Args:
            new_item_name: Name of the new item
            new_item_brand: Brand of the new item
            new_item_features: Optional additional features
            top_k: Number of similar items to find
            
        Returns:
            List of similar existing items
        """
        logger.info(f"New item: {new_item_name} ({new_item_brand})")
        
        # Use content-based similarity to find similar items
        similar_items = self.recommender.get_similar_products(
            new_item_name,
            new_item_brand,
            top_k=top_k
        )
        
        return similar_items
    
    def _get_popular_items(
        self,
        brand_filter: Optional[str] = None,
        budget_max: Optional[float] = None,
        top_k: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get popular items as fallback for new users.
        
        Uses bestseller_rank if available, otherwise uses price as proxy.
        
        Args:
            brand_filter: Optional brand filter
            budget_max: Optional budget constraint
            top_k: Number of items to return
            
        Returns:
            List of popular products
        """
        df = self.df.copy()
        
        # Apply brand filter
        if brand_filter:
            df = df[df['brand'].astype(str).str.lower() == brand_filter.lower()]
        
        # Apply budget filter
        if budget_max:
            price_col = df.get('price_current', df.get('price_retail', pd.Series()))
            df = df[price_col.astype(float) <= budget_max]
        
        # Sort by bestseller_rank if available
        if 'bestseller_rank' in df.columns:
            df = df.sort_values('bestseller_rank', ascending=True, na_last=True)
        else:
            # Fallback: sort by price (lower is better for popularity proxy)
            price_col = df.get('price_current', df.get('price_retail', pd.Series()))
            df = df.sort_values(price_col, ascending=True, na_last=True)
        
        # Get top k
        top_items = df.head(top_k)
        
        # Format results
        results = []
        for _, row in top_items.iterrows():
            product_data = self.recommender._extract_product_data(row)
            product_data.pop("_price_num", None)
            product_data.pop("_rating_num", None)
            product_data.pop("_similarity_num", None)
            results.append(product_data)
        
        return results
    
    def get_category_based_recommendations(
        self,
        category: str,
        top_k: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get recommendations based on product category.
        
        Useful for new users who specify a category interest.
        
        Args:
            category: Product category (e.g., "refrigerator", "washing machine")
            top_k: Number of recommendations
            
        Returns:
            List of recommended products in the category
        """
        return self.recommender.recommend(category, top_k=top_k)
    
    def evaluate_cold_start_performance(
        self,
        new_user_queries: List[str],
        new_item_names: List[Tuple[str, str]],  # List of (name, brand) tuples
        ground_truth: Optional[Dict[str, Set[int]]] = None
    ) -> Dict[str, float]:
        """
        Evaluate cold start handling performance.
        
        Args:
            new_user_queries: List of queries from new users
            new_item_names: List of new items (name, brand tuples)
            ground_truth: Optional ground truth for evaluation
            
        Returns:
            Dictionary of performance metrics
        """
        from src.evaluation.evaluation_metrics import evaluate_recommendations
        
        results = {
            'new_user_queries_handled': len(new_user_queries),
            'new_items_handled': len(new_item_names),
            'avg_recommendations_per_query': 0.0
        }
        
        # Evaluate new user handling
        if new_user_queries:
            total_recs = 0
            for query in new_user_queries:
                recs = self.handle_new_user(product_query=query, top_k=10)
                if isinstance(recs, list):
                    total_recs += len(recs)
            results['avg_recommendations_per_query'] = total_recs / len(new_user_queries)
        
        # Evaluate new item handling
        if new_item_names:
            total_similar = 0
            for name, brand in new_item_names:
                similar = self.handle_new_item(name, brand, top_k=10)
                if isinstance(similar, list):
                    total_similar += len(similar)
            results['avg_similar_items_found'] = total_similar / len(new_item_names)
        
        return results


def document_cold_start_strategies() -> str:
    """
    Generate documentation for cold start handling strategies.
    
    Returns:
        Documentation string
    """
    doc = """
# Cold Start Problem Handling

## Overview
The cold start problem occurs when the system needs to make recommendations for:
1. **New Users**: Users with no interaction history
2. **New Items**: Products with no interaction history

## Strategies Implemented

### 1. New User Cold Start

**Problem**: New users have no purchase history or preferences.

**Solutions**:
- **Content-Based Filtering**: Users provide a query (e.g., "refrigerator"), and we use content-based similarity
- **Popular Items Fallback**: If no query, recommend popular/bestselling items
- **Category-Based**: Users can specify category interest
- **Hybrid Approach**: Combine query + brand preference + budget constraints

**Advantages**:
- Works immediately without user history
- Leverages product features (name, brand, category)
- Can incorporate explicit preferences (budget, brand)

### 2. New Item Cold Start

**Problem**: New products have no interaction history (no purchases, no ratings).

**Solutions**:
- **Content-Based Similarity**: Find similar existing items based on:
  - Product name similarity
  - Brand similarity
  - Feature similarity (if available)
- **Recommend Similar Items**: Users who liked similar items might like the new item

**Advantages**:
- Works immediately for new products
- Leverages product content (name, brand, features)
- No need to wait for user interactions

## Implementation Details

### Content-Based Approach
- Uses TF-IDF, BM25, and Semantic Embeddings
- Calculates similarity between query/item and existing products
- No dependency on user interaction history

### Popular Items Fallback
- Uses bestseller_rank if available
- Falls back to price-based sorting
- Ensures recommendations even without query

### Category-Based Recommendations
- Users can specify product category
- System recommends items within that category
- Useful for exploratory browsing

## Performance Considerations

- **Latency**: Content-based filtering is fast (no need to compute user-item matrices)
- **Scalability**: Works well with large catalogs
- **Accuracy**: May be lower than collaborative filtering for users with rich history, but works for cold start

## Future Enhancements

1. **Hybrid with Collaborative**: Once user has some history, switch to collaborative filtering
2. **Demographic Features**: Use user demographics (age, location) for better cold start
3. **Social Signals**: Use social connections for new user recommendations
4. **Active Learning**: Ask new users explicit questions to build initial profile
"""
    return doc

