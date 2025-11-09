"""
Scalability and Computational Efficiency Measurements

Measures and analyzes:
- Query response time
- Memory usage
- Scalability with dataset size
- Model loading time
- Embedding generation time
"""

import time
import os
import tracemalloc
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from contextlib import contextmanager

# Try to import psutil, make it optional
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. Memory usage measurements will be limited.")

from src.core.recommender_engine import ProductRecommender

logger = logging.getLogger(__name__)


@contextmanager
def measure_time():
    """Context manager to measure execution time."""
    start = time.time()
    yield
    end = time.time()
    elapsed = end - start
    return elapsed


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory usage in MB
    """
    if not PSUTIL_AVAILABLE:
        return {
            'rss_mb': 0.0,
            'vms_mb': 0.0,
            'percent': 0.0,
            'note': 'psutil not available'
        }
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
        'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
        'percent': process.memory_percent()
    }


def measure_query_time(
    recommender: ProductRecommender,
    queries: List[str],
    top_k: int = 10
) -> Dict[str, float]:
    """
    Measure query response time.
    
    Args:
        recommender: ProductRecommender instance
        queries: List of test queries
        top_k: Number of recommendations
        
    Returns:
        Dictionary with timing statistics
    """
    times = []
    
    for query in queries:
        start = time.time()
        try:
            recommender.recommend(query, top_k=top_k)
            elapsed = time.time() - start
            times.append(elapsed)
        except Exception as e:
            logger.warning(f"Error measuring query time for '{query}': {e}")
    
    if not times:
        return {}
    
    return {
        'mean_time_ms': np.mean(times) * 1000,
        'median_time_ms': np.median(times) * 1000,
        'min_time_ms': np.min(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'total_queries': len(times)
    }


def measure_model_loading_time(
    csv_path: str,
    use_genai: bool = True
) -> Dict[str, float]:
    """
    Measure model loading and initialization time.
    
    Args:
        csv_path: Path to CSV dataset
        use_genai: Whether to use GenAI features
        
    Returns:
        Dictionary with timing information
    """
    times = {}
    
    # Measure dataset loading
    start = time.time()
    recommender = ProductRecommender(
        csv_path=csv_path,
        use_genai=use_genai,
        load_models_immediately=False  # Don't load models yet
    )
    times['dataset_loading_ms'] = (time.time() - start) * 1000
    
    # Measure model loading
    start = time.time()
    recommender.load_models_and_embeddings()
    times['model_loading_ms'] = (time.time() - start) * 1000
    
    times['total_initialization_ms'] = times['dataset_loading_ms'] + times['model_loading_ms']
    
    return times


def measure_memory_usage(
    recommender: ProductRecommender
) -> Dict[str, float]:
    """
    Measure memory usage of recommender system.
    
    Args:
        recommender: ProductRecommender instance
        
    Returns:
        Dictionary with memory usage information
    """
    memory = get_memory_usage()
    
    # Estimate size of key data structures
    size_info = {}
    
    if hasattr(recommender, 'df') and not recommender.df.empty:
        size_info['dataset_size_mb'] = recommender.df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    if hasattr(recommender, 'tfidf_matrix') and recommender.tfidf_matrix is not None:
        # Estimate sparse matrix size
        size_info['tfidf_matrix_size_mb'] = recommender.tfidf_matrix.data.nbytes / (1024 * 1024)
    
    if hasattr(recommender, 'genai_embeddings') and recommender.genai_embeddings is not None:
        size_info['embeddings_size_mb'] = recommender.genai_embeddings.nbytes / (1024 * 1024)
    
    memory.update(size_info)
    
    return memory


def measure_scalability(
    csv_path: str,
    dataset_sizes: List[int],
    test_queries: List[str],
    use_genai: bool = True
) -> Dict[int, Dict[str, float]]:
    """
    Measure scalability with different dataset sizes.
    
    Args:
        csv_path: Path to CSV dataset
        dataset_sizes: List of dataset sizes to test (number of rows)
        test_queries: List of test queries
        use_genai: Whether to use GenAI features
        
    Returns:
        Dictionary mapping dataset size to performance metrics
    """
    # Load full dataset
    df_full = pd.read_csv(csv_path)
    
    results = {}
    
    for size in dataset_sizes:
        if size > len(df_full):
            logger.warning(f"Requested size {size} exceeds dataset size {len(df_full)}")
            size = len(df_full)
        
        # Sample dataset
        df_sample = df_full.head(size).copy()
        
        # Save temporary CSV
        temp_csv = f"temp_dataset_{size}.csv"
        df_sample.to_csv(temp_csv, index=False)
        
        try:
            # Measure initialization time
            init_start = time.time()
            recommender = ProductRecommender(
                csv_path=temp_csv,
                use_genai=use_genai,
                load_models_immediately=True
            )
            init_time = (time.time() - init_start) * 1000
            
            # Measure query time
            query_times = measure_query_time(recommender, test_queries)
            
            # Measure memory
            memory = measure_memory_usage(recommender)
            
            results[size] = {
                'initialization_time_ms': init_time,
                'mean_query_time_ms': query_times.get('mean_time_ms', 0),
                'memory_usage_mb': memory.get('rss_mb', 0),
                **query_times,
                **memory
            }
            
        except Exception as e:
            logger.error(f"Error measuring scalability for size {size}: {e}")
            results[size] = {'error': str(e)}
        finally:
            # Clean up temp file
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
    
    return results


def measure_embedding_generation_time(
    recommender: ProductRecommender,
    num_items: Optional[int] = None
) -> Dict[str, float]:
    """
    Measure time to generate embeddings.
    
    Args:
        recommender: ProductRecommender instance
        num_items: Number of items to test (None = all items)
        
    Returns:
        Dictionary with timing information
    """
    if not hasattr(recommender, 'genai_model') or recommender.genai_model is None:
        return {'error': 'GenAI model not available'}
    
    # Get text data
    df = recommender.df
    if num_items:
        df = df.head(num_items)
    
    text_data = []
    for _, row in df.iterrows():
        name = str(row.get("product_name", "")) if pd.notna(row.get("product_name")) else ""
        brand = str(row.get("brand", "")) if pd.notna(row.get("brand")) else ""
        combined = f"{name} {brand}".strip()
        text_data.append(combined)
    
    # Measure embedding generation
    start = time.time()
    embeddings = recommender.genai_model.encode(
        text_data,
        show_progress_bar=False,
        batch_size=64,
        convert_to_numpy=True
    )
    elapsed = time.time() - start
    
    return {
        'total_time_ms': elapsed * 1000,
        'items_processed': len(text_data),
        'time_per_item_ms': (elapsed * 1000) / len(text_data) if text_data else 0
    }


def comprehensive_efficiency_analysis(
    recommender: ProductRecommender,
    test_queries: List[str]
) -> Dict[str, any]:
    """
    Comprehensive efficiency analysis.
    
    Args:
        recommender: ProductRecommender instance
        test_queries: List of test queries
        
    Returns:
        Dictionary with comprehensive efficiency metrics
    """
    results = {}
    
    # Query performance
    logger.info("Measuring query performance...")
    results['query_performance'] = measure_query_time(recommender, test_queries)
    
    # Memory usage
    logger.info("Measuring memory usage...")
    results['memory_usage'] = measure_memory_usage(recommender)
    
    # Dataset statistics
    if hasattr(recommender, 'df') and not recommender.df.empty:
        results['dataset_stats'] = {
            'num_products': len(recommender.df),
            'num_brands': recommender.df['brand'].nunique() if 'brand' in recommender.df.columns else 0,
            'dataset_size_mb': recommender.df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
    
    # Embedding generation (if GenAI available)
    if hasattr(recommender, 'genai_model') and recommender.genai_model is not None:
        logger.info("Measuring embedding generation time...")
        results['embedding_generation'] = measure_embedding_generation_time(recommender)
    
    return results


def generate_efficiency_report(
    results: Dict[str, any]
) -> str:
    """
    Generate human-readable efficiency report.
    
    Args:
        results: Results from comprehensive_efficiency_analysis
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("COMPUTATIONAL EFFICIENCY REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Query Performance
    if 'query_performance' in results:
        qp = results['query_performance']
        report.append("QUERY PERFORMANCE:")
        report.append(f"  Mean Response Time: {qp.get('mean_time_ms', 0):.2f} ms")
        report.append(f"  Median Response Time: {qp.get('median_time_ms', 0):.2f} ms")
        report.append(f"  Min Response Time: {qp.get('min_time_ms', 0):.2f} ms")
        report.append(f"  Max Response Time: {qp.get('max_time_ms', 0):.2f} ms")
        report.append(f"  Std Deviation: {qp.get('std_time_ms', 0):.2f} ms")
        report.append(f"  Total Queries: {qp.get('total_queries', 0)}")
        report.append("")
    
    # Memory Usage
    if 'memory_usage' in results:
        mu = results['memory_usage']
        report.append("MEMORY USAGE:")
        report.append(f"  RSS (Resident Set Size): {mu.get('rss_mb', 0):.2f} MB")
        report.append(f"  VMS (Virtual Memory Size): {mu.get('vms_mb', 0):.2f} MB")
        report.append(f"  Memory Percent: {mu.get('percent', 0):.2f}%")
        if 'dataset_size_mb' in mu:
            report.append(f"  Dataset Size: {mu['dataset_size_mb']:.2f} MB")
        if 'embeddings_size_mb' in mu:
            report.append(f"  Embeddings Size: {mu['embeddings_size_mb']:.2f} MB")
        report.append("")
    
    # Dataset Stats
    if 'dataset_stats' in results:
        ds = results['dataset_stats']
        report.append("DATASET STATISTICS:")
        report.append(f"  Number of Products: {ds.get('num_products', 0):,}")
        report.append(f"  Number of Brands: {ds.get('num_brands', 0):,}")
        report.append(f"  Dataset Size: {ds.get('dataset_size_mb', 0):.2f} MB")
        report.append("")
    
    # Embedding Generation
    if 'embedding_generation' in results:
        eg = results['embedding_generation']
        if 'error' not in eg:
            report.append("EMBEDDING GENERATION:")
            report.append(f"  Total Time: {eg.get('total_time_ms', 0):.2f} ms")
            report.append(f"  Items Processed: {eg.get('items_processed', 0):,}")
            report.append(f"  Time per Item: {eg.get('time_per_item_ms', 0):.4f} ms")
            report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)

