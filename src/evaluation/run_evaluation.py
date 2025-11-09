"""
Main Evaluation Script

Runs comprehensive evaluation of the recommendation system including:
- Performance metrics (Precision@K, Recall@K, NDCG, MAP, RMSE, MAE)
- Baseline comparisons
- Diversity, novelty, and coverage analysis
- Cold start problem handling
- Scalability and efficiency measurements
- Parameter tuning
- A/B testing

Usage:
    python run_evaluation.py [--csv-path PATH] [--output-dir DIR] [--quick]
"""

import argparse
import logging
import os
import json
import pandas as pd
from typing import Dict, List, Set, Tuple
from datetime import datetime

# Set environment variable to suppress tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from src.core.recommender_engine import ProductRecommender
from src.evaluation.evaluation_metrics import evaluate_recommendations, evaluate_rating_predictions
from src.evaluation.diversity_metrics import evaluate_diversity_novelty_coverage
from src.evaluation.baseline_comparison import evaluate_baselines_with_metrics, compare_baselines
from src.evaluation.cold_start import ColdStartHandler, document_cold_start_strategies
from src.evaluation.scalability_efficiency import (
    comprehensive_efficiency_analysis,
    generate_efficiency_report,
    measure_model_loading_time
)
from src.evaluation.parameter_tuning import (
    tune_bm25_parameters,
    tune_hybrid_weights,
    tune_feature_weights,
    find_best_parameters,
    generate_tuning_report
)
from src.evaluation.ab_testing import ABTestFramework, generate_ab_test_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_ground_truth(
    df: pd.DataFrame,
    test_queries: List[str],
    method: str = 'semantic'
) -> Dict[str, Set[int]]:
    """
    Create ground truth for evaluation.
    
    For demonstration, we'll use semantic similarity to create "relevant" items.
    In a real scenario, this would come from actual user interactions.
    
    Args:
        df: Product dataframe
        test_queries: List of test queries
        method: Method to create ground truth ('semantic' or 'keyword')
        
    Returns:
        Dictionary mapping queries to sets of relevant item indices
    """
    import tempfile
    import os
    
    ground_truth = {}
    
    # Create temporary CSV file for initialization
    temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_csv.name, index=False)
    temp_csv.close()
    
    try:
        # Initialize a simple recommender for ground truth creation
        recommender = ProductRecommender(
            csv_path=temp_csv.name,
            use_genai=(method == 'semantic'),
            load_models_immediately=True
        )
    finally:
        # Clean up temp file
        if os.path.exists(temp_csv.name):
            os.unlink(temp_csv.name)
    
    for query in test_queries:
        try:
            # Get top recommendations as "relevant" items
            results = recommender.recommend(query, top_k=20)
            
            if isinstance(results, list) and results and isinstance(results[0], dict):
                # Extract indices
                relevant_indices = set()
                for item in results[:15]:  # Top 15 as relevant
                    mask = (
                        (df['product_name'].astype(str) == item.get('name', '')) &
                        (df['brand'].astype(str) == item.get('brand', ''))
                    )
                    matches = df[mask].index.tolist()
                    if matches:
                        relevant_indices.add(matches[0])
                
                ground_truth[query] = relevant_indices
            else:
                ground_truth[query] = set()
        except Exception as e:
            logger.warning(f"Error creating ground truth for '{query}': {e}")
            ground_truth[query] = set()
    
    return ground_truth


def run_comprehensive_evaluation(
    csv_path: str,
    output_dir: str = "evaluation_results",
    quick_mode: bool = False
) -> Dict[str, any]:
    """
    Run comprehensive evaluation.
    
    Args:
        csv_path: Path to CSV dataset
        output_dir: Directory to save results
        quick_mode: If True, run faster evaluation with fewer queries
        
    Returns:
        Dictionary with all evaluation results
    """
    logger.info("=" * 60)
    logger.info("STARTING COMPREHENSIVE EVALUATION")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    logger.info(f"Loaded {len(df)} products")
    
    # Prepare test queries
    if quick_mode:
        test_queries = ["refrigerator", "washing machine", "microwave", "air conditioner"]
    else:
        test_queries = [
            "refrigerator", "washing machine", "microwave", "air conditioner",
            "dishwasher", "oven", "dryer", "freezer", "range", "cooktop"
        ]
    
    logger.info(f"Using {len(test_queries)} test queries")
    
    # Initialize recommender
    logger.info("Initializing recommendation system...")
    recommender = ProductRecommender(
        csv_path=csv_path,
        use_ngrams=True,
        use_advanced_preprocessing=True,
        use_genai=True,
        load_models_immediately=True
    )
    
    # Create ground truth
    logger.info("Creating ground truth for evaluation...")
    ground_truth = create_ground_truth(df, test_queries, method='semantic')
    
    results = {}
    
    # 1. Performance Metrics
    logger.info("\n" + "=" * 60)
    logger.info("1. PERFORMANCE METRICS")
    logger.info("=" * 60)
    performance_results = []
    for query in test_queries:
        try:
            rec_results = recommender.recommend(query, top_k=10)
            if isinstance(rec_results, list) and rec_results and isinstance(rec_results[0], dict):
                # Extract indices
                recommendations = []
                for item in rec_results[:10]:
                    mask = (
                        (df['product_name'].astype(str) == item.get('name', '')) &
                        (df['brand'].astype(str) == item.get('brand', ''))
                    )
                    matches = df[mask].index.tolist()
                    if matches:
                        recommendations.append(matches[0])
                
                relevant_items = ground_truth.get(query, set())
                if relevant_items:
                    metrics = evaluate_recommendations(recommendations, relevant_items, k_values=[5, 10])
                    metrics['query'] = query
                    performance_results.append(metrics)
        except Exception as e:
            logger.warning(f"Error evaluating query '{query}': {e}")
    
    # Average metrics
    if performance_results:
        avg_metrics = {}
        for k in [5, 10]:
            avg_metrics[f'Precision@{k}'] = sum(m[f'Precision@{k}'] for m in performance_results) / len(performance_results)
            avg_metrics[f'Recall@{k}'] = sum(m[f'Recall@{k}'] for m in performance_results) / len(performance_results)
            avg_metrics[f'NDCG@{k}'] = sum(m[f'NDCG@{k}'] for m in performance_results) / len(performance_results)
        avg_metrics['MAP'] = sum(m['MAP'] for m in performance_results) / len(performance_results)
        results['performance_metrics'] = {
            'average': avg_metrics,
            'per_query': performance_results
        }
        logger.info(f"Average Precision@10: {avg_metrics['Precision@10']:.4f}")
        logger.info(f"Average Recall@10: {avg_metrics['Recall@10']:.4f}")
        logger.info(f"Average NDCG@10: {avg_metrics['NDCG@10']:.4f}")
        logger.info(f"Average MAP: {avg_metrics['MAP']:.4f}")
    
    # 2. Baseline Comparison
    logger.info("\n" + "=" * 60)
    logger.info("2. BASELINE COMPARISON")
    logger.info("=" * 60)
    try:
        baseline_results = evaluate_baselines_with_metrics(
            test_queries[:5] if quick_mode else test_queries,
            df,
            ground_truth,
            top_k=10,
            k_values=[5, 10]
        )
        results['baseline_comparison'] = baseline_results
        logger.info("Baseline comparison completed")
    except Exception as e:
        logger.error(f"Error in baseline comparison: {e}")
        results['baseline_comparison'] = {'error': str(e)}
    
    # 3. Diversity, Novelty, Coverage
    logger.info("\n" + "=" * 60)
    logger.info("3. DIVERSITY, NOVELTY, COVERAGE")
    logger.info("=" * 60)
    try:
        all_recommendations = []
        for query in test_queries:
            rec_results = recommender.recommend(query, top_k=10)
            if isinstance(rec_results, list) and rec_results and isinstance(rec_results[0], dict):
                recommendations = []
                for item in rec_results[:10]:
                    mask = (
                        (df['product_name'].astype(str) == item.get('name', '')) &
                        (df['brand'].astype(str) == item.get('brand', ''))
                    )
                    matches = df[mask].index.tolist()
                    if matches:
                        recommendations.append(matches[0])
                all_recommendations.append(recommendations)
        
        # Prepare item features for diversity
        item_features = {}
        if hasattr(recommender, 'genai_embeddings') and recommender.genai_embeddings is not None:
            for idx in df.index:
                if idx < len(recommender.genai_embeddings):
                    item_features[idx] = recommender.genai_embeddings[idx]
        
        # Prepare item categories and brands
        item_categories = {}
        item_brands = {}
        for idx, row in df.iterrows():
            # Use product name as category proxy (first word)
            name = str(row.get('product_name', ''))
            category = name.split()[0] if name else 'unknown'
            item_categories[idx] = category
            item_brands[idx] = str(row.get('brand', 'unknown'))
        
        # Calculate popularity (inverse of bestseller rank)
        item_popularity = {}
        if 'bestseller_rank' in df.columns:
            max_rank = df['bestseller_rank'].max()
            for idx, row in df.iterrows():
                rank = row.get('bestseller_rank', max_rank)
                if pd.notna(rank):
                    popularity = 1.0 - (rank / max_rank) if max_rank > 0 else 0.5
                else:
                    popularity = 0.5
                item_popularity[idx] = max(popularity, 0.01)  # Minimum 0.01
        
        diversity_results = evaluate_diversity_novelty_coverage(
            all_recommendations,
            item_features=item_features if item_features else None,
            item_categories=item_categories,
            item_brands=item_brands,
            item_popularity=item_popularity if item_popularity else None,
            total_catalog_size=len(df)
        )
        results['diversity_novelty_coverage'] = diversity_results
        logger.info(f"Catalog Coverage: {diversity_results.get('Catalog Coverage', 0):.4f}")
        if 'Novelty' in diversity_results:
            logger.info(f"Novelty: {diversity_results['Novelty']:.4f}")
    except Exception as e:
        logger.error(f"Error in diversity analysis: {e}")
        results['diversity_novelty_coverage'] = {'error': str(e)}
    
    # 4. Cold Start Handling
    logger.info("\n" + "=" * 60)
    logger.info("4. COLD START HANDLING")
    logger.info("=" * 60)
    try:
        cold_start_handler = ColdStartHandler(recommender)
        new_user_queries = ["refrigerator", "washing machine"]
        new_item_names = [("Samsung Smart Refrigerator", "Samsung"), ("LG Washer", "LG")]
        cold_start_results = cold_start_handler.evaluate_cold_start_performance(
            new_user_queries,
            new_item_names
        )
        results['cold_start'] = cold_start_results
        logger.info("Cold start evaluation completed")
    except Exception as e:
        logger.error(f"Error in cold start evaluation: {e}")
        results['cold_start'] = {'error': str(e)}
    
    # 5. Scalability and Efficiency
    logger.info("\n" + "=" * 60)
    logger.info("5. SCALABILITY AND EFFICIENCY")
    logger.info("=" * 60)
    try:
        efficiency_results = comprehensive_efficiency_analysis(recommender, test_queries)
        results['scalability_efficiency'] = efficiency_results
        logger.info(f"Mean Query Time: {efficiency_results.get('query_performance', {}).get('mean_time_ms', 0):.2f} ms")
    except Exception as e:
        logger.error(f"Error in efficiency analysis: {e}")
        results['scalability_efficiency'] = {'error': str(e)}
    
    # 6. Parameter Tuning (simplified for quick mode)
    if not quick_mode:
        logger.info("\n" + "=" * 60)
        logger.info("6. PARAMETER TUNING")
        logger.info("=" * 60)
        try:
            # Tune hybrid weights (simplified)
            hybrid_results = tune_hybrid_weights(
                recommender,
                test_queries[:3],
                ground_truth,
                bm25_weight_range=[0.2, 0.4, 0.5, 0.6, 0.8],
                top_k=10
            )
            results['parameter_tuning'] = {
                'hybrid_weights': hybrid_results
            }
            logger.info("Parameter tuning completed")
        except Exception as e:
            logger.error(f"Error in parameter tuning: {e}")
            results['parameter_tuning'] = {'error': str(e)}
    
    # 7. A/B Testing (simplified)
    if not quick_mode:
        logger.info("\n" + "=" * 60)
        logger.info("7. A/B TESTING")
        logger.info("=" * 60)
        try:
            # Create control (TF-IDF only) and treatment (Hybrid)
            from src.evaluation.baseline_comparison import TFIDFBaseline, HybridRecommender
            
            control = TFIDFBaseline(df)
            treatment = HybridRecommender(df)
            
            ab_test = ABTestFramework(control, treatment, test_queries[:5])
            ab_results = ab_test.run_ab_test(
                num_users=100 if quick_mode else 500,
                split_ratio=0.5,
                ground_truth=ground_truth,
                top_k=10
            )
            results['ab_testing'] = ab_results['analysis']
            logger.info("A/B testing completed")
        except Exception as e:
            logger.error(f"Error in A/B testing: {e}")
            results['ab_testing'] = {'error': str(e)}
    
    # Save results
    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)
    
    # Save JSON (overwrite existing file)
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")
    
    # Generate text report (overwrite existing file)
    report_file = os.path.join(output_dir, "evaluation_report.txt")
    report = generate_evaluation_report(results)
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    
    return results


def generate_evaluation_report(results: Dict[str, any]) -> str:
    """Generate comprehensive evaluation report."""
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE EVALUATION REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Performance Metrics
    if 'performance_metrics' in results:
        pm = results['performance_metrics']
        if 'average' in pm:
            avg = pm['average']
            report.append("PERFORMANCE METRICS")
            report.append("-" * 80)
            report.append(f"Precision@5:  {avg.get('Precision@5', 0):.4f} (0.0-1.0, higher is better)")
            report.append(f"Precision@10: {avg.get('Precision@10', 0):.4f} (0.0-1.0, higher is better)")
            report.append(f"Recall@5:     {avg.get('Recall@5', 0):.4f} (0.0-1.0, higher is better)")
            report.append(f"Recall@10:    {avg.get('Recall@10', 0):.4f} (0.0-1.0, higher is better)")
            report.append(f"NDCG@5:       {avg.get('NDCG@5', 0):.4f} (0.0-1.0, higher is better)")
            report.append(f"NDCG@10:      {avg.get('NDCG@10', 0):.4f} (0.0-1.0, higher is better)")
            report.append(f"MAP:          {avg.get('MAP', 0):.4f} (0.0-1.0, higher is better)")
            report.append("")
    
    # Baseline Comparison
    if 'baseline_comparison' in results:
        bc = results['baseline_comparison']
        if 'error' not in bc:
            report.append("BASELINE COMPARISON")
            report.append("-" * 80)
            for method, metrics in bc.items():
                if isinstance(metrics, dict):
                    report.append(f"\n{method}:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            if 'Precision' in metric or 'Recall' in metric or 'NDCG' in metric or 'MAP' in metric:
                                report.append(f"  {metric}: {value:.4f} (0.0-1.0, higher is better)")
                            else:
                                report.append(f"  {metric}: {value:.4f}")
                        else:
                            report.append(f"  {metric}: {value}")
            report.append("")
    
    # Diversity, Novelty, Coverage
    if 'diversity_novelty_coverage' in results:
        dnc = results['diversity_novelty_coverage']
        if 'error' not in dnc:
            report.append("DIVERSITY, NOVELTY & COVERAGE")
            report.append("-" * 80)
            for metric, value in dnc.items():
                if isinstance(value, (int, float)):
                    if 'Coverage' in metric:
                        report.append(f"{metric}: {value:.4f} (0.0-1.0, percentage of catalog)")
                    elif 'Diversity' in metric or 'Novelty' in metric:
                        report.append(f"{metric}: {value:.4f} (0.0-1.0, higher = more diverse/novel)")
                    else:
                        report.append(f"{metric}: {value:.4f}")
                else:
                    report.append(f"{metric}: {value}")
            report.append("")
    
    # Cold Start
    if 'cold_start' in results:
        cs = results['cold_start']
        if 'error' not in cs:
            report.append("COLD START HANDLING")
            report.append("-" * 80)
            for key, value in cs.items():
                if isinstance(value, (int, float)):
                    if 'avg_recommendations' in key or 'avg_similar' in key:
                        report.append(f"{key}: {value:.2f} (average count)")
                    else:
                        report.append(f"{key}: {value}")
                else:
                    report.append(f"{key}: {value}")
            report.append("")
    
    # Scalability
    if 'scalability_efficiency' in results:
        se = results['scalability_efficiency']
        if 'error' not in se:
            report.append("SCALABILITY & EFFICIENCY")
            report.append("-" * 80)
            if 'query_performance' in se:
                qp = se['query_performance']
                mean_time = qp.get('mean_time_ms', 0)
                if isinstance(mean_time, (int, float)):
                    report.append(f"Mean Query Time: {mean_time:.2f} ms (milliseconds, lower is better)")
                if 'median_time_ms' in qp:
                    median_time = qp.get('median_time_ms', 0)
                    if isinstance(median_time, (int, float)):
                        report.append(f"Median Query Time: {median_time:.2f} ms")
                if 'std_time_ms' in qp:
                    std_time = qp.get('std_time_ms', 0)
                    if isinstance(std_time, (int, float)):
                        report.append(f"Std Dev Query Time: {std_time:.2f} ms")
            if 'memory_usage' in se:
                mu = se['memory_usage']
                rss_mb = mu.get('rss_mb', 0)
                if isinstance(rss_mb, (int, float)) and rss_mb > 0:
                    report.append(f"Memory Usage (RSS): {rss_mb:.2f} MB (megabytes)")
                if 'percent' in mu:
                    mem_percent = mu.get('percent', 0)
                    if isinstance(mem_percent, (int, float)) and mem_percent > 0:
                        report.append(f"Memory Usage: {mem_percent:.2f}% (percentage of system memory)")
            if 'dataset_stats' in se:
                ds = se['dataset_stats']
                if 'num_products' in ds:
                    report.append(f"Dataset Size: {ds.get('num_products', 0):,} products")
            report.append("")
    
    # Parameter Tuning
    if 'parameter_tuning' in results:
        pt = results['parameter_tuning']
        if 'error' not in pt:
            report.append("PARAMETER TUNING RESULTS")
            report.append("-" * 80)
            if 'hybrid_weights' in pt:
                hw = pt['hybrid_weights']
                if hw:
                    try:
                        best_weight = max(hw.items(), key=lambda x: x[1].get('NDCG@10', 0) if isinstance(x[1], dict) else 0)
                        bm25_w = best_weight[0]
                        semantic_w = 1.0 - bm25_w if isinstance(bm25_w, (int, float)) else 0.0
                        best_ndcg = best_weight[1].get('NDCG@10', 0) if isinstance(best_weight[1], dict) else 0
                        if isinstance(bm25_w, (int, float)):
                            report.append(f"Best Hybrid Weight - BM25: {bm25_w:.2f} (0.0-1.0)")
                            report.append(f"Best Hybrid Weight - Semantic: {semantic_w:.2f} (0.0-1.0)")
                        if isinstance(best_ndcg, (int, float)):
                            report.append(f"  Best NDCG@10: {best_ndcg:.4f} (0.0-1.0, higher is better)")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error formatting parameter tuning: {e}")
            report.append("")
    
    # A/B Testing
    if 'ab_testing' in results:
        ab = results['ab_testing']
        if 'error' not in ab:
            report.append("A/B TESTING RESULTS")
            report.append("-" * 80)
            if 'control' in ab and 'treatment' in ab:
                control = ab['control']
                treatment = ab['treatment']
                ctrl_ctr = control.get('mean_ctr', 0)
                treat_ctr = treatment.get('mean_ctr', 0)
                if isinstance(ctrl_ctr, (int, float)):
                    report.append(f"Control Group CTR: {ctrl_ctr:.4f} (0.0-1.0, click-through rate)")
                if isinstance(treat_ctr, (int, float)):
                    report.append(f"Treatment Group CTR: {treat_ctr:.4f} (0.0-1.0, click-through rate)")
                ctrl_rating = control.get('mean_rating', 0)
                treat_rating = treatment.get('mean_rating', 0)
                if isinstance(ctrl_rating, (int, float)) and ctrl_rating > 0:
                    report.append(f"Control Group Rating: {ctrl_rating:.2f} / 5.0 (stars)")
                if isinstance(treat_rating, (int, float)) and treat_rating > 0:
                    report.append(f"Treatment Group Rating: {treat_rating:.2f} / 5.0 (stars)")
            if 'ctr_test' in ab and 'error' not in ab['ctr_test']:
                ctr = ab['ctr_test']
                improvement = ctr.get('improvement', 0)
                if isinstance(improvement, (int, float)):
                    report.append(f"CTR Improvement: {improvement:+.2f}% (percentage change)")
                p_value = ctr.get('p_value', 0)
                if isinstance(p_value, (int, float)):
                    report.append(f"P-value: {p_value:.4f} (< 0.05 indicates significance)")
                report.append(f"Statistically Significant: {'Yes' if ctr.get('significant', False) else 'No'} (p < 0.05)")
            report.append("")
    
    # Note about RMSE/MAE and User-Item Matrix
    report.append("NOTES")
    report.append("-" * 80)
    report.append("RMSE and MAE: These metrics are for rating prediction tasks.")
    report.append("              Content-based filtering uses similarity scores, not ratings.")
    report.append("              These metrics are more relevant for collaborative filtering.")
    report.append("")
    report.append("User-Item Interaction Matrix: Not applicable for content-based filtering.")
    report.append("                                Content-based systems use item features, not user interactions.")
    report.append("                                This matrix is required for collaborative filtering approaches.")
    report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run comprehensive evaluation')
    parser.add_argument('--csv-path', type=str, default='data/home appliance skus lowes.csv',
                        help='Path to CSV dataset')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick evaluation (fewer queries, skip some tests)')
    
    args = parser.parse_args()
    
    # Get project root (parent of src directory)
    # __file__ is at src/evaluation/run_evaluation.py
    # Go up: src/evaluation -> src -> project_root
    current_file = os.path.abspath(__file__)
    eval_dir = os.path.dirname(current_file)  # src/evaluation
    src_dir = os.path.dirname(eval_dir)        # src
    project_root = os.path.dirname(src_dir)    # project_root
    
    # Convert to absolute path if relative
    if not os.path.isabs(args.csv_path):
        args.csv_path = os.path.join(project_root, args.csv_path)
    
    # Convert output directory to absolute path if relative
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    if not os.path.exists(args.csv_path):
        logger.error(f"CSV file not found: {args.csv_path}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Project root: {project_root}")
        return
    
    results = run_comprehensive_evaluation(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        quick_mode=args.quick
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    if 'performance_metrics' in results and 'average' in results['performance_metrics']:
        avg = results['performance_metrics']['average']
        print(f"Precision@10: {avg.get('Precision@10', 0):.4f}")
        print(f"NDCG@10:      {avg.get('NDCG@10', 0):.4f}")
        print(f"MAP:          {avg.get('MAP', 0):.4f}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

