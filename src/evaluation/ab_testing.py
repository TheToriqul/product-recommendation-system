"""
A/B Testing Framework for Recommendation System

Implements A/B testing to compare different recommendation strategies:
- Control group (baseline method)
- Treatment group (new method)
- Statistical significance testing
- User satisfaction metrics simulation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
import logging
from scipy import stats
import random

from src.core.recommender_engine import ProductRecommender
from src.evaluation.evaluation_metrics import evaluate_recommendations

logger = logging.getLogger(__name__)


class ABTestFramework:
    """Framework for A/B testing recommendation systems."""
    
    def __init__(
        self,
        control_recommender: ProductRecommender,
        treatment_recommender: ProductRecommender,
        test_queries: List[str]
    ):
        """
        Initialize A/B test framework.
        
        Args:
            control_recommender: Control group recommender (baseline)
            treatment_recommender: Treatment group recommender (new method)
            test_queries: List of test queries
        """
        self.control_recommender = control_recommender
        self.treatment_recommender = treatment_recommender
        self.test_queries = test_queries
        
        self.control_results = []
        self.treatment_results = []
    
    def assign_users(self, num_users: int, split_ratio: float = 0.5) -> Dict[str, List[int]]:
        """
        Assign users to control and treatment groups.
        
        Args:
            num_users: Total number of users
            split_ratio: Ratio for control group (default 0.5 = 50/50 split)
            
        Returns:
            Dictionary with 'control' and 'treatment' user lists
        """
        all_users = list(range(num_users))
        random.shuffle(all_users)
        
        split_point = int(num_users * split_ratio)
        control_users = all_users[:split_point]
        treatment_users = all_users[split_point:]
        
        return {
            'control': control_users,
            'treatment': treatment_users
        }
    
    def simulate_user_interactions(
        self,
        user_assignments: Dict[str, List[int]],
        ground_truth: Optional[Dict[str, Set[int]]] = None,
        top_k: int = 10
    ) -> Dict[str, List[Dict]]:
        """
        Simulate user interactions for A/B test.
        
        Args:
            user_assignments: User assignments from assign_users
            ground_truth: Optional ground truth for evaluation
            top_k: Number of recommendations
            
        Returns:
            Dictionary with interaction data for control and treatment groups
        """
        control_interactions = []
        treatment_interactions = []
        
        # Simulate control group interactions
        for user_id in user_assignments['control']:
            query = random.choice(self.test_queries)
            try:
                recommendations = self.control_recommender.recommend(query, top_k=top_k)
                
                # Simulate user satisfaction (click-through, rating, etc.)
                satisfaction = self._simulate_satisfaction(recommendations, query, ground_truth)
                
                control_interactions.append({
                    'user_id': user_id,
                    'query': query,
                    'recommendations': recommendations,
                    'satisfaction': satisfaction,
                    'group': 'control'
                })
            except Exception as e:
                logger.warning(f"Error simulating control interaction: {e}")
        
        # Simulate treatment group interactions
        for user_id in user_assignments['treatment']:
            query = random.choice(self.test_queries)
            try:
                recommendations = self.treatment_recommender.recommend(query, top_k=top_k)
                
                # Simulate user satisfaction
                satisfaction = self._simulate_satisfaction(recommendations, query, ground_truth)
                
                treatment_interactions.append({
                    'user_id': user_id,
                    'query': query,
                    'recommendations': recommendations,
                    'satisfaction': satisfaction,
                    'group': 'treatment'
                })
            except Exception as e:
                logger.warning(f"Error simulating treatment interaction: {e}")
        
        return {
            'control': control_interactions,
            'treatment': treatment_interactions
        }
    
    def _simulate_satisfaction(
        self,
        recommendations: List,
        query: str,
        ground_truth: Optional[Dict[str, Set[int]]] = None
    ) -> Dict[str, float]:
        """
        Simulate user satisfaction metrics.
        
        Args:
            recommendations: List of recommendations
            query: User query
            ground_truth: Optional ground truth
            
        Returns:
            Dictionary of satisfaction metrics
        """
        satisfaction = {}
        
        if isinstance(recommendations, list) and recommendations:
            # Click-through rate (CTR) - probability of clicking on recommendations
            if ground_truth and query in ground_truth:
                relevant_items = ground_truth[query]
                # Count relevant items in recommendations
                if isinstance(recommendations[0], dict):
                    rec_indices = []
                    for item in recommendations:
                        mask = (
                            (self.control_recommender.df['product_name'].astype(str) == item.get('name', '')) &
                            (self.control_recommender.df['brand'].astype(str) == item.get('brand', ''))
                        )
                        matches = self.control_recommender.df[mask].index.tolist()
                        if matches:
                            rec_indices.append(matches[0])
                    relevant_count = sum(1 for idx in rec_indices if idx in relevant_items)
                    satisfaction['ctr'] = relevant_count / len(recommendations) if recommendations else 0
                else:
                    relevant_count = sum(1 for idx in recommendations if idx in relevant_items)
                    satisfaction['ctr'] = relevant_count / len(recommendations) if recommendations else 0
            else:
                # Simulate CTR based on recommendation quality (random but realistic)
                satisfaction['ctr'] = np.random.uniform(0.1, 0.4)  # 10-40% CTR
            
            # Average rating (simulated)
            satisfaction['avg_rating'] = np.random.uniform(3.5, 5.0)  # 3.5-5.0 stars
            
            # Engagement score (combination of metrics)
            satisfaction['engagement'] = satisfaction['ctr'] * 0.6 + (satisfaction['avg_rating'] / 5.0) * 0.4
        else:
            satisfaction = {
                'ctr': 0.0,
                'avg_rating': 0.0,
                'engagement': 0.0
            }
        
        return satisfaction
    
    def analyze_results(
        self,
        interactions: Dict[str, List[Dict]]
    ) -> Dict[str, any]:
        """
        Analyze A/B test results with statistical significance.
        
        Args:
            interactions: Interaction data from simulate_user_interactions
            
        Returns:
            Dictionary with analysis results
        """
        control_satisfactions = [i['satisfaction'] for i in interactions['control']]
        treatment_satisfactions = [i['satisfaction'] for i in interactions['treatment']]
        
        # Extract metrics
        control_ctr = [s['ctr'] for s in control_satisfactions]
        treatment_ctr = [s['ctr'] for s in treatment_satisfactions]
        
        control_ratings = [s['avg_rating'] for s in control_satisfactions]
        treatment_ratings = [s['avg_rating'] for s in treatment_satisfactions]
        
        control_engagement = [s['engagement'] for s in control_satisfactions]
        treatment_engagement = [s['engagement'] for s in treatment_satisfactions]
        
        # Calculate statistics
        results = {
            'control': {
                'sample_size': len(control_satisfactions),
                'mean_ctr': np.mean(control_ctr),
                'mean_rating': np.mean(control_ratings),
                'mean_engagement': np.mean(control_engagement)
            },
            'treatment': {
                'sample_size': len(treatment_satisfactions),
                'mean_ctr': np.mean(treatment_ctr),
                'mean_rating': np.mean(treatment_ratings),
                'mean_engagement': np.mean(treatment_engagement)
            }
        }
        
        # Statistical significance tests
        # T-test for CTR
        try:
            t_stat_ctr, p_value_ctr = stats.ttest_ind(control_ctr, treatment_ctr)
            results['ctr_test'] = {
                't_statistic': t_stat_ctr,
                'p_value': p_value_ctr,
                'significant': p_value_ctr < 0.05,
                'improvement': (np.mean(treatment_ctr) - np.mean(control_ctr)) / np.mean(control_ctr) * 100
            }
        except Exception as e:
            logger.warning(f"Error in CTR t-test: {e}")
            results['ctr_test'] = {'error': str(e)}
        
        # T-test for ratings
        try:
            t_stat_rating, p_value_rating = stats.ttest_ind(control_ratings, treatment_ratings)
            results['rating_test'] = {
                't_statistic': t_stat_rating,
                'p_value': p_value_rating,
                'significant': p_value_rating < 0.05,
                'improvement': (np.mean(treatment_ratings) - np.mean(control_ratings)) / np.mean(control_ratings) * 100
            }
        except Exception as e:
            logger.warning(f"Error in rating t-test: {e}")
            results['rating_test'] = {'error': str(e)}
        
        # T-test for engagement
        try:
            t_stat_eng, p_value_eng = stats.ttest_ind(control_engagement, treatment_engagement)
            results['engagement_test'] = {
                't_statistic': t_stat_eng,
                'p_value': p_value_eng,
                'significant': p_value_eng < 0.05,
                'improvement': (np.mean(treatment_engagement) - np.mean(control_engagement)) / np.mean(control_engagement) * 100
            }
        except Exception as e:
            logger.warning(f"Error in engagement t-test: {e}")
            results['engagement_test'] = {'error': str(e)}
        
        return results
    
    def run_ab_test(
        self,
        num_users: int = 1000,
        split_ratio: float = 0.5,
        ground_truth: Optional[Dict[str, Set[int]]] = None,
        top_k: int = 10
    ) -> Dict[str, any]:
        """
        Run complete A/B test.
        
        Args:
            num_users: Number of users in test
            split_ratio: Ratio for control group
            ground_truth: Optional ground truth
            top_k: Number of recommendations
            
        Returns:
            Complete A/B test results
        """
        logger.info(f"Running A/B test with {num_users} users...")
        
        # Assign users
        user_assignments = self.assign_users(num_users, split_ratio)
        
        # Simulate interactions
        interactions = self.simulate_user_interactions(user_assignments, ground_truth, top_k)
        
        # Analyze results
        analysis = self.analyze_results(interactions)
        
        return {
            'user_assignments': user_assignments,
            'interactions': interactions,
            'analysis': analysis
        }


def generate_ab_test_report(results: Dict[str, any]) -> str:
    """
    Generate human-readable A/B test report.
    
    Args:
        results: Results from run_ab_test
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("A/B TEST RESULTS")
    report.append("=" * 60)
    report.append("")
    
    analysis = results.get('analysis', {})
    
    # Control group stats
    if 'control' in analysis:
        control = analysis['control']
        report.append("CONTROL GROUP (Baseline):")
        report.append(f"  Sample Size: {control.get('sample_size', 0):,}")
        report.append(f"  Mean CTR: {control.get('mean_ctr', 0):.4f} ({control.get('mean_ctr', 0)*100:.2f}%)")
        report.append(f"  Mean Rating: {control.get('mean_rating', 0):.2f}/5.0")
        report.append(f"  Mean Engagement: {control.get('mean_engagement', 0):.4f}")
        report.append("")
    
    # Treatment group stats
    if 'treatment' in analysis:
        treatment = analysis['treatment']
        report.append("TREATMENT GROUP (New Method):")
        report.append(f"  Sample Size: {treatment.get('sample_size', 0):,}")
        report.append(f"  Mean CTR: {treatment.get('mean_ctr', 0):.4f} ({treatment.get('mean_ctr', 0)*100:.2f}%)")
        report.append(f"  Mean Rating: {treatment.get('mean_rating', 0):.2f}/5.0")
        report.append(f"  Mean Engagement: {treatment.get('mean_engagement', 0):.4f}")
        report.append("")
    
    # Statistical tests
    if 'ctr_test' in analysis:
        ctr_test = analysis['ctr_test']
        if 'error' not in ctr_test:
            report.append("CLICK-THROUGH RATE (CTR) TEST:")
            report.append(f"  P-value: {ctr_test.get('p_value', 0):.4f}")
            report.append(f"  Significant: {'Yes' if ctr_test.get('significant', False) else 'No'}")
            improvement = ctr_test.get('improvement', 0)
            report.append(f"  Improvement: {improvement:+.2f}%")
            report.append("")
    
    if 'rating_test' in analysis:
        rating_test = analysis['rating_test']
        if 'error' not in rating_test:
            report.append("RATING TEST:")
            report.append(f"  P-value: {rating_test.get('p_value', 0):.4f}")
            report.append(f"  Significant: {'Yes' if rating_test.get('significant', False) else 'No'}")
            improvement = rating_test.get('improvement', 0)
            report.append(f"  Improvement: {improvement:+.2f}%")
            report.append("")
    
    if 'engagement_test' in analysis:
        eng_test = analysis['engagement_test']
        if 'error' not in eng_test:
            report.append("ENGAGEMENT TEST:")
            report.append(f"  P-value: {eng_test.get('p_value', 0):.4f}")
            report.append(f"  Significant: {'Yes' if eng_test.get('significant', False) else 'No'}")
            improvement = eng_test.get('improvement', 0)
            report.append(f"  Improvement: {improvement:+.2f}%")
            report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)

