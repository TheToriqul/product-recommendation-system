"""
Unit tests for the Product Recommendation Engine.

Run with: python -m pytest test_recommender.py
Or: python test_recommender.py
"""

import unittest
import os
import pandas as pd
from recommender_engine import ProductRecommender


class TestProductRecommender(unittest.TestCase):
    """Test cases for ProductRecommender class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        csv_path = "home appliance skus lowes.csv"
        if os.path.exists(csv_path):
            cls.recommender = ProductRecommender(csv_path)
        else:
            # Create a minimal test dataset
            cls._create_test_dataset(csv_path)
            cls.recommender = ProductRecommender(csv_path)
    
    @classmethod
    def _create_test_dataset(cls, csv_path: str):
        """Create a minimal test dataset."""
        test_data = {
            'product_name': [
                'Samsung Refrigerator 25 cu ft',
                'LG Refrigerator 20 cu ft',
                'Whirlpool Washing Machine',
                'Samsung Washing Machine',
                'GE Microwave Oven',
                'Panasonic Microwave Oven'
            ],
            'brand': ['Samsung', 'LG', 'Whirlpool', 'Samsung', 'GE', 'Panasonic'],
            'price_current': [899.99, 799.99, 599.99, 649.99, 199.99, 179.99],
            'price_retail': [999.99, 899.99, 699.99, 749.99, 249.99, 219.99],
            'bestseller_rank': [1, 2, 1, 3, 1, 2],
            'product_url': [
                'https://example.com/samsung-fridge',
                'https://example.com/lg-fridge',
                'https://example.com/whirlpool-washer',
                'https://example.com/samsung-washer',
                'https://example.com/ge-microwave',
                'https://example.com/panasonic-microwave'
            ]
        }
        df = pd.DataFrame(test_data)
        df.to_csv(csv_path, index=False)
    
    def test_initialization(self):
        """Test recommender initialization."""
        self.assertIsNotNone(self.recommender)
        self.assertFalse(self.recommender.df.empty)
        self.assertIsNotNone(self.recommender.vectorizer)
    
    def test_get_available_brands(self):
        """Test getting available brands."""
        brands = self.recommender.get_available_brands()
        self.assertIsInstance(brands, list)
        self.assertGreater(len(brands), 0)
    
    def test_get_available_brands_with_query(self):
        """Test getting brands filtered by product query."""
        brands = self.recommender.get_available_brands("refrigerator")
        self.assertIsInstance(brands, list)
    
    def test_recommend_basic(self):
        """Test basic recommendation functionality."""
        results = self.recommender.recommend("refrigerator", top_k=5)
        self.assertIsInstance(results, list)
        if results and isinstance(results[0], dict):
            self.assertIn('name', results[0])
            self.assertIn('brand', results[0])
            self.assertIn('price', results[0])
    
    def test_recommend_with_brand(self):
        """Test recommendations with brand filter."""
        results = self.recommender.recommend("refrigerator", brand_query="Samsung", top_k=5)
        self.assertIsInstance(results, list)
    
    def test_recommend_with_budget(self):
        """Test recommendations with budget constraint."""
        results = self.recommender.recommend("refrigerator", budget_max=900, top_k=5)
        self.assertIsInstance(results, list)
    
    def test_recommend_sorting(self):
        """Test recommendation sorting options."""
        for sort_option in ["Price: Low to High", "Price: High to Low", "Rating: Best First"]:
            results = self.recommender.recommend("refrigerator", sort_by=sort_option, top_k=5)
            self.assertIsInstance(results, list)
    
    def test_get_similar_products(self):
        """Test getting similar products."""
        similar = self.recommender.get_similar_products("Samsung Refrigerator", "Samsung", top_k=3)
        self.assertIsInstance(similar, list)
        if similar:
            self.assertIn('name', similar[0])
            self.assertIn('brand', similar[0])
    
    def test_format_price(self):
        """Test price formatting."""
        price = self.recommender._format_price(99.99)
        self.assertEqual(price, "$99.99")
        
        price = self.recommender._format_price(None)
        self.assertEqual(price, "N/A")
    
    def test_format_rating(self):
        """Test rating formatting."""
        rating = self.recommender._format_rating(1)
        self.assertEqual(rating, "Rank #1")
        
        rating = self.recommender._format_rating(None)
        self.assertEqual(rating, "N/A")


if __name__ == '__main__':
    unittest.main()

