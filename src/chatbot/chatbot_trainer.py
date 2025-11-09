"""
Chatbot Training Module

This module creates training data and knowledge base from the product dataset
to improve chatbot responses for product recommendations.
"""

import logging
import os
import json
import pandas as pd
from typing import List, Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# Training data configuration
TRAINING_DATA_DIR = 'training_data'
KNOWLEDGE_BASE_FILE = os.path.join(TRAINING_DATA_DIR, 'knowledge_base.json')
TRAINING_PROMPTS_FILE = os.path.join(TRAINING_DATA_DIR, 'training_prompts.json')


class ProductKnowledgeBase:
    """
    Knowledge base for product information extracted from the dataset.
    """
    
    def __init__(self, csv_path: str = "home appliance skus lowes.csv"):
        """
        Initialize the knowledge base from product dataset.
        
        Args:
            csv_path: Path to the product CSV file
        """
        self.csv_path = csv_path
        self.knowledge_base: Dict = {
            'products_by_category': defaultdict(list),
            'products_by_brand': defaultdict(list),
            'price_ranges': {},
            'common_products': [],
            'brands': set(),
            'categories': set(),
            'product_features': defaultdict(list)
        }
        self.load_knowledge_base()
    
    def load_knowledge_base(self) -> None:
        """Load or build knowledge base from dataset."""
        if os.path.exists(KNOWLEDGE_BASE_FILE):
            try:
                with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                # Convert sets back from lists
                self.knowledge_base['brands'] = set(self.knowledge_base.get('brands', []))
                self.knowledge_base['categories'] = set(self.knowledge_base.get('categories', []))
                logger.info(f"Loaded knowledge base from {KNOWLEDGE_BASE_FILE}")
                return
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}. Rebuilding...")
        
        # Build knowledge base from dataset
        self._build_knowledge_base()
    
    def _build_knowledge_base(self) -> None:
        """Build knowledge base from product dataset."""
        if not os.path.exists(self.csv_path):
            logger.error(f"Dataset file not found: {self.csv_path}")
            return
        
        try:
            logger.info(f"Building knowledge base from {self.csv_path}...")
            df = pd.read_csv(self.csv_path)
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Extract information
            for _, row in df.iterrows():
                product_name = str(row.get('product_name', '')).strip()
                brand = str(row.get('brand', '')).strip()
                category = str(row.get('category', '')).strip()
                subcategory = str(row.get('subcategory', '')).strip()
                price_current = row.get('price_current', 0)
                price_retail = row.get('price_retail', 0)
                
                if not product_name or product_name == 'nan':
                    continue
                
                # Store by category
                if category:
                    self.knowledge_base['products_by_category'][category].append({
                        'name': product_name,
                        'brand': brand,
                        'price': price_current,
                        'category': category,
                        'subcategory': subcategory
                    })
                    self.knowledge_base['categories'].add(category)
                
                # Store by brand
                if brand:
                    self.knowledge_base['products_by_brand'][brand].append({
                        'name': product_name,
                        'price': price_current,
                        'category': category
                    })
                    self.knowledge_base['brands'].add(brand)
                
                # Extract common products (appliances)
                product_lower = product_name.lower()
                if any(keyword in product_lower for keyword in ['refrigerator', 'fridge']):
                    self.knowledge_base['common_products'].append({
                        'type': 'refrigerator',
                        'name': product_name,
                        'brand': brand,
                        'price': price_current
                    })
                elif any(keyword in product_lower for keyword in ['washing machine', 'washer']):
                    self.knowledge_base['common_products'].append({
                        'type': 'washing_machine',
                        'name': product_name,
                        'brand': brand,
                        'price': price_current
                    })
                elif any(keyword in product_lower for keyword in ['air conditioner', 'ac']):
                    self.knowledge_base['common_products'].append({
                        'type': 'air_conditioner',
                        'name': product_name,
                        'brand': brand,
                        'price': price_current
                    })
                elif 'microwave' in product_lower:
                    self.knowledge_base['common_products'].append({
                        'type': 'microwave',
                        'name': product_name,
                        'brand': brand,
                        'price': price_current
                    })
            
            # Calculate price ranges
            prices = [p.get('price', 0) for p in self.knowledge_base['common_products'] if p.get('price', 0) > 0]
            if prices:
                self.knowledge_base['price_ranges'] = {
                    'min': min(prices),
                    'max': max(prices),
                    'avg': sum(prices) / len(prices)
                }
            
            # Convert sets to lists for JSON serialization
            kb_to_save = self.knowledge_base.copy()
            kb_to_save['brands'] = sorted(list(self.knowledge_base['brands']))
            kb_to_save['categories'] = sorted(list(self.knowledge_base['categories']))
            
            # Save knowledge base
            os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
            with open(KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
                json.dump(kb_to_save, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Knowledge base built successfully with {len(self.knowledge_base['common_products'])} products")
            logger.info(f"Found {len(self.knowledge_base['brands'])} brands and {len(self.knowledge_base['categories'])} categories")
        
        except Exception as e:
            logger.error(f"Error building knowledge base: {e}", exc_info=True)
    
    def get_product_context(self, product_type: Optional[str] = None, brand: Optional[str] = None) -> str:
        """
        Get context about products for chatbot prompts.
        
        Args:
            product_type: Type of product (e.g., 'refrigerator')
            brand: Brand name (optional)
            
        Returns:
            Context string with product information
        """
        context_parts = []
        
        # Add general product information
        if self.knowledge_base.get('common_products'):
            total_products = len(self.knowledge_base['common_products'])
            context_parts.append(f"We have {total_products} products available in our catalog.")
        
        # Add brand information
        brands = list(self.knowledge_base.get('brands', []))[:10]  # Top 10 brands
        if brands:
            context_parts.append(f"Popular brands include: {', '.join(brands)}.")
        
        # Add product type specific information
        if product_type:
            matching_products = [
                p for p in self.knowledge_base.get('common_products', [])
                if p.get('type') == product_type.lower().replace(' ', '_')
            ]
            if matching_products:
                context_parts.append(f"We have {len(matching_products)} {product_type} options available.")
                
                # Add price range for this product type
                prices = [p.get('price', 0) for p in matching_products if p.get('price', 0) > 0]
                if prices:
                    min_price = min(prices)
                    max_price = max(prices)
                    context_parts.append(f"Prices range from ${min_price:.2f} to ${max_price:.2f}.")
        
        # Add brand specific information
        if brand:
            brand_products = self.knowledge_base.get('products_by_brand', {}).get(brand, [])
            if brand_products:
                context_parts.append(f"{brand} has {len(brand_products)} products in our catalog.")
        
        return " ".join(context_parts) if context_parts else "We have a wide selection of home appliances available."
    
    def get_available_brands(self, product_type: Optional[str] = None) -> List[str]:
        """Get list of available brands."""
        brands = list(self.knowledge_base.get('brands', []))
        return sorted(brands)
    
    def get_product_types(self) -> List[str]:
        """Get list of available product types."""
        types = set()
        for product in self.knowledge_base.get('common_products', []):
            product_type = product.get('type', '').replace('_', ' ').title()
            if product_type:
                types.add(product_type)
        return sorted(list(types))


class ChatbotTrainer:
    """
    Creates training prompts and improves chatbot responses.
    """
    
    def __init__(self, knowledge_base: ProductKnowledgeBase):
        """
        Initialize the trainer.
        
        Args:
            knowledge_base: ProductKnowledgeBase instance
        """
        self.knowledge_base = knowledge_base
        self.training_prompts: List[Dict[str, str]] = []
    
    def generate_training_prompts(self) -> List[Dict[str, str]]:
        """
        Generate training prompts from knowledge base.
        
        Returns:
            List of training prompt dictionaries
        """
        prompts = []
        
        # Product recommendation prompts
        product_types = ['refrigerator', 'washing machine', 'air conditioner', 'microwave', 'oven', 'dishwasher']
        brands = self.knowledge_base.get_available_brands()[:20]  # Top 20 brands
        
        for product_type in product_types:
            # Basic product queries
            prompts.append({
                'user': f"find me a {product_type}",
                'context': self.knowledge_base.get_product_context(product_type),
                'expected_intent': 'product_search'
            })
            
            prompts.append({
                'user': f"show me {product_type}s",
                'context': self.knowledge_base.get_product_context(product_type),
                'expected_intent': 'product_search'
            })
            
            prompts.append({
                'user': f"I need a {product_type}",
                'context': self.knowledge_base.get_product_context(product_type),
                'expected_intent': 'product_search'
            })
            
            # Brand-specific queries
            for brand in brands[:5]:  # Top 5 brands per product type
                prompts.append({
                    'user': f"find me a {brand} {product_type}",
                    'context': self.knowledge_base.get_product_context(product_type, brand),
                    'expected_intent': 'product_search'
                })
            
            # Budget-specific queries
            for budget in [100, 300, 500, 1000, 2000]:
                prompts.append({
                    'user': f"show me {product_type}s under ${budget}",
                    'context': self.knowledge_base.get_product_context(product_type),
                    'expected_intent': 'product_search'
                })
        
        # General help prompts
        prompts.extend([
            {
                'user': 'what brands are available?',
                'context': self.knowledge_base.get_product_context(),
                'expected_intent': 'brand_inquiry'
            },
            {
                'user': 'what products do you have?',
                'context': self.knowledge_base.get_product_context(),
                'expected_intent': 'product_inquiry'
            },
            {
                'user': 'help me find products',
                'context': self.knowledge_base.get_product_context(),
                'expected_intent': 'help'
            },
            {
                'user': 'show me top rated products',
                'context': self.knowledge_base.get_product_context(),
                'expected_intent': 'product_search'
            },
            {
                'user': 'show me products under $500',
                'context': self.knowledge_base.get_product_context(),
                'expected_intent': 'product_search'
            }
        ])
        
        # Greetings and basic communication prompts
        greetings = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'greetings', 'howdy', 'what\'s up', 'sup'
        ]
        
        for greeting in greetings:
            prompts.append({
                'user': greeting,
                'context': self.knowledge_base.get_product_context(),
                'expected_intent': 'greeting'
            })
        
        # Basic communication patterns
        basic_communications = [
            ('thank you', 'thanks', 'thank you so much', 'thanks a lot'),
            ('goodbye', 'bye', 'see you', 'farewell', 'have a good day'),
            ('how are you', 'how are you doing', 'how\'s it going'),
            ('what can you do', 'what do you do', 'what are your capabilities'),
            ('who are you', 'what are you', 'introduce yourself'),
            ('yes', 'yeah', 'yep', 'sure', 'ok', 'okay'),
            ('no', 'nope', 'not really', 'not interested'),
            ('maybe', 'perhaps', 'i\'m not sure'),
            ('i need help', 'can you help', 'help me', 'i need assistance'),
            ('that\'s great', 'awesome', 'cool', 'nice', 'good'),
            ('i don\'t understand', 'what do you mean', 'can you explain'),
            ('tell me more', 'more information', 'more details'),
            ('that\'s all', 'nothing else', 'no more questions')
        ]
        
        for patterns in basic_communications:
            for pattern in patterns:
                prompts.append({
                    'user': pattern,
                    'context': self.knowledge_base.get_product_context(),
                    'expected_intent': 'basic_communication'
                })
        
        self.training_prompts = prompts
        return prompts
    
    def save_training_prompts(self) -> None:
        """Save training prompts to file."""
        os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
        with open(TRAINING_PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.training_prompts, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.training_prompts)} training prompts to {TRAINING_PROMPTS_FILE}")
    
    def get_enhanced_prompt_template(self) -> str:
        """
        Get enhanced prompt template for chatbot.
        
        Returns:
            Enhanced prompt template string
        """
        brands = ', '.join(self.knowledge_base.get_available_brands()[:15])
        product_types = ', '.join(self.knowledge_base.get_product_types()[:10])
        
        template = f"""You are a helpful AI assistant for a product recommendation system specializing in home appliances.

KNOWLEDGE BASE:
- Available brands: {brands}
- Product types: {product_types}
- We have thousands of products including refrigerators, washing machines, air conditioners, microwaves, and more.

YOUR CAPABILITIES:
1. Help users find products by type, brand, or budget
2. Answer questions about available products and brands
3. Provide product recommendations based on user needs
4. Guide users on how to use the search features

RESPONSE GUIDELINES:
- Be friendly, helpful, and concise
- When users ask for products, acknowledge their request and guide them to use the search feature
- If you can provide specific recommendations, do so
- Always be professional and product-focused

User: {{user_message}}
Assistant:"""
        
        return template


def ensure_training_data_ready(csv_path: str = "home appliance skus lowes.csv") -> bool:
    """
    Ensure training data is ready. Auto-generate if missing.
    
    This function checks if both knowledge base and training prompts exist.
    If either is missing, it will generate them automatically.
    
    Args:
        csv_path: Path to product CSV file
        
    Returns:
        True if training data is ready, False otherwise
    """
    knowledge_base_exists = os.path.exists(KNOWLEDGE_BASE_FILE)
    training_prompts_exists = os.path.exists(TRAINING_PROMPTS_FILE)
    
    # If both exist, we're good
    if knowledge_base_exists and training_prompts_exists:
        logger.info("Training data already exists. Skipping generation.")
        return True
    
    # If either is missing, generate both
    logger.info("Training data missing or incomplete. Auto-generating...")
    try:
        train_chatbot(csv_path)
        return True
    except Exception as e:
        logger.error(f"Failed to generate training data: {e}")
        return False


def train_chatbot(csv_path: str = "home appliance skus lowes.csv") -> None:
    """
    Main training function to build knowledge base and generate training data.
    
    This function:
    1. Builds/updates the knowledge base from the CSV dataset
    2. Generates training prompts for the chatbot
    3. Saves both to disk for future use
    
    Args:
        csv_path: Path to product CSV file
    """
    logger.info("Starting chatbot training process...")
    
    # Build knowledge base (will auto-generate if missing)
    knowledge_base = ProductKnowledgeBase(csv_path)
    
    # Generate training prompts
    trainer = ChatbotTrainer(knowledge_base)
    trainer.generate_training_prompts()
    trainer.save_training_prompts()
    
    logger.info("Chatbot training completed successfully!")
    logger.info(f"Knowledge base: {KNOWLEDGE_BASE_FILE}")
    logger.info(f"Training prompts: {TRAINING_PROMPTS_FILE}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_chatbot()

