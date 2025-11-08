"""
Chatbot Module for Product Recommendation System

This module provides a chatbot interface using LLM models for answering questions
about products and providing recommendations.
"""

import logging
import os
from typing import Optional, List, Dict
import shutil

# Try to import chatbot trainer for knowledge base
try:
    from chatbot_trainer import ProductKnowledgeBase, TRAINING_DATA_DIR, ensure_training_data_ready
    TRAINER_AVAILABLE = True
except ImportError:
    TRAINER_AVAILABLE = False
    ProductKnowledgeBase = None
    ensure_training_data_ready = None

# Try to import Hugging Face Transformers for LLM features
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logging.warning("Transformers library not available. Chatbot features disabled.")

logger = logging.getLogger(__name__)

# Chatbot model configuration
CHATBOT_MODEL_NAME = 'gpt2'  # Small, free model (~500MB)
CHATBOT_MODELS_DIR = 'models'


def get_chatbot_model_path(model_name: str = CHATBOT_MODEL_NAME) -> str:
    """
    Get the local path for the chatbot model cache directory.
    
    Args:
        model_name: Name of the model to cache
        
    Returns:
        Path to the model directory
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, CHATBOT_MODELS_DIR, model_name)
    return model_path


def ensure_chatbot_model_cached(model_name: str = CHATBOT_MODEL_NAME) -> str:
    """
    Ensure the chatbot model is cached locally. If not present, download it.
    
    Args:
        model_name: Name of the model to cache
        
    Returns:
        Path to the cached model directory or model name if download fails
    """
    model_path = get_chatbot_model_path(model_name)
    
    # Check if model directory exists and contains necessary files
    if os.path.exists(model_path):
        # Check for key files that indicate a complete model
        config_exists = os.path.exists(os.path.join(model_path, 'config.json'))
        model_file_exists = any(
            os.path.exists(os.path.join(model_path, f)) 
            for f in ['pytorch_model.bin', 'model.safetensors', 'tf_model.h5']
        )
        
        if config_exists and model_file_exists:
            logger.info(f"✓ Chatbot model found in cache: {model_path}")
            return model_path
        else:
            logger.warning(f"Chatbot model directory exists but appears incomplete. Re-downloading...")
            try:
                shutil.rmtree(model_path)
            except Exception as e:
                logger.warning(f"Could not remove incomplete model directory: {e}")
    
    # Model not found, download it
    logger.info(f"Chatbot model not found in cache. Downloading {model_name}...")
    logger.info(f"This may take a few minutes on first run. Model will be cached at: {model_path}")
    logger.info(f"Model size: ~500MB for GPT-2")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.dirname(model_path)
    os.makedirs(models_dir, exist_ok=True)
    
    if LLM_AVAILABLE:
        try:
            # Download model - it will be cached by transformers library
            # We'll use the default HuggingFace cache, then copy if needed
            logger.info(f"Downloading {model_name} from HuggingFace...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Save to our custom location
            logger.info(f"Saving model to cache: {model_path}")
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            
            logger.info(f"✓ Chatbot model downloaded and cached successfully at: {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Failed to download chatbot model: {e}")
            logger.info("Falling back to default model loading...")
            return model_name
    else:
        logger.warning("Transformers library not available. Cannot download chatbot model.")
        return model_name


class ProductChatbot:
    """
    Chatbot for answering questions about products and providing recommendations.
    
    Uses a lightweight LLM model to generate conversational responses.
    Can also recommend products from the dataset.
    """
    
    def __init__(self, use_llm: bool = True, recommender_engine=None):
        """
        Initialize the chatbot.
        
        Args:
            use_llm: Whether to use LLM for chatbot responses
            recommender_engine: Optional ProductRecommender instance for product recommendations
        """
        self.use_llm = use_llm and LLM_AVAILABLE
        self.pipeline = None
        self.conversation_history: List[Dict[str, str]] = []
        self.recommender_engine = recommender_engine
        
        # Initialize knowledge base if available
        self.knowledge_base = None
        if TRAINER_AVAILABLE:
            try:
                # Ensure training data is ready (auto-generate if missing)
                # This is especially important for new devices
                if ensure_training_data_ready:
                    ensure_training_data_ready()
                
                # Load knowledge base (will auto-generate if missing)
                self.knowledge_base = ProductKnowledgeBase()
                logger.info("Knowledge base loaded for enhanced chatbot responses")
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}")
                self.knowledge_base = None
        
        if self.use_llm:
            try:
                logger.info("Initializing chatbot with LLM...")
                # Use a lightweight model - GPT-2 is small and fast
                model_name = ensure_chatbot_model_cached(CHATBOT_MODEL_NAME)
                
                # Load model from cache or default location
                abs_model_path = os.path.abspath(model_name) if os.path.exists(model_name) else model_name
                
                if os.path.exists(abs_model_path) and os.path.exists(os.path.join(abs_model_path, 'config.json')):
                    logger.info(f"Loading chatbot model from cache: {abs_model_path}")
                    # Set offline mode
                    import os as os_module
                    original_hf_offline = os_module.environ.get('HF_HUB_OFFLINE', None)
                    os_module.environ['HF_HUB_OFFLINE'] = '1'
                    
                    try:
                        self.pipeline = pipeline(
                            "text-generation",
                            model=abs_model_path,
                            tokenizer=abs_model_path,
                            max_length=150,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=50256  # GPT-2 pad token
                        )
                        logger.info("✓ Chatbot LLM initialized successfully from cache")
                    finally:
                        if original_hf_offline is not None:
                            os_module.environ['HF_HUB_OFFLINE'] = original_hf_offline
                        elif 'HF_HUB_OFFLINE' in os_module.environ:
                            del os_module.environ['HF_HUB_OFFLINE']
                else:
                    logger.info(f"Loading chatbot model from default location (may download on first run)...")
                    self.pipeline = pipeline(
                        "text-generation",
                        model=model_name,
                        max_length=150,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=50256  # GPT-2 pad token
                    )
                    logger.info("✓ Chatbot LLM initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize chatbot LLM: {e}")
                logger.info("Chatbot will use rule-based responses (fallback mode)")
                self.use_llm = False
                self.pipeline = None
    
    def generate_response(self, user_message: str, context: Optional[str] = None) -> str:
        """
        Generate a chatbot response to user message.
        
        Args:
            user_message: User's message/question
            context: Optional context about products or recommendations
            
        Returns:
            Chatbot response string
        """
        if not user_message.strip():
            return "Please ask me a question about products!"
        
        # Check if user is asking for product recommendations
        message_lower = user_message.lower()
        recommendation_keywords = [
            'find', 'search', 'show', 'recommend', 'suggest', 'looking for',
            'need', 'want', 'buy', 'purchase', 'get', 'available'
        ]
        
        # Check if this is a product recommendation request
        is_product_query = any(keyword in message_lower for keyword in recommendation_keywords)
        
        # Try to extract product type, brand, and budget from the message
        if is_product_query and self.recommender_engine:
            try:
                products = self._extract_and_recommend_products(user_message)
                if products:
                    return self._format_product_recommendations(products, user_message)
            except Exception as e:
                logger.error(f"Error getting product recommendations: {e}")
                # Fall back to regular response
        
        # Build enhanced prompt with knowledge base context
        if self.knowledge_base:
            # Get product context from knowledge base
            product_type = self._extract_product_type(user_message)
            brand = self._extract_brand_from_message(user_message)
            kb_context = self.knowledge_base.get_product_context(product_type, brand)
            
            # Enhanced prompt template
            brands = ', '.join(self.knowledge_base.get_available_brands()[:10])
            prompt = f"""You are a helpful AI assistant for a product recommendation system specializing in home appliances.

KNOWLEDGE BASE:
- Available brands: {brands}
- Context: {kb_context}

YOUR CAPABILITIES:
1. Help users find products by type, brand, or budget
2. Answer questions about available products and brands
3. Provide product recommendations based on user needs
4. Guide users on how to use the search features

RESPONSE GUIDELINES:
- Be friendly, helpful, and concise
- When users ask for products, acknowledge their request
- If you can provide specific recommendations, do so
- Always be professional and product-focused

User: {user_message}
Assistant:"""
        elif context:
            prompt = f"Context: {context}\n\nUser: {user_message}\nAssistant:"
        else:
            # Create a product-focused prompt
            prompt = f"You are a helpful assistant for a product recommendation system. Answer questions about electrical appliances and products.\n\nUser: {user_message}\nAssistant:"
        
        if self.use_llm and self.pipeline:
            try:
                # Generate response using LLM
                response = self.pipeline(
                    prompt,
                    max_new_tokens=80,
                    num_return_sequences=1,
                    pad_token_id=50256,  # GPT-2 pad token
                    eos_token_id=50256
                )
                
                # Extract generated text
                generated_text = response[0]['generated_text']
                # Remove the prompt from the response
                if "Assistant:" in generated_text:
                    assistant_response = generated_text.split("Assistant:")[-1].strip()
                else:
                    # If no "Assistant:" marker, take text after the prompt
                    assistant_response = generated_text[len(prompt):].strip()
                
                # Clean up the response - take first sentence or first 200 chars
                assistant_response = assistant_response.split("\n")[0].strip()
                assistant_response = assistant_response.split(".")[0] + "." if "." in assistant_response else assistant_response
                assistant_response = assistant_response[:200].strip()
                
                # If response is too short or empty, provide a fallback
                if len(assistant_response) < 10 or not assistant_response:
                    return self._get_fallback_response(user_message)
                
                return assistant_response
            except Exception as e:
                logger.error(f"Error generating LLM response: {e}")
                return self._get_fallback_response(user_message)
        else:
            # Fallback to rule-based responses
            return self._get_fallback_response(user_message)
    
    def _get_fallback_response(self, user_message: str) -> str:
        """
        Get a fallback response when LLM is not available or fails.
        
        Args:
            user_message: User's message
            
        Returns:
            Fallback response string
        """
        message_lower = user_message.lower().strip()
        
        # Greetings
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings', 'howdy']):
            return "Hello! 👋 I'm your AI assistant for product recommendations. I can help you find the perfect appliances. What are you looking for today?"
        
        # Thank you responses
        elif any(word in message_lower for word in ['thank', 'thanks', 'appreciate']):
            return "You're very welcome! 😊 Happy shopping! If you need more help finding products, just ask!"
        
        # Goodbye responses
        elif any(word in message_lower for word in ['goodbye', 'bye', 'see you', 'farewell', 'have a good day']):
            return "Goodbye! 👋 Feel free to come back anytime if you need help finding products. Have a great day!"
        
        # How are you
        elif any(phrase in message_lower for phrase in ['how are you', 'how are you doing', 'how\'s it going', 'how do you do']):
            return "I'm doing great, thank you for asking! 😊 I'm here and ready to help you find the perfect products. What can I assist you with today?"
        
        # What can you do / capabilities
        elif any(phrase in message_lower for phrase in ['what can you do', 'what do you do', 'what are your capabilities', 'what can you help with']):
            return "I can help you:\n• Find products by type (refrigerators, washing machines, etc.)\n• Search by brand or budget\n• Answer questions about available products\n• Provide product recommendations\n\nWhat would you like to find?"
        
        # Who are you / introduction
        elif any(phrase in message_lower for phrase in ['who are you', 'what are you', 'introduce yourself', 'tell me about yourself']):
            return "I'm your AI assistant for the Product Recommendation System! 🤖 I specialize in helping you find the perfect home appliances. I can search through thousands of products and provide personalized recommendations. How can I help you today?"
        
        # Yes / confirmation
        elif any(word in message_lower for word in ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'correct', 'right']):
            return "Great! 😊 What would you like to do next? I can help you find products or answer any questions."
        
        # No / negative
        elif any(word in message_lower for word in ['no', 'nope', 'not really', 'not interested', 'don\'t want']):
            return "No problem! Is there something else I can help you with? I'm here to assist with product recommendations whenever you need."
        
        # Need help
        elif any(phrase in message_lower for phrase in ['i need help', 'can you help', 'help me', 'i need assistance', 'i\'m confused']):
            return "Of course! I'm here to help. 😊 I can assist you with:\n• Finding specific products\n• Searching by brand or budget\n• Answering questions about our catalog\n\nWhat would you like help with?"
        
        # Positive feedback
        elif any(word in message_lower for word in ['great', 'awesome', 'cool', 'nice', 'good', 'excellent', 'perfect', 'wonderful']):
            return "I'm glad I could help! 😊 Is there anything else you'd like to know or find?"
        
        # Don't understand
        elif any(phrase in message_lower for phrase in ['i don\'t understand', 'what do you mean', 'can you explain', 'i\'m confused', 'unclear']):
            return "I apologize for the confusion. Let me clarify - I'm here to help you find products. You can ask me to find specific appliances, search by brand, or filter by budget. What would you like to search for?"
        
        # More information
        elif any(phrase in message_lower for phrase in ['tell me more', 'more information', 'more details', 'elaborate']):
            return "I'd be happy to provide more details! What specific information would you like? I can tell you about products, brands, prices, or help you search for something specific."
        
        # Product-specific queries
        elif any(word in message_lower for word in ['refrigerator', 'fridge']):
            return "I can help you find refrigerators! 🧊 Try asking me to 'find me a refrigerator' or use the quick suggestions above. You can also filter by brand or budget!"
        elif any(word in message_lower for word in ['washing machine', 'washer']):
            return "Looking for a washing machine? 🌀 I can help you find the perfect one! Try asking 'find me a washing machine' or use the search feature with filters."
        elif any(word in message_lower for word in ['air conditioner', 'ac']):
            return "I can help you find air conditioners! ❄️ Search for 'air conditioner' and I'll show you options. You can filter by brand, budget, and more!"
        elif 'microwave' in message_lower:
            return "Microwaves are available! 🔥 I can help you find one. Try asking 'find me a microwave' or use the search feature."
        
        # Help / how to use
        elif any(word in message_lower for word in ['help', 'how', 'how to', 'how do i']):
            return "I can help you find products! Here's how:\n1. Ask me to find a product (e.g., 'find me a refrigerator')\n2. Use the quick suggestions above for common searches\n3. Specify brand or budget if needed\n4. I'll show you the best recommendations!\n\nWhat would you like to find?"
        
        # Top rated
        elif any(phrase in message_lower for phrase in ['top rated', 'best products', 'highest rated', 'most popular']):
            return "I can show you top-rated products! ⭐ Try asking me to find a specific product type, and I'll show you the best options. For example: 'find me a refrigerator' or 'show me top rated washing machines'."
        
        # Budget queries
        elif any(phrase in message_lower for phrase in ['under $', 'less than $', 'budget', 'cheap', 'affordable']):
            return "I can help you find products within your budget! 💰 Try asking like: 'find me a refrigerator under $500' or 'show me washing machines under $1000'. I'll filter the results for you!"
        
        # Default response
        else:
            return "I'm here to help you find products! 😊 Try asking me to find a specific appliance, or use the quick suggestions above. For example:\n• 'Find me a refrigerator'\n• 'Show me washing machines'\n• 'What brands are available?'\n\nWhat would you like to search for?"
    
    def _extract_product_type(self, user_message: str) -> Optional[str]:
        """
        Extract product type from user message.
        
        Args:
            user_message: User's message
            
        Returns:
            Product type string or None
        """
        message_lower = user_message.lower()
        product_types = {
            'refrigerator': ['refrigerator', 'fridge'],
            'washing machine': ['washing machine', 'washer'],
            'air conditioner': ['air conditioner', 'ac'],
            'microwave': ['microwave'],
            'oven': ['oven'],
            'dishwasher': ['dishwasher']
        }
        
        for product_type, keywords in product_types.items():
            if any(keyword in message_lower for keyword in keywords):
                return product_type
        return None
    
    def _extract_brand_from_message(self, user_message: str) -> Optional[str]:
        """
        Extract brand from user message.
        
        Args:
            user_message: User's message
            
        Returns:
            Brand name or None
        """
        if not self.knowledge_base:
            return None
        
        message_lower = user_message.lower()
        available_brands = self.knowledge_base.get_available_brands()
        
        for brand in available_brands:
            if brand.lower() in message_lower:
                return brand
        return None
    
    def _extract_and_recommend_products(self, user_message: str) -> List[Dict[str, str]]:
        """
        Extract product query from user message and get recommendations.
        
        Args:
            user_message: User's message
            
        Returns:
            List of product recommendations
        """
        if not self.recommender_engine:
            return []
        
        # Common product types
        product_types = [
            'refrigerator', 'fridge', 'washing machine', 'washer', 'dryer',
            'air conditioner', 'ac', 'microwave', 'oven', 'dishwasher',
            'vacuum', 'blender', 'mixer', 'toaster', 'coffee maker'
        ]
        
        # Extract product type
        product_query = ""
        for product_type in product_types:
            if product_type in user_message.lower():
                product_query = product_type
                break
        
        # If no specific product type found, try to use the whole message
        if not product_query:
            # Remove common words and use the message as query
            words = user_message.lower().split()
            filtered_words = [w for w in words if w not in ['find', 'show', 'me', 'a', 'an', 'the', 'for', 'i', 'want', 'need', 'looking']]
            if filtered_words:
                product_query = ' '.join(filtered_words[:3])  # Take first few words
        
        if not product_query:
            return []
        
        # Extract brand if mentioned
        brand_query = ""
        if self.recommender_engine:
            available_brands = self.recommender_engine.get_available_brands(product_query)
            for brand in available_brands:
                if brand.lower() in user_message.lower():
                    brand_query = brand
                    break
        
        # Extract budget if mentioned
        budget_max = None
        budget_keywords = {
            'under 100': 100, 'under $100': 100, 'less than 100': 100,
            'under 300': 300, 'under $300': 300, 'less than 300': 300,
            'under 500': 500, 'under $500': 500, 'less than 500': 500,
            'under 1000': 1000, 'under $1000': 1000, 'less than 1000': 1000,
            'under 2000': 2000, 'under $2000': 2000, 'less than 2000': 2000
        }
        for keyword, budget in budget_keywords.items():
            if keyword in user_message.lower():
                budget_max = budget
                break
        
        # Get recommendations
        try:
            recommendations = self.recommender_engine.recommend(
                product_query,
                brand_query=brand_query,
                budget_max=budget_max,
                top_k=5,  # Show top 5 in chat
                sort_by="Similarity (Default)"
            )
            
            # Filter out error messages
            if recommendations and isinstance(recommendations[0], dict):
                return recommendations[:5]  # Return top 5
            return []
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def _format_product_recommendations(self, products: List[Dict[str, str]], user_message: str) -> str:
        """
        Format product recommendations as a chat message.
        
        Args:
            products: List of product dictionaries
            user_message: Original user message
            
        Returns:
            Formatted string with product recommendations
        """
        if not products:
            return "I couldn't find any products matching your request. Try being more specific or check the Search tab!"
        
        response = f"Here are {len(products)} product recommendations for you:\n\n"
        response += "=" * 50 + "\n\n"
        
        for i, product in enumerate(products, 1):
            name = product.get('name', 'Unknown Product')
            brand = product.get('brand', 'Unknown Brand')
            price = product.get('price', 'N/A')
            rating = product.get('rating', 'N/A')
            url = product.get('url', '')
            
            response += f"📦 {i}. {name}\n"
            response += f"   Brand: {brand}\n"
            response += f"   Price: {price}\n"
            response += f"   Rating: {rating}\n"
            if url and url.startswith('http'):
                response += f"   🔗 {url}\n"
            response += "\n"
        
        response += "=" * 50 + "\n"
        response += "\n💡 Tip: Switch to the 'Search Products' tab to see more results and filter options!"
        
        return response
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Chatbot conversation history cleared")

