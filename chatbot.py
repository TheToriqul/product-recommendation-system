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
    
    def _is_out_of_scope(self, user_message: str) -> bool:
        """
        Check if user message is about products outside our scope (home appliances).
        
        Args:
            user_message: User's message
            
        Returns:
            True if message is about non-appliance products, False otherwise
        """
        message_lower = user_message.lower()
        
        # Keywords for products outside our scope
        out_of_scope_keywords = [
            # Mobile devices
            'mobile phone', 'smartphone', 'cell phone', 'iphone', 'android phone',
            'tablet', 'ipad', 'phone',
            # Computers
            'laptop', 'computer', 'pc', 'desktop', 'macbook', 'notebook',
            # TVs and displays
            'tv', 'television', 'smart tv', 'monitor', 'display', 'screen',
            # Gaming
            'gaming console', 'playstation', 'xbox', 'nintendo', 'playstation',
            'video game', 'game console',
            # Cameras
            'camera', 'dslr', 'mirrorless', 'photography', 'lens',
            # Audio (non-appliance)
            'headphone', 'earphone', 'earbud', 'speaker', 'bluetooth speaker',
            # Other electronics
            'smartwatch', 'watch', 'fitness tracker', 'drone',
            # Clothing and fashion
            'clothes', 'clothing', 'shirt', 'pants', 'shoes', 'fashion',
            # Food and beverages
            'food', 'restaurant', 'recipe', 'cooking recipe',
            # Vehicles
            'car', 'vehicle', 'automobile', 'motorcycle', 'bike',
            # Books and media
            'book', 'movie', 'music', 'song', 'album'
        ]
        
        # Check if message contains out-of-scope keywords
        for keyword in out_of_scope_keywords:
            if keyword in message_lower:
                # Double-check: make sure it's not about home appliances
                # Some words might overlap (e.g., "oven" in "microwave oven")
                appliance_keywords = [
                    'refrigerator', 'fridge', 'washing machine', 'washer', 'dryer',
                    'air conditioner', 'ac', 'microwave', 'oven', 'dishwasher',
                    'vacuum', 'blender', 'mixer', 'toaster', 'coffee maker',
                    'appliance', 'home appliance'
                ]
                
                # If it's clearly an appliance query, it's in scope
                if any(appliance in message_lower for appliance in appliance_keywords):
                    return False
                
                # Otherwise, it's out of scope
                return True
        
        return False
    
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
            return "Hey! What can I help you find today?"
        
        # Check if query is out of scope (not about home appliances)
        if self._is_out_of_scope(user_message):
            return ("I'm sorry, but I specialize in home appliances only (refrigerators, washing machines, "
                   "air conditioners, microwaves, etc.). I don't have information about mobile phones, "
                   "electronics, or other products outside of home appliances.\n\n"
                   "Please read the README to know what we recommend. I can help you find:\n"
                   "• Refrigerators and freezers\n"
                   "• Washing machines and dryers\n"
                   "• Air conditioners\n"
                   "• Microwaves and ovens\n"
                   "• Dishwashers\n"
                   "• And other home appliances\n\n"
                   "What home appliance are you looking for?")
        
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
            
            # Enhanced prompt template - more conversational and natural
            brands = ', '.join(self.knowledge_base.get_available_brands()[:10])
            prompt = f"""You are a friendly and helpful AI assistant that helps people find home appliances. Chat naturally and be conversational.

Available brands: {brands}
Context: {kb_context}

You can help users find products, answer questions about brands, and give recommendations. Be friendly, casual, and talk like a normal person would. Keep responses natural and not too formal.

User: {user_message}
Assistant:"""
        elif context:
            prompt = f"You're a friendly AI assistant helping with home appliances. Chat naturally.\n\nContext: {context}\n\nUser: {user_message}\nAssistant:"
        else:
            # Create a more conversational prompt
            prompt = f"You're a friendly AI assistant that helps people find home appliances. Chat naturally and be helpful.\n\nUser: {user_message}\nAssistant:"
        
        if self.use_llm and self.pipeline:
            try:
                # Generate response using LLM - allow longer, more natural responses
                response = self.pipeline(
                    prompt,
                    max_new_tokens=120,  # Increased for more natural conversation
                    num_return_sequences=1,
                    pad_token_id=50256,  # GPT-2 pad token
                    eos_token_id=50256,
                    temperature=0.8,  # Slightly higher for more natural variation
                    top_p=0.9
                )
                
                # Extract generated text
                generated_text = response[0]['generated_text']
                # Remove the prompt from the response
                if "Assistant:" in generated_text:
                    assistant_response = generated_text.split("Assistant:")[-1].strip()
                else:
                    # If no "Assistant:" marker, take text after the prompt
                    assistant_response = generated_text[len(prompt):].strip()
                
                # Clean up the response - allow longer, more natural responses
                # Take up to 2-3 sentences for more natural conversation
                sentences = assistant_response.split(".")
                if len(sentences) > 1:
                    # Take first 2-3 sentences for natural flow
                    assistant_response = ". ".join(sentences[:3]).strip()
                    if assistant_response and not assistant_response.endswith("."):
                        assistant_response += "."
                else:
                    assistant_response = assistant_response.split("\n")[0].strip()
                
                # Limit to reasonable length but allow more than before
                assistant_response = assistant_response[:300].strip()
                
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
        
        # Greetings - more natural and casual
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings', 'howdy']):
            return "Hey there! 👋 I'm here to help you find home appliances. What are you looking for?"
        
        # Thank you responses - more casual
        elif any(word in message_lower for word in ['thank', 'thanks', 'appreciate']):
            return "You're welcome! 😊 Happy to help. Let me know if you need anything else!"
        
        # Goodbye responses - more natural
        elif any(word in message_lower for word in ['goodbye', 'bye', 'see you', 'farewell', 'have a good day']):
            return "See you later! 👋 Feel free to come back if you need help finding products. Have a great day!"
        
        # How are you - more conversational
        elif any(phrase in message_lower for phrase in ['how are you', 'how are you doing', 'how\'s it going', 'how do you do']):
            return "I'm doing great, thanks! 😊 Ready to help you find some appliances. What can I help you with?"
        
        # What can you do / capabilities - more casual
        elif any(phrase in message_lower for phrase in ['what can you do', 'what do you do', 'what are your capabilities', 'what can you help with']):
            return "I can help you find home appliances like refrigerators, washing machines, air conditioners, and more. You can ask me to find products by type, brand, or budget. What are you looking for?"
        
        # Who are you / introduction - more natural
        elif any(phrase in message_lower for phrase in ['who are you', 'what are you', 'introduce yourself', 'tell me about yourself']):
            return "I'm an AI assistant that helps people find home appliances! 🤖 I can search through our product catalog and help you find what you need. What can I help you with today?"
        
        # Yes / confirmation - more casual
        elif any(word in message_lower for word in ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'correct', 'right']):
            return "Cool! 😊 What would you like to do next? I can help you find products or answer questions."
        
        # No / negative - more natural
        elif any(word in message_lower for word in ['no', 'nope', 'not really', 'not interested', 'don\'t want']):
            return "No worries! Is there something else I can help you with?"
        
        # Need help - more casual
        elif any(phrase in message_lower for phrase in ['i need help', 'can you help', 'help me', 'i need assistance', 'i\'m confused']):
            return "Sure thing! 😊 I can help you find products, search by brand or budget, or answer questions. What do you need help with?"
        
        # Positive feedback - more natural
        elif any(word in message_lower for word in ['great', 'awesome', 'cool', 'nice', 'good', 'excellent', 'perfect', 'wonderful']):
            return "Glad I could help! 😊 Anything else you'd like to know?"
        
        # Don't understand - more conversational
        elif any(phrase in message_lower for phrase in ['i don\'t understand', 'what do you mean', 'can you explain', 'i\'m confused', 'unclear']):
            return "No worries! I'm here to help you find home appliances. You can ask me to find specific products, search by brand, or filter by budget. What are you looking for?"
        
        # More information - more casual
        elif any(phrase in message_lower for phrase in ['tell me more', 'more information', 'more details', 'elaborate']):
            return "Sure! What would you like to know more about? I can tell you about products, brands, prices, or help you search for something."
        
        # Product-specific queries - more natural
        elif any(word in message_lower for word in ['refrigerator', 'fridge']):
            return "Sure! I can help you find refrigerators. 🧊 Just ask me to 'find me a refrigerator' and I'll show you options. You can also filter by brand or budget if you want!"
        elif any(word in message_lower for word in ['washing machine', 'washer']):
            return "Got it! I can help you find washing machines. 🌀 Try asking 'find me a washing machine' and I'll show you what's available."
        elif any(word in message_lower for word in ['air conditioner', 'ac']):
            return "Yep! I can help you find air conditioners. ❄️ Just ask me to find one and I'll show you options. You can filter by brand or budget too!"
        elif 'microwave' in message_lower:
            return "Sure thing! I can help you find microwaves. 🔥 Just ask me to 'find me a microwave' and I'll show you what's available."
        
        # Help / how to use - more casual
        elif any(word in message_lower for word in ['help', 'how', 'how to', 'how do i']):
            return "I can help you find home appliances! Just ask me things like:\n• 'Find me a refrigerator'\n• 'Show me washing machines under $500'\n• 'What brands do you have?'\n\nWhat are you looking for?"
        
        # Top rated - more natural
        elif any(phrase in message_lower for phrase in ['top rated', 'best products', 'highest rated', 'most popular']):
            return "I can show you top-rated products! ⭐ Just ask me to find a specific product type and I'll show you the best options. For example: 'find me a refrigerator' or 'show me top rated washing machines'."
        
        # Budget queries - more casual
        elif any(phrase in message_lower for phrase in ['under $', 'less than $', 'budget', 'cheap', 'affordable']):
            return "I can help you find products within your budget! 💰 Just ask like: 'find me a refrigerator under $500' or 'show me washing machines under $1000' and I'll filter the results for you."
        
        # Default response - more natural and conversational
        else:
            return "I'm here to help you find home appliances! 😊 You can ask me things like:\n• 'Find me a refrigerator'\n• 'Show me washing machines'\n• 'What brands are available?'\n\nWhat are you looking for?"
    
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

