"""
Product Recommendation Engine

This module implements a content-based recommendation system using TF-IDF vectorization
and cosine similarity. Supports both basic and advanced AI features (optional).
Includes Generative AI (LLM) integration using Sentence Transformers for semantic understanding.
"""

import logging
import os
import re
import shutil
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import Sentence Transformers for Generative AI features
try:
    from sentence_transformers import SentenceTransformer
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logging.warning("Sentence Transformers not available. Generative AI features disabled.")

# Try to import Hugging Face Transformers for LLM features (optional)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logging.warning("Transformers library not available. LLM text generation features disabled.")

# Try to import NLTK for advanced features
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from config import get_csv_path, get_log_level
except ImportError:
    def get_csv_path() -> str:
        return "home appliance skus lowes.csv"
    def get_log_level() -> str:
        return "INFO"

# Configure logging
log_level = getattr(logging, get_log_level().upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
MODELS_DIR = 'models'


def get_model_path(model_name: str = MODEL_NAME) -> str:
    """
    Get the local path for the model cache directory.
    
    Args:
        model_name: Name of the model to cache
        
    Returns:
        Path to the model directory
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODELS_DIR, model_name)
    return model_path


def ensure_model_cached(model_name: str = MODEL_NAME) -> str:
    """
    Ensure the model is cached locally. If not present, download it.
    
    Args:
        model_name: Name of the model to cache
        
    Returns:
        Path to the cached model directory
    """
    model_path = get_model_path(model_name)
    
    # Check if model directory exists and contains necessary files
    if os.path.exists(model_path):
        # Check for key files that indicate a complete model
        # Sentence Transformers uses config.json and either pytorch_model.bin or model.safetensors
        config_exists = os.path.exists(os.path.join(model_path, 'config.json'))
        model_file_exists = any(
            os.path.exists(os.path.join(model_path, f)) 
            for f in ['pytorch_model.bin', 'model.safetensors']
        )
        # Also check for modules.json which is required by SentenceTransformers
        modules_exists = os.path.exists(os.path.join(model_path, 'modules.json'))
        
        if config_exists and model_file_exists and modules_exists:
            logger.info(f"✓ Model found in cache: {model_path}")
            logger.info(f"  Using cached model (no download needed)")
            return model_path
        else:
            missing = []
            if not config_exists:
                missing.append('config.json')
            if not model_file_exists:
                missing.append('model file (pytorch_model.bin or model.safetensors)')
            if not modules_exists:
                missing.append('modules.json')
            logger.warning(f"Model directory exists but appears incomplete. Missing: {', '.join(missing)}")
            logger.warning("Re-downloading...")
            # Remove incomplete directory
            try:
                shutil.rmtree(model_path)
            except Exception as e:
                logger.warning(f"Could not remove incomplete model directory: {e}")
    
    # Model not found, download it
    logger.info(f"Model not found in cache. Downloading {model_name}...")
    logger.info(f"This may take a few minutes on first run. Model will be cached at: {model_path}")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.dirname(model_path)
    os.makedirs(models_dir, exist_ok=True)
    
    # Download model using SentenceTransformer
    # This will automatically cache it in the specified directory
    if GENAI_AVAILABLE:
        try:
            logger.info(f"Downloading model '{model_name}' from HuggingFace...")
            # Download and save model
            temp_model = SentenceTransformer(model_name)
            # Save to our custom location
            logger.info(f"Saving model to cache: {model_path}")
            temp_model.save(model_path)
            logger.info(f"✓ Model downloaded and cached successfully at: {model_path}")
            logger.info(f"  Model size: ~90MB. Future runs will use cached version (no download).")
            return model_path
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            # Fallback to default behavior (will use HuggingFace cache)
            logger.info("Falling back to default model loading...")
            return model_name
    else:
        logger.warning("Sentence Transformers not available. Cannot download model.")
        return model_name

# Initialize NLTK resources if available
if NLTK_AVAILABLE:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
    except Exception as e:
        logger.warning(f"Failed to initialize NLTK resources: {e}")
        NLTK_AVAILABLE = False

# Constants
DEFAULT_CSV_PATH = get_csv_path()
DEFAULT_TOP_K = 10
DEFAULT_SIMILAR_TOP_K = 8
BRAND_FILTER_TOP_N = 100
CANDIDATE_MULTIPLIER = 10
EXPANDED_CANDIDATE_MULTIPLIER = 20
SIMILAR_PRODUCT_MULTIPLIER = 5


class TextPreprocessor:
    """Text preprocessing with optional advanced features."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def preprocess_text(text: str, use_stemming: bool = False, use_lemmatization: bool = True) -> str:
        """Advanced text preprocessing with optional stemming/lemmatization."""
        if not NLTK_AVAILABLE:
            return TextPreprocessor.clean_text(text)
        
        text = TextPreprocessor.clean_text(text)
        if not text:
            return ""
        
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        
        processed_tokens = []
        for token in tokens:
            if token in stop_words:
                continue
            if use_lemmatization:
                token = lemmatizer.lemmatize(token)
            elif use_stemming:
                token = stemmer.stem(token)
            if len(token) > 2:
                processed_tokens.append(token)
        
        return ' '.join(processed_tokens)


class ProductRecommender:
    """
    A content-based product recommendation engine using TF-IDF and cosine similarity.
    
    Supports both basic and advanced AI features through optional parameters.
    
    Attributes:
        df (pd.DataFrame): The product dataset
        vectorizer (TfidfVectorizer): TF-IDF vectorizer for text processing
        tfidf_matrix: Sparse matrix of TF-IDF features
    """
    
    def __init__(
        self, 
        csv_path: str = DEFAULT_CSV_PATH,
        use_ngrams: bool = False,
        use_advanced_preprocessing: bool = False,
        use_genai: bool = False,
        feature_weights: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initialize the ProductRecommender with a dataset.
        
        Args:
            csv_path: Path to the CSV file containing product data
            use_ngrams: Whether to use n-gram features (1-3 grams) for better phrase matching
            use_advanced_preprocessing: Whether to use advanced text preprocessing (stemming/lemmatization)
            use_genai: Whether to use Generative AI (Sentence Transformers) for semantic embeddings
            feature_weights: Optional weights for features (default: product_name=0.7, brand=0.3)
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
        """
        self.df = pd.DataFrame()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.use_ngrams = use_ngrams
        self.use_advanced_preprocessing = use_advanced_preprocessing and NLTK_AVAILABLE
        self.use_genai = use_genai and GENAI_AVAILABLE
        self.feature_weights = feature_weights or {'product_name': 0.7, 'brand': 0.3}
        
        # Initialize Generative AI model if enabled
        self.genai_model = None
        self.genai_embeddings = None
        self.llm_model = None
        self.llm_tokenizer = None
        
        if self.use_genai:
            try:
                logger.info("Loading Generative AI model (Sentence Transformer)...")
                # Using all-MiniLM-L6-v2: Best small model for semantic similarity
                # - Free and open-source (Apache 2.0)
                # - Small size: ~90MB
                # - Fast inference: ~10-50ms per query
                # - Good performance: 384-dim embeddings
                # - Optimized for semantic search and similarity
                # Alternative models: 'all-mpnet-base-v2' (better but 420MB), 
                #                     'multi-qa-MiniLM-L6-cos-v1' (optimized for Q&A)
                
                # Check if model is cached locally, download if not
                model_path = ensure_model_cached(MODEL_NAME)
                
                # Verify we got a path, not just the model name
                # If we got the model name, it means cache check failed
                if model_path == MODEL_NAME:
                    logger.warning(f"Cache check returned model name instead of path. This should not happen.")
                    logger.warning(f"Will attempt to load from default location (may download)")
                    abs_model_path = None
                else:
                    abs_model_path = os.path.abspath(model_path)
                
                if abs_model_path and os.path.exists(abs_model_path) and os.path.exists(os.path.join(abs_model_path, 'config.json')):
                    # Model is in our cache, load it directly
                    logger.info(f"✓ Model found in local cache: {abs_model_path}")
                    logger.info(f"  Cache verified - all required files present")
                    logger.info(f"  Loading from cache (OFFLINE MODE - no download)")
                    
                    # Set environment variable to force offline mode
                    # This prevents HuggingFace from checking online
                    import os as os_module
                    original_hf_offline = os_module.environ.get('HF_HUB_OFFLINE', None)
                    os_module.environ['HF_HUB_OFFLINE'] = '1'
                    
                    try:
                        # Load model from local cache using absolute path
                        # HF_HUB_OFFLINE=1 ensures no online checks
                        self.genai_model = SentenceTransformer(abs_model_path)
                        logger.info(f"✓ Generative AI model ({MODEL_NAME}) loaded successfully from cache")
                        logger.info(f"  ✓ No download occurred - using cached model files only")
                    except Exception as cache_error:
                        logger.error(f"Error loading from cache: {cache_error}")
                        logger.warning("This should not happen if model cache is complete.")
                        # Try one more time
                        try:
                            self.genai_model = SentenceTransformer(abs_model_path)
                            logger.info(f"✓ Model loaded successfully (retry)")
                        except Exception as e2:
                            logger.error(f"Failed to load model from cache: {e2}")
                            raise
                    finally:
                        # Restore original environment variable
                        if original_hf_offline is not None:
                            os_module.environ['HF_HUB_OFFLINE'] = original_hf_offline
                        elif 'HF_HUB_OFFLINE' in os_module.environ:
                            del os_module.environ['HF_HUB_OFFLINE']
                else:
                    # Fallback: load from HuggingFace (shouldn't happen if caching works)
                    if abs_model_path:
                        logger.warning(f"Model cache path exists but config.json not found: {abs_model_path}")
                    else:
                        logger.warning(f"Model cache check failed. Loading from default location...")
                    logger.warning(f"⚠️  This will download the model if not in HuggingFace cache!")
                    logger.warning(f"⚠️  Expected cache location: {get_model_path(MODEL_NAME)}")
                    self.genai_model = SentenceTransformer(MODEL_NAME)
                    logger.info(f"Generative AI model ({MODEL_NAME}) loaded from default location")
            except Exception as e:
                logger.error(f"Failed to load Generative AI model: {e}")
                self.use_genai = False
            
            # Optionally load a text-generating LLM for query expansion
            # Using a small, free model that can run locally
            if LLM_AVAILABLE:
                try:
                    logger.info("Loading LLM for query understanding (optional)...")
                    # Using a small, free model - can be changed to other free models
                    # Options: 'gpt2', 'distilgpt2', 'microsoft/DialoGPT-small'
                    # For now, we'll use it only if explicitly needed
                    self.llm_model = None  # Load on demand if needed
                    logger.info("LLM support available")
                except Exception as e:
                    logger.warning(f"LLM not loaded (optional): {e}")
        
        # Validate file exists
        if not os.path.exists(csv_path):
            logger.error(f"Dataset file not found: {csv_path}")
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        
        # Load dataset
        try:
            logger.info(f"Loading dataset from {csv_path}")
            self.df = pd.read_csv(csv_path)
            self.df.columns = [c.strip().lower() for c in self.df.columns]
            logger.info(f"Loaded {len(self.df)} products from dataset")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}", exc_info=True)
            self.df = pd.DataFrame()
            raise

        # Initialize vectorizer if data is available
        if not self.df.empty:
            self._preprocess_text_data()
            self._initialize_vectorizer()
            # Generate GenAI embeddings if enabled
            if self.use_genai:
                self._generate_genai_embeddings()
        else:
            logger.warning("Dataset is empty, vectorizer not initialized")
    
    def _preprocess_text_data(self) -> None:
        """Preprocess text columns if advanced preprocessing is enabled."""
        if self.use_advanced_preprocessing:
            logger.info("Applying advanced text preprocessing...")
            if 'product_name' in self.df.columns:
                self.df['product_name_processed'] = self.df['product_name'].apply(
                    lambda x: TextPreprocessor.preprocess_text(str(x) if pd.notna(x) else "")
                )
            if 'brand' in self.df.columns:
                self.df['brand_processed'] = self.df['brand'].apply(
                    lambda x: TextPreprocessor.clean_text(str(x) if pd.notna(x) else "")
                )
        else:
            # Simple preprocessing
            if 'product_name' in self.df.columns:
                self.df['product_name_processed'] = self.df['product_name'].apply(
                    lambda x: TextPreprocessor.clean_text(str(x) if pd.notna(x) else "")
                )
            if 'brand' in self.df.columns:
                self.df['brand_processed'] = self.df['brand'].apply(
                    lambda x: TextPreprocessor.clean_text(str(x) if pd.notna(x) else "")
                )
    
    def _initialize_vectorizer(self) -> None:
        """Initialize TF-IDF vectorizer with product data."""
        try:
            # Combine features with weights
            text_data = []
            for _, row in self.df.iterrows():
                name = row.get('product_name_processed', row.get('product_name', ''))
                brand = row.get('brand_processed', row.get('brand', ''))
                
                if self.use_ngrams or self.use_advanced_preprocessing:
                    # Weight features by repeating them
                    name_weighted = ' '.join([name] * int(self.feature_weights.get('product_name', 0.7) * 10))
                    brand_weighted = ' '.join([brand] * int(self.feature_weights.get('brand', 0.3) * 10))
                    combined = f"{name_weighted} {brand_weighted}".strip()
                else:
                    # Simple concatenation for basic mode
                    combined = f"{name} {brand}".strip()
                
                text_data.append(combined)
            
            # Configure vectorizer parameters
            if self.use_ngrams or self.use_advanced_preprocessing:
                # Advanced parameters
                vectorizer_params = {
                    'stop_words': 'english',
                    'max_features': 5000,
                    'min_df': 2,
                    'max_df': 0.95,
                    'ngram_range': (1, 3) if self.use_ngrams else (1, 1),
                    'sublinear_tf': True,
                    'norm': 'l2'
                }
            else:
                # Basic parameters
                vectorizer_params = {
                    'stop_words': 'english'
                }
            
            self.vectorizer = TfidfVectorizer(**vectorizer_params)
            self.tfidf_matrix = self.vectorizer.fit_transform(text_data)
            logger.info("TF-IDF vectorizer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vectorizer: {e}", exc_info=True)
            self.vectorizer = None
            self.tfidf_matrix = None
    
    def _generate_genai_embeddings(self) -> None:
        """Generate semantic embeddings using Generative AI (Sentence Transformers)."""
        if not self.use_genai or self.genai_model is None:
            return
        
        try:
            logger.info("Generating semantic embeddings using Generative AI...")
            # Combine product name and brand for embedding
            text_data = []
            for _, row in self.df.iterrows():
                name = str(row.get("product_name", "")) if pd.notna(row.get("product_name")) else ""
                brand = str(row.get("brand", "")) if pd.notna(row.get("brand")) else ""
                combined = f"{name} {brand}".strip()
                text_data.append(combined)
            
            # Generate embeddings
            self.genai_embeddings = self.genai_model.encode(
                text_data,
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )
            logger.info(f"Generated {len(self.genai_embeddings)} semantic embeddings")
        except Exception as e:
            logger.error(f"Error generating GenAI embeddings: {e}", exc_info=True)
            self.use_genai = False
            self.genai_embeddings = None
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        synonyms = {
            'fridge': 'refrigerator',
            'refrigerator': 'fridge',
            'washer': 'washing machine',
            'washing machine': 'washer',
            'ac': 'air conditioner',
            'air conditioner': 'ac',
            'microwave': 'microwave oven',
            'microwave oven': 'microwave'
        }
        expanded = query.lower()
        for key, value in synonyms.items():
            if key in expanded:
                expanded = expanded.replace(key, f"{key} {value}")
        return expanded
    
    def _calculate_weighted_similarity(self, similarity_scores: np.ndarray) -> np.ndarray:
        """Calculate weighted similarity scores with normalization."""
        weighted_scores = similarity_scores.copy()
        if weighted_scores.max() > 0:
            weighted_scores = weighted_scores / weighted_scores.max()
        return weighted_scores
    
    def _apply_diversity_filter(
        self,
        indices: np.ndarray,
        scores: np.ndarray,
        diversity_weight: float,
        top_k: int
    ) -> np.ndarray:
        """Apply diversity filter to ensure variety in recommendations."""
        selected = []
        brands_seen = set()
        for idx in indices:
            if len(selected) >= top_k:
                break
            brand = str(self.df.iloc[idx].get("brand", "")).lower()
            if brand not in brands_seen or np.random.random() > diversity_weight:
                selected.append(idx)
                brands_seen.add(brand)
        return np.array(selected[:top_k])

    def get_available_brands(self, product_query: str = "") -> List[str]:
        """Get sorted list of unique brands from the dataset, optionally filtered by product type."""
        if self.df.empty:
            logger.warning("Dataset is empty, returning empty brand list")
            return []
        
        if not product_query or product_query.strip() == "":
            brands = self.df["brand"].dropna().unique().tolist()
            return sorted([b for b in brands if b and str(b).strip()])
        
        if self.vectorizer is None:
            logger.warning("Vectorizer not initialized, returning empty brand list")
            return []
        
        try:
            # Preprocess query if advanced features enabled
            if self.use_advanced_preprocessing:
                query_text = TextPreprocessor.preprocess_text(product_query.strip())
                query_text = self._expand_query(query_text)
            else:
                query_text = product_query.strip()
            
            query_vec = self.vectorizer.transform([query_text])
            similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = similarity_scores.argsort()[-BRAND_FILTER_TOP_N:][::-1]
            relevant_products = self.df.iloc[top_indices]
            brands = relevant_products["brand"].dropna().unique().tolist()
            filtered_brands = sorted([b for b in brands if b and str(b).strip()])
            logger.debug(f"Found {len(filtered_brands)} brands for query: {product_query}")
            return filtered_brands
        except Exception as e:
            logger.warning(f"Error filtering brands, falling back to all brands: {e}")
            brands = self.df["brand"].dropna().unique().tolist()
            return sorted([b for b in brands if b and str(b).strip()])
    
    def _format_price(self, price_value: Union[str, float, None]) -> str:
        """Format price value as currency string."""
        try:
            if price_value and pd.notna(price_value):
                return f"${float(price_value):.2f}"
            return "N/A"
        except (ValueError, TypeError):
            return "N/A"
    
    def _format_rating(self, bestseller_rank: Union[str, float, int, None]) -> str:
        """Format bestseller rank as rating string."""
        try:
            if bestseller_rank and pd.notna(bestseller_rank):
                return f"Rank #{int(float(bestseller_rank))}"
            return "N/A"
        except (ValueError, TypeError):
            return "N/A"
    
    def _extract_product_data(self, row: pd.Series, similarity_score: Optional[float] = None) -> Dict[str, Union[str, float]]:
        """Extract and format product data from a DataFrame row."""
        name = str(row.get("product_name", "Unknown Product")) if pd.notna(row.get("product_name")) else "Unknown Product"
        brand = str(row.get("brand", "Unknown")) if pd.notna(row.get("brand")) else "Unknown"
        
        price_current = row.get("price_current", "")
        price_retail = row.get("price_retail", "")
        price_value = price_current if price_current else price_retail
        
        price = self._format_price(price_value)
        try:
            price_num = float(price_value) if price_value and pd.notna(price_value) else float('inf')
        except (ValueError, TypeError):
            price_num = float('inf')
        
        bestseller_rank = row.get("bestseller_rank", "")
        rating = self._format_rating(bestseller_rank)
        try:
            rating_num = int(float(bestseller_rank)) if bestseller_rank and pd.notna(bestseller_rank) else float('inf')
        except (ValueError, TypeError):
            rating_num = float('inf')
        
        product_url = str(row.get("product_url", "")) if pd.notna(row.get("product_url")) and row.get("product_url") else ""
        
        result = {
            "name": name,
            "brand": brand,
            "price": price,
            "rating": rating,
            "url": product_url,
            "_price_num": price_num,
            "_rating_num": rating_num
        }
        
        # Add similarity score if provided (for advanced mode)
        if similarity_score is not None:
            result["similarity_score"] = f"{similarity_score:.3f}"
            result["_similarity_num"] = similarity_score
        
        return result
    
    def get_similar_products(self, product_name: str, brand: str, top_k: int = DEFAULT_SIMILAR_TOP_K) -> List[Dict[str, str]]:
        """Get similar products based on a specific product name and brand."""
        if self.df.empty:
            logger.warning("Cannot get similar products: dataset not available")
            return []
        if not self.use_genai and self.vectorizer is None:
            logger.warning("Cannot get similar products: vectorizer not available")
            return []
        if self.use_genai and self.genai_embeddings is None:
            logger.warning("Cannot get similar products: GenAI embeddings not available")
            return []
        
        query_text = f"{product_name} {brand}".strip()
        
        # Preprocess query if advanced features enabled
        if self.use_advanced_preprocessing:
            query_text = TextPreprocessor.preprocess_text(query_text)
            query_text = self._expand_query(query_text)
        else:
            query_text = TextPreprocessor.clean_text(query_text)
        
        # Use Generative AI embeddings if available, otherwise use TF-IDF
        if self.use_genai and self.genai_embeddings is not None:
            query_embedding = self.genai_model.encode([query_text], convert_to_numpy=True)
            similarity_scores = cosine_similarity(query_embedding, self.genai_embeddings).flatten()
        else:
            query_vec = self.vectorizer.transform([query_text])
            similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Apply weighted similarity if advanced features enabled
        if self.use_ngrams or self.use_advanced_preprocessing or self.use_genai:
            similarity_scores = self._calculate_weighted_similarity(similarity_scores)
        
        max_candidates = min(len(self.df), top_k * SIMILAR_PRODUCT_MULTIPLIER)
        top_indices = similarity_scores.argsort()[-max_candidates:][::-1]
        
        results = self.df.iloc[top_indices]
        if results.empty:
            logger.info(f"No similar products found for {product_name} ({brand})")
            return []
        
        seen_products = set()
        current_product_key = (product_name.lower().strip(), brand.lower().strip())
        output = []
        
        for _, row in results.iterrows():
            if len(output) >= top_k:
                break
            
            similarity = similarity_scores[row.name] if (self.use_ngrams or self.use_advanced_preprocessing) else None
            product_data = self._extract_product_data(row, similarity)
            product_key = (product_data["name"].lower().strip(), product_data["brand"].lower().strip())
            
            if product_key == current_product_key:
                continue
            if product_key in seen_products:
                continue
            
            seen_products.add(product_key)
            product_data.pop("_price_num", None)
            product_data.pop("_rating_num", None)
            product_data.pop("_similarity_num", None)
            output.append(product_data)
        
        logger.info(f"Found {len(output)} similar products for {product_name} ({brand})")
        return output
    
    def _process_search_results(
        self, 
        results: pd.DataFrame, 
        seen_products: set, 
        budget_max: Optional[float],
        top_k: int,
        similarity_scores: Optional[np.ndarray] = None
    ) -> List[Dict[str, Union[str, float]]]:
        """Process search results and extract product data with filtering."""
        output = []
        for _, row in results.iterrows():
            if len(output) >= top_k:
                break
            
            similarity = similarity_scores[row.name] if similarity_scores is not None else None
            product_data = self._extract_product_data(row, similarity)
            product_key = (product_data["name"].lower().strip(), product_data["brand"].lower().strip())
            
            if product_key in seen_products:
                continue
            
            if budget_max is not None:
                price_num = product_data.get("_price_num", float('inf'))
                if price_num != float('inf') and price_num > budget_max:
                    continue
            
            seen_products.add(product_key)
            output.append(product_data)
        return output
    
    def recommend(
        self, 
        product_query: str, 
        brand_query: str = "", 
        budget_max: Optional[float] = None, 
        top_k: int = DEFAULT_TOP_K, 
        sort_by: str = "Similarity (Default)",
        diversity_weight: float = 0.0
    ) -> Union[List[Dict[str, str]], List[str]]:
        """
        Get product recommendations based on query, brand, and budget constraints.
        
        Args:
            product_query: Product type or name to search for
            brand_query: Optional brand filter
            budget_max: Optional maximum price constraint
            top_k: Number of recommendations to return
            sort_by: Sorting method ("Similarity (Default)", "Price: Low to High", 
                     "Price: High to Low", "Rating: Best First")
            diversity_weight: Weight for diversity in results (0-1, only used if advanced features enabled)
        
        Returns:
            List of product dictionaries or error message list
        """
        if self.df.empty:
            logger.error("Dataset not available")
            return ["Dataset not loaded properly."]
        if not self.use_genai and self.vectorizer is None:
            logger.error("Vectorizer not available")
            return ["Dataset not loaded properly."]
        if self.use_genai and self.genai_embeddings is None:
            logger.error("GenAI embeddings not available")
            return ["Generative AI embeddings not loaded properly."]

        # Preprocess and expand query if advanced features enabled
        if self.use_advanced_preprocessing:
            expanded_query = self._expand_query(product_query)
            query_text = TextPreprocessor.preprocess_text(f"{expanded_query} {brand_query}".strip())
        else:
            query_text = f"{product_query} {brand_query}".strip()
        
        logger.info(f"Searching for: {query_text} (budget: {budget_max}, top_k: {top_k})")
        
        # Use Generative AI embeddings if available, otherwise use TF-IDF
        if self.use_genai and self.genai_embeddings is not None:
            # Generate query embedding using GenAI
            query_embedding = self.genai_model.encode([query_text], convert_to_numpy=True)
            # Calculate cosine similarity with GenAI embeddings
            similarity_scores = cosine_similarity(query_embedding, self.genai_embeddings).flatten()
            logger.info("Using Generative AI semantic embeddings for similarity calculation")
        else:
            # Use traditional TF-IDF
            query_vec = self.vectorizer.transform([query_text])
            similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Apply weighted similarity if advanced features enabled
        if self.use_ngrams or self.use_advanced_preprocessing or self.use_genai:
            similarity_scores = self._calculate_weighted_similarity(similarity_scores)
        
        max_candidates = min(len(self.df), top_k * CANDIDATE_MULTIPLIER)
        top_indices = similarity_scores.argsort()[-max_candidates:][::-1]
        
        # Apply diversity filter if enabled
        if diversity_weight > 0 and (self.use_ngrams or self.use_advanced_preprocessing or self.use_genai):
            top_indices = self._apply_diversity_filter(top_indices, similarity_scores, diversity_weight, top_k)

        results = self.df.iloc[top_indices]
        if results.empty:
            logger.info(f"No relevant products found for '{product_query}'")
            return [f"No relevant products found for '{product_query}'."]

        seen_products = set()
        output = self._process_search_results(results, seen_products, budget_max, top_k, similarity_scores)
        
        # If we don't have enough results, try to get more
        if len(output) < top_k and max_candidates < len(self.df):
            try:
                expanded_candidates = min(len(self.df), top_k * EXPANDED_CANDIDATE_MULTIPLIER)
                all_indices = similarity_scores.argsort()[-expanded_candidates:][::-1]
                processed_indices_set = set(top_indices)
                new_indices = [idx for idx in all_indices if idx not in processed_indices_set]
                
                if new_indices:
                    additional_results = self.df.iloc[new_indices]
                    additional_output = self._process_search_results(
                        additional_results, seen_products, budget_max, top_k - len(output), similarity_scores
                    )
                    output.extend(additional_output)
            except Exception as e:
                logger.warning(f"Error in expanded search: {e}")
        
        # Sort the results
        if sort_by == "Price: Low to High":
            output.sort(key=lambda x: x.get("_price_num", float('inf')))
        elif sort_by == "Price: High to Low":
            output.sort(
                key=lambda x: x.get("_price_num") if x.get("_price_num") != float('inf') else float('-inf'), 
                reverse=True
            )
        elif sort_by == "Rating: Best First":
            output.sort(key=lambda x: x.get("_rating_num", float('inf')))
        
        # Remove sorting helper keys before returning
        for item in output:
            item.pop("_price_num", None)
            item.pop("_rating_num", None)
            item.pop("_similarity_num", None)
        
        logger.info(f"Returning {len(output)} recommendations")
        return output
