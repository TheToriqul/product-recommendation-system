"""
Product Recommendation Engine

Modern content-based recommendation system with:
- BM25 algorithm (industry-standard keyword search)
- Semantic embeddings (Sentence Transformers for semantic understanding)
- Hybrid search (combines BM25 + semantic for optimal results)
- Faceted filtering (modern e-commerce style)

Supports both basic and advanced AI features with progressive enhancement.
"""

import logging
import os
import re
import shutil
import hashlib
import pickle
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math

# Try to import Sentence Transformers for Generative AI features
try:
    from sentence_transformers import SentenceTransformer
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logging.warning("Sentence Transformers not available. Generative AI features disabled.")

# Note: LLM features removed as they were not being used

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
    from src.core.config import get_csv_path, get_log_level
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
EMBEDDINGS_CACHE_DIR = 'embeddings_cache'


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
        import ssl
        import os
        
        # Fix SSL certificate issues on macOS (common problem)
        # Try multiple approaches to handle SSL certificate verification
        try:
            # Method 1: Try using certifi if available (commonly installed with requests)
            try:
                import certifi
                cert_path = certifi.where()
                os.environ['SSL_CERT_FILE'] = cert_path
                os.environ['REQUESTS_CA_BUNDLE'] = cert_path
                logger.debug("Using certifi certificates for SSL")
            except ImportError:
                # Method 2: Disable SSL verification for NLTK downloads only (less secure but works)
                # This is acceptable for downloading NLTK data which is public/open source
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context
                    logger.debug("Using unverified SSL context for NLTK downloads")
        except Exception as ssl_error:
            logger.debug(f"SSL configuration attempt failed: {ssl_error}")
        
        # Download NLTK resources with individual error handling
        nltk_resources_available = {'punkt': False, 'stopwords': False, 'wordnet': False}
        
        for resource in ['punkt', 'stopwords', 'wordnet']:
            try:
                nltk.download(resource, quiet=True)
                nltk_resources_available[resource] = True
            except Exception as e:
                logger.debug(f"Could not download NLTK resource '{resource}': {e}")
        
        # Only initialize if all required resources are available
        if all(nltk_resources_available.values()):
            try:
                stemmer = PorterStemmer()
                lemmatizer = WordNetLemmatizer()
                stop_words = set(stopwords.words('english'))
                logger.info("✓ NLTK resources initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NLTK resources after download: {e}")
                NLTK_AVAILABLE = False
        else:
            missing = [r for r, available in nltk_resources_available.items() if not available]
            logger.warning(f"NLTK data downloads failed for: {', '.join(missing)} (SSL/certificate issues)")
            logger.info("Application will continue with basic preprocessing and GenAI features (recommended).")
            NLTK_AVAILABLE = False
    except Exception as e:
        logger.warning(f"Failed to initialize NLTK resources: {e}")
        logger.info("Application will continue with basic preprocessing and GenAI features (recommended).")
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


class BM25:
    """
    BM25 (Best Matching 25) - Modern industry-standard ranking algorithm.
    Better than TF-IDF for keyword search, used by Elasticsearch, Solr, and major search engines.
    
    BM25 addresses TF-IDF limitations:
    - Better term frequency saturation (prevents over-weighting frequent terms)
    - Document length normalization (handles varying document sizes)
    - Tuned parameters (k1, b) for optimal ranking
    """
    
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with a corpus of documents.
        
        Args:
            corpus: List of document strings
            k1: Term frequency saturation parameter (default 1.5)
            b: Document length normalization parameter (default 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_freqs = []
        self.idf = {}
        self.avgdl = 0
        self._initialize(corpus)
    
    def _initialize(self, corpus: List[str]) -> None:
        """Initialize BM25 statistics from corpus."""
        # Tokenize documents
        doc_tokens = []
        for doc in corpus:
            tokens = doc.lower().split()
            doc_tokens.append(tokens)
            self.doc_freqs.append(Counter(tokens))
        
        # Calculate average document length
        self.avgdl = sum(len(tokens) for tokens in doc_tokens) / len(doc_tokens) if doc_tokens else 0
        
        # Calculate IDF (Inverse Document Frequency)
        df = Counter()
        for doc_freq in self.doc_freqs:
            for term in doc_freq:
                df[term] += 1
        
        num_docs = len(corpus)
        for term, freq in df.items():
            # Standard IDF formula: log((N - n + 0.5) / (n + 0.5))
            # where N = total docs, n = docs containing term
            self.idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5))
    
    def get_scores(self, query: str) -> np.ndarray:
        """
        Calculate BM25 scores for all documents given a query.
        
        Args:
            query: Search query string
            
        Returns:
            Array of BM25 scores for each document
        """
        query_terms = query.lower().split()
        scores = np.zeros(len(self.corpus))
        
        for i, doc_freq in enumerate(self.doc_freqs):
            doc_len = sum(doc_freq.values())
            score = 0
            
            for term in query_terms:
                if term in doc_freq:
                    # BM25 formula
                    tf = doc_freq[term]
                    idf = self.idf.get(term, 0)
                    
                    # BM25 scoring: idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    score += idf * (numerator / denominator) if denominator > 0 else 0
            
            scores[i] = score
        
        return scores


class ProductRecommender:
    """
    Modern content-based product recommendation engine with:
    - BM25 algorithm (industry standard for keyword search)
    - Semantic embeddings (Sentence Transformers for semantic understanding)
    - Hybrid search (combines BM25 + semantic for best results)
    - Faceted filtering (modern e-commerce style)
    
    Attributes:
        df (pd.DataFrame): The product dataset
        vectorizer (TfidfVectorizer): TF-IDF vectorizer (fallback)
        tfidf_matrix: Sparse matrix of TF-IDF features
        bm25: BM25 ranking algorithm instance
    """
    
    def __init__(
        self, 
        csv_path: str = DEFAULT_CSV_PATH,
        use_ngrams: bool = False,
        use_advanced_preprocessing: bool = False,
        use_genai: bool = False,
        feature_weights: Optional[Dict[str, float]] = None,
        load_models_immediately: bool = True
    ) -> None:
        """
        Initialize the ProductRecommender with a dataset.
        
        Args:
            csv_path: Path to the CSV file containing product data
            use_ngrams: Whether to use n-gram features (1-3 grams) for better phrase matching
            use_advanced_preprocessing: Whether to use advanced text preprocessing (stemming/lemmatization)
            use_genai: Whether to use Generative AI (Sentence Transformers) for semantic embeddings
            feature_weights: Optional weights for features (default: product_name=0.7, brand=0.3)
            load_models_immediately: If False, only load dataset. Models/embeddings loaded via load_models_and_embeddings()
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
        """
        self.df = pd.DataFrame()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.bm25: Optional[BM25] = None  # Modern BM25 algorithm
        self.use_ngrams = use_ngrams
        self.use_advanced_preprocessing = use_advanced_preprocessing and NLTK_AVAILABLE
        self.use_genai = use_genai and GENAI_AVAILABLE
        self.feature_weights = feature_weights or {'product_name': 0.7, 'brand': 0.3}
        self.use_hybrid_search = True  # Enable hybrid search (BM25 + semantic)
        self.hybrid_weight_bm25 = 0.4  # Weight for BM25 in hybrid search
        self.hybrid_weight_semantic = 0.6  # Weight for semantic in hybrid search
        
        # Initialize Generative AI model if enabled
        self.genai_model = None
        self.genai_embeddings = None
        self._models_loaded = False
        
        # Load dataset first (fast)
        if not os.path.exists(csv_path):
            logger.error(f"Dataset file not found: {csv_path}")
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        
        try:
            logger.info(f"Loading dataset from {csv_path}")
            self.df = pd.read_csv(csv_path)
            self.df.columns = [c.strip().lower() for c in self.df.columns]
            logger.info(f"Loaded {len(self.df)} products from dataset")
            
            # Do minimal preprocessing immediately so get_available_brands() works
            # This is fast and needed for brand dropdown to work
            if 'product_name' in self.df.columns:
                self.df['product_name_processed'] = (
                    self.df['product_name']
                    .astype(str)
                    .str.lower()
                    .str.replace(r'[^a-z0-9\s]', ' ', regex=True)
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.strip()
                    .replace('nan', '')
                )
            if 'brand' in self.df.columns:
                self.df['brand_processed'] = (
                    self.df['brand']
                    .astype(str)
                    .str.lower()
                    .str.replace(r'[^a-z0-9\s]', ' ', regex=True)
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.strip()
                    .replace('nan', '')
                )
        except Exception as e:
            logger.error(f"Error loading dataset: {e}", exc_info=True)
            self.df = pd.DataFrame()
            raise
        
        # Load models/embeddings if requested (can be deferred for background loading)
        if load_models_immediately:
            self._load_models_and_embeddings()
        else:
            logger.info("Models/embeddings loading deferred - will be loaded in background")
    
    def _load_models_and_embeddings(self) -> None:
        """Load AI models and generate embeddings. Can be called in background thread."""
        if self._models_loaded:
            return
        
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
        
        # Initialize vectorizer and BM25 if data is available
        if not self.df.empty:
            # If GenAI is enabled, prioritize loading embeddings cache first (faster)
            # Do minimal preprocessing first, then try cache
            if self.use_genai:
                # Do minimal preprocessing first (needed for hash calculation)
                self._preprocess_text_data()
                # Try to load embeddings cache (fast path)
                cached_embeddings = self._load_cached_embeddings()
                if cached_embeddings is not None:
                    self.genai_embeddings = cached_embeddings
                    logger.info("✓ Embeddings loaded from cache - initializing BM25 for hybrid search")
                    # Initialize BM25 for hybrid search (even with cached embeddings)
                    self._initialize_bm25()
                    # Keep TF-IDF as fallback
                    self._initialize_vectorizer()
                else:
                    # Cache miss - need to do full initialization
                    self._initialize_vectorizer()
                    self._initialize_bm25()
                    self._generate_genai_embeddings()
            else:
                # No GenAI - do standard initialization with BM25
                self._preprocess_text_data()
                self._initialize_vectorizer()
                self._initialize_bm25()
        else:
            logger.warning("Dataset is empty, vectorizer not initialized")
        
        self._models_loaded = True
        logger.info("✓ Models and embeddings loaded successfully (BM25 + Semantic Hybrid Search enabled)")
    
    def load_models_and_embeddings(self) -> None:
        """
        Public method to load models and embeddings in background thread.
        This method is thread-safe and can be called after initial dataset load.
        """
        self._load_models_and_embeddings()
    
    def _preprocess_text_data(self) -> None:
        """Preprocess text columns if advanced preprocessing is enabled (optimized with vectorization)."""
        # Skip if already preprocessed (done during dataset load for immediate brand access)
        if 'product_name_processed' in self.df.columns and 'brand_processed' in self.df.columns:
            return
        
        # Use vectorized operations for faster processing
        if self.use_advanced_preprocessing:
            logger.info("Applying advanced text preprocessing...")
            if 'product_name' in self.df.columns:
                # Convert to string first, then apply preprocessing
                self.df['product_name_processed'] = self.df['product_name'].astype(str).apply(
                    TextPreprocessor.preprocess_text
                )
            if 'brand' in self.df.columns:
                self.df['brand_processed'] = self.df['brand'].astype(str).apply(
                    TextPreprocessor.clean_text
                )
        else:
            # Simple preprocessing - use vectorized string operations for speed
            if 'product_name' in self.df.columns:
                # Fast vectorized cleaning
                self.df['product_name_processed'] = (
                    self.df['product_name']
                    .astype(str)
                    .str.lower()
                    .str.replace(r'[^a-z0-9\s]', ' ', regex=True)
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.strip()
                    .replace('nan', '')
                )
            if 'brand' in self.df.columns:
                self.df['brand_processed'] = (
                    self.df['brand']
                    .astype(str)
                    .str.lower()
                    .str.replace(r'[^a-z0-9\s]', ' ', regex=True)
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.strip()
                    .replace('nan', '')
                )
    
    def _initialize_bm25(self) -> None:
        """Initialize BM25 algorithm for modern keyword search."""
        try:
            if self.df.empty:
                return
            
            # Prepare text data for BM25 (product name + brand)
            name_col = self.df.get('product_name_processed', self.df.get('product_name', ''))
            brand_col = self.df.get('brand_processed', self.df.get('brand', ''))
            
            # Fill NaN values
            name_col = name_col.fillna('')
            brand_col = brand_col.fillna('')
            
            # Combine product name and brand for BM25 indexing
            text_data = (name_col + ' ' + brand_col).str.strip().tolist()
            
            # Initialize BM25 with optimized parameters
            self.bm25 = BM25(text_data, k1=1.5, b=0.75)
            logger.info("✓ BM25 algorithm initialized (modern keyword search)")
        except Exception as e:
            logger.warning(f"Error initializing BM25: {e}")
            self.bm25 = None
    
    def _initialize_vectorizer(self) -> None:
        """Initialize TF-IDF vectorizer with product data (optimized with vectorization)."""
        try:
            # Use vectorized operations for faster text combination
            name_col = self.df.get('product_name_processed', self.df.get('product_name', ''))
            brand_col = self.df.get('brand_processed', self.df.get('brand', ''))
            
            # Fill NaN values
            name_col = name_col.fillna('')
            brand_col = brand_col.fillna('')
            
            if self.use_ngrams or self.use_advanced_preprocessing:
                # Weight features by repeating them (vectorized)
                name_weight = int(self.feature_weights.get('product_name', 0.7) * 10)
                brand_weight = int(self.feature_weights.get('brand', 0.3) * 10)
                # Use vectorized string operations
                name_weighted = (name_col + ' ') * name_weight
                brand_weighted = (brand_col + ' ') * brand_weight
                text_data = (name_weighted + brand_weighted).str.strip().tolist()
            else:
                # Simple concatenation for basic mode (vectorized)
                text_data = (name_col + ' ' + brand_col).str.strip().tolist()
            
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
    
    def _get_dataset_hash(self) -> str:
        """Generate a hash of the dataset to use as cache key."""
        # Create a hash based on dataset content and size (optimized for speed)
        # Use first and last few rows + total count for faster hashing
        n_rows = len(self.df)
        if n_rows == 0:
            return hashlib.md5(b"empty").hexdigest()
        
        # Sample strategy: use first 10, last 10, and middle rows for hash
        sample_size = min(30, n_rows)
        if n_rows <= sample_size:
            sample_indices = range(n_rows)
        else:
            # Take first 10, last 10, and evenly spaced middle rows
            step = max(1, (n_rows - 20) // 10)
            sample_indices = list(range(10)) + list(range(10, n_rows - 10, step))[:10] + list(range(n_rows - 10, n_rows))
        
        sample_data = []
        for idx in sample_indices:
            name = str(self.df.iloc[idx].get("product_name", "")) if pd.notna(self.df.iloc[idx].get("product_name")) else ""
            brand = str(self.df.iloc[idx].get("brand", "")) if pd.notna(self.df.iloc[idx].get("brand")) else ""
            sample_data.append(f"{name}|{brand}")
        
        dataset_str = f"{n_rows}_{''.join(sample_data)}"
        return hashlib.md5(dataset_str.encode()).hexdigest()
    
    def _get_embeddings_cache_path(self) -> str:
        """Get the path to the embeddings cache file."""
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), EMBEDDINGS_CACHE_DIR)
        os.makedirs(cache_dir, exist_ok=True)
        dataset_hash = self._get_dataset_hash()
        cache_filename = f"embeddings_{dataset_hash}.pkl"
        return os.path.join(cache_dir, cache_filename)
    
    def _load_cached_embeddings(self) -> Optional[np.ndarray]:
        """Load cached embeddings if they exist and are valid (optimized for speed)."""
        # Note: This can be called before self.df is fully loaded, so we need to handle that
        if not hasattr(self, 'df') or self.df.empty:
            return None
            
        cache_path = self._get_embeddings_cache_path()
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            logger.info(f"Loading cached embeddings...")
            # Use faster pickle protocol
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Verify cache is valid (check shape matches dataset)
            if isinstance(cached_data, np.ndarray) and len(cached_data) == len(self.df):
                logger.info(f"✓ Loaded {len(cached_data)} cached embeddings (fast startup!)")
                return cached_data
            else:
                logger.warning(f"Cached embeddings shape mismatch. Regenerating...")
                return None
        except Exception as e:
            logger.warning(f"Error loading cached embeddings: {e}. Regenerating...")
            return None
    
    def _save_embeddings_cache(self, embeddings: np.ndarray) -> None:
        """Save embeddings to cache for faster future startups."""
        cache_path = self._get_embeddings_cache_path()
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"✓ Saved embeddings cache to: {cache_path}")
        except Exception as e:
            logger.warning(f"Could not save embeddings cache: {e}")
    
    def _generate_genai_embeddings(self) -> None:
        """Generate semantic embeddings using Generative AI (Sentence Transformers) with caching."""
        if not self.use_genai or self.genai_model is None:
            return
        
        # Try to load from cache first
        cached_embeddings = self._load_cached_embeddings()
        if cached_embeddings is not None:
            self.genai_embeddings = cached_embeddings
            return
        
        # Cache miss - generate embeddings
        try:
            logger.info("Generating semantic embeddings using Generative AI...")
            logger.info("This may take 1-2 minutes on first run. Embeddings will be cached for future use.")
            # Combine product name and brand for embedding
            text_data = []
            for _, row in self.df.iterrows():
                name = str(row.get("product_name", "")) if pd.notna(row.get("product_name")) else ""
                brand = str(row.get("brand", "")) if pd.notna(row.get("brand")) else ""
                combined = f"{name} {brand}".strip()
                text_data.append(combined)
            
            # Generate embeddings with optimized batch size
            self.genai_embeddings = self.genai_model.encode(
                text_data,
                show_progress_bar=True,
                batch_size=64,  # Increased batch size for faster processing
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better cosine similarity
            )
            logger.info(f"Generated {len(self.genai_embeddings)} semantic embeddings")
            
            # Save to cache for future use
            self._save_embeddings_cache(self.genai_embeddings)
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
        """
        Get sorted list of unique brands from the dataset, optionally filtered by product type.
        
        Professional best practice:
        - When no query: return all brands
        - When query provided: filter brands to show only those with matching products
        - Uses simple text matching when models aren't ready (fast, works immediately)
        - Uses semantic similarity when models are ready (more accurate)
        """
        if self.df.empty:
            logger.warning("Dataset is empty, returning empty brand list")
            return []
        
        # If no product query, return all brands
        if not product_query or product_query.strip() == "":
            brands = self.df["brand"].dropna().unique().tolist()
            return sorted([b for b in brands if b and str(b).strip()])
        
        # If vectorizer not ready, use simple text-based filtering (fast, works immediately)
        # This is a professional fallback that provides immediate value
        if self.vectorizer is None:
            logger.debug("Using simple text-based brand filtering (models loading)")
            query_lower = product_query.strip().lower()
            
            # Filter products by simple text matching in product names
            if 'product_name' in self.df.columns:
                # Use case-insensitive contains check (simple and reliable)
                # This works better than regex for user queries
                product_names_lower = self.df['product_name'].astype(str).str.lower()
                
                # Check if query appears anywhere in product name
                mask = product_names_lower.str.contains(query_lower, case=False, na=False, regex=False)
                
                # Also check individual words if query has multiple words
                if ' ' in query_lower:
                    query_words = [w.strip() for w in query_lower.split() if w.strip()]
                    if query_words:
                        # At least one word must match
                        word_mask = pd.Series([False] * len(self.df), index=self.df.index)
                        for word in query_words:
                            word_mask = word_mask | product_names_lower.str.contains(word, case=False, na=False, regex=False)
                        mask = mask | word_mask
                
                filtered_products = self.df[mask]
                if not filtered_products.empty:
                    brands = filtered_products["brand"].dropna().unique().tolist()
                    filtered_brands = sorted([b for b in brands if b and str(b).strip()])
                    logger.debug(f"Found {len(filtered_brands)} brands using simple text matching for: {product_query}")
                    return filtered_brands
            
            # Fallback: return all brands if no matches
            logger.debug("No matches found with simple text filtering, returning all brands")
            brands = self.df["brand"].dropna().unique().tolist()
            return sorted([b for b in brands if b and str(b).strip()])
        
        # Vectorizer is ready - use semantic similarity (more accurate)
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
            logger.debug(f"Found {len(filtered_brands)} brands using semantic similarity for: {product_query}")
            return filtered_brands
        except Exception as e:
            logger.warning(f"Error filtering brands with semantic similarity, falling back to text matching: {e}")
            # Fallback to simple text matching
            query_lower = product_query.strip().lower()
            if 'product_name' in self.df.columns:
                product_names_lower = self.df['product_name'].astype(str).str.lower()
                mask = product_names_lower.str.contains(query_lower, case=False, na=False, regex=False)
                
                # Also check individual words
                if ' ' in query_lower:
                    query_words = [w.strip() for w in query_lower.split() if w.strip()]
                    if query_words:
                        word_mask = pd.Series([False] * len(self.df), index=self.df.index)
                        for word in query_words:
                            word_mask = word_mask | product_names_lower.str.contains(word, case=False, na=False, regex=False)
                        mask = mask | word_mask
                
                filtered_products = self.df[mask]
                if not filtered_products.empty:
                    brands = filtered_products["brand"].dropna().unique().tolist()
                    return sorted([b for b in brands if b and str(b).strip()])
            # Final fallback: all brands
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
        similarity_scores: Optional[np.ndarray] = None,
        brand_filter: Optional[str] = None
    ) -> List[Dict[str, Union[str, float]]]:
        """Process search results and extract product data with filtering."""
        output = []
        brand_filter_lower = brand_filter.lower().strip() if brand_filter else None
        
        for _, row in results.iterrows():
            if len(output) >= top_k:
                break
            
            similarity = similarity_scores[row.name] if similarity_scores is not None else None
            product_data = self._extract_product_data(row, similarity)
            product_key = (product_data["name"].lower().strip(), product_data["brand"].lower().strip())
            
            # Enforce brand filter strictly
            if brand_filter_lower:
                product_brand = str(product_data.get("brand", "")).lower().strip()
                if product_brand != brand_filter_lower:
                    continue  # Skip products that don't match the brand filter
            
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
        
        # Check if models are ready - fallback to TF-IDF if GenAI not ready
        use_genai_for_search = self.use_genai and self.genai_embeddings is not None and self.genai_model is not None
        
        if not use_genai_for_search:
            # Fallback to BM25 or TF-IDF if GenAI not ready or not enabled
            if self.bm25 is None and self.vectorizer is None:
                # Try to initialize BM25 and TF-IDF if not already done
                if not self.df.empty:
                    logger.info("GenAI not ready, initializing BM25 and TF-IDF for search...")
                    self._preprocess_text_data()
                    self._initialize_bm25()
                    self._initialize_vectorizer()
                else:
                    logger.error("Search algorithms not available and dataset is empty")
                    return ["Dataset not loaded properly."]
            
            if self.bm25 is None and self.vectorizer is None:
                logger.error("Search algorithms not available")
                return ["Search not ready. Please wait for models to load."]

        # Filter by brand FIRST if brand_query is provided (AND logic)
        # This ensures products must match BOTH product_query AND brand_query
        filtered_df = self.df.copy()
        brand_filter_mask = None
        
        if brand_query and brand_query.strip():
            brand_query_clean = brand_query.strip().lower()
            # Create case-insensitive brand filter
            brand_filter_mask = filtered_df['brand'].astype(str).str.lower().str.strip() == brand_query_clean
            filtered_df = filtered_df[brand_filter_mask].copy()
            
            if filtered_df.empty:
                logger.info(f"No products found for brand '{brand_query}'")
                return [f"No products found for brand '{brand_query}'."]
            
            logger.info(f"Filtered to {len(filtered_df)} products from brand '{brand_query}'")
        
        # Preprocess and expand product query (without brand_query)
        if self.use_advanced_preprocessing:
            expanded_query = self._expand_query(product_query)
            query_text = TextPreprocessor.preprocess_text(expanded_query.strip())
        else:
            query_text = product_query.strip()
        
        logger.info(f"Searching for: '{query_text}' with brand filter: '{brand_query if brand_query else 'None'}' (budget: {budget_max}, top_k: {top_k})")
        
        # Modern Hybrid Search: Combine BM25 (keyword) + Semantic (understanding)
        # This is the industry standard used by major search engines
        if use_genai_for_search and self.use_hybrid_search and self.bm25 is not None:
            # HYBRID SEARCH: BM25 + Semantic Embeddings
            logger.info("Using Hybrid Search (BM25 + Semantic Embeddings)")
            
            # 1. BM25 scores for keyword matching
            bm25_scores = self.bm25.get_scores(query_text)
            # Normalize BM25 scores to 0-1 range
            if bm25_scores.max() > 0:
                bm25_scores_normalized = bm25_scores / bm25_scores.max()
            else:
                bm25_scores_normalized = bm25_scores
            
            # 2. Semantic similarity scores
            query_embedding = self.genai_model.encode([query_text], convert_to_numpy=True)
            semantic_scores = cosine_similarity(query_embedding, self.genai_embeddings).flatten()
            # Normalize semantic scores to 0-1 range
            if semantic_scores.max() > 0:
                semantic_scores_normalized = semantic_scores / semantic_scores.max()
            else:
                semantic_scores_normalized = semantic_scores
            
            # 3. Combine both scores with weighted average
            all_similarity_scores = (
                self.hybrid_weight_bm25 * bm25_scores_normalized +
                self.hybrid_weight_semantic * semantic_scores_normalized
            )
            logger.info(f"Hybrid search: BM25 weight={self.hybrid_weight_bm25}, Semantic weight={self.hybrid_weight_semantic}")
        elif use_genai_for_search:
            # Semantic search only (if BM25 not available)
            logger.info("Using Semantic Embeddings (GenAI)")
            query_embedding = self.genai_model.encode([query_text], convert_to_numpy=True)
            all_similarity_scores = cosine_similarity(query_embedding, self.genai_embeddings).flatten()
        elif self.bm25 is not None:
            # BM25 search only (modern keyword search)
            logger.info("Using BM25 algorithm (modern keyword search)")
            bm25_scores = self.bm25.get_scores(query_text)
            # Normalize BM25 scores
            if bm25_scores.max() > 0:
                all_similarity_scores = bm25_scores / bm25_scores.max()
            else:
                all_similarity_scores = bm25_scores
        else:
            # Fallback to TF-IDF
            logger.info("Using TF-IDF (fallback)")
            query_vec = self.vectorizer.transform([query_text])
            all_similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Apply weighted similarity if advanced features enabled
        if self.use_ngrams or self.use_advanced_preprocessing or self.use_genai:
            all_similarity_scores = self._calculate_weighted_similarity(all_similarity_scores)
        
        # If brand filter is applied, only consider similarity scores for filtered products
        if brand_filter_mask is not None:
            # Get original indices of filtered products
            filtered_indices = filtered_df.index.values
            # Create a mask for similarity scores (set non-filtered products to -inf)
            similarity_scores = np.full(len(self.df), -np.inf)
            similarity_scores[filtered_indices] = all_similarity_scores[filtered_indices]
        else:
            similarity_scores = all_similarity_scores
        
        max_candidates = min(len(filtered_df) if brand_filter_mask is not None else len(self.df), top_k * CANDIDATE_MULTIPLIER)
        top_indices = similarity_scores.argsort()[-max_candidates:][::-1]
        
        # Filter out indices with -inf scores (products that don't match brand filter)
        if brand_filter_mask is not None:
            top_indices = top_indices[similarity_scores[top_indices] != -np.inf]
            # Double-check: ensure all indices are in the filtered set
            filtered_indices_set = set(filtered_indices)
            top_indices = np.array([idx for idx in top_indices if idx in filtered_indices_set])
        
        # Apply diversity filter if enabled (but only within the brand filter if applied)
        if diversity_weight > 0 and (self.use_ngrams or self.use_advanced_preprocessing or self.use_genai):
            top_indices = self._apply_diversity_filter(top_indices, similarity_scores, diversity_weight, top_k)
            # Re-verify brand filter after diversity filter
            if brand_filter_mask is not None:
                filtered_indices_set = set(filtered_indices)
                top_indices = np.array([idx for idx in top_indices if idx in filtered_indices_set])

        results = self.df.iloc[top_indices]
        if results.empty:
            logger.info(f"No relevant products found for '{product_query}'")
            return [f"No relevant products found for '{product_query}'."]

        seen_products = set()
        # Pass brand_query to enforce brand filtering in results processing
        output = self._process_search_results(
            results, seen_products, budget_max, top_k, similarity_scores, brand_query
        )
        
        # If we don't have enough results, try to get more (only if brand filter allows)
        if len(output) < top_k and max_candidates < len(self.df):
            try:
                expanded_candidates = min(len(self.df), top_k * EXPANDED_CANDIDATE_MULTIPLIER)
                all_indices = similarity_scores.argsort()[-expanded_candidates:][::-1]
                processed_indices_set = set(top_indices)
                new_indices = [idx for idx in all_indices if idx not in processed_indices_set]
                
                # If brand filter is applied, only consider indices from filtered products
                if brand_filter_mask is not None:
                    filtered_indices_set = set(filtered_indices)
                    new_indices = [idx for idx in new_indices if idx in filtered_indices_set]
                
                if new_indices:
                    additional_results = self.df.iloc[new_indices]
                    additional_output = self._process_search_results(
                        additional_results, seen_products, budget_max, top_k - len(output), 
                        similarity_scores, brand_query
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
