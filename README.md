# AI-Based Product Recommendation System for Electrical Appliances

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/TheToriqul/product-recommendation-system)

A content-based recommendation system that suggests electrical appliances using Artificial Intelligence (AI) and data-driven insights. This project leverages machine learning techniques to analyze appliance features and recommend similar or relevant products based on user preferences.

## 👥 Project Team Members

- Jason Goh Lik Jhien
- Md Toriqul Islam
- Nada Ahmed Abdulwahab Shalaby
- Phuah Jun Hao
- Tan Kai Yang

---

## 🧠 Project Overview

This repository contains the implementation of an **AI-driven Product Recommendation System** specifically designed for **electrical appliances**. The system leverages **Content-Based Filtering** techniques enhanced by **Artificial Intelligence (AI)** and **Generative AI (LLM)** to provide personalized recommendations based on product features and user preferences.

**Key Innovation**: This project implements **Generative AI** using **Sentence Transformers (BERT-based models)** for semantic understanding, meeting the requirement for Generative AI application in the e-commerce recommendation domain.

The goal of this project is to demonstrate how recommendation algorithms enhanced with Generative AI can improve customer experience in the e-commerce domain, particularly within the electrical appliance industry.

### Key Highlights

- 🤖 **Generative AI Integration**: Uses Sentence Transformers (BERT-based) for semantic understanding and embeddings
- 💬 **AI Chatbot Assistant**: Interactive chatbot powered by GPT-2 LLM for natural language product queries
- 🎯 **Intelligent Recommendations**: Uses TF-IDF vectorization and cosine similarity for accurate product matching
- 🧠 **Semantic Understanding**: LLM-powered embeddings capture meaning and context beyond keyword matching
- 🖥️ **User-Friendly GUI**: Modern desktop application with tabbed interface (Search & Chat)
- 🔍 **Advanced Filtering**: Filter by product type, brand, budget, and sorting preferences
- 🔗 **Product Links**: Direct access to product URLs for easy browsing
- 📊 **Similar Products**: "You May Also Like" feature for discovering related items
- 📚 **Knowledge Base**: Auto-generated product knowledge base for enhanced chatbot responses
- 🎨 **Modern UI**: Dark-themed interface with ChatGPT-style chat interface

---

## ⚙️ Tech Stack

- **Programming Language:** Python 3.10+
- **Core Libraries:**
  - `pandas` – data manipulation and analysis
  - `scikit-learn` – TF-IDF vectorization and cosine similarity computation
  - `numpy` – numerical operations
  - `sentence-transformers` – **Generative AI (LLM)** for semantic embeddings (BERT-based)
  - `transformers` – **LLM Framework** for chatbot text generation (GPT-2)
  - `torch` – PyTorch backend for sentence transformers and LLM models
  - `nltk` – advanced text preprocessing (optional)
  - `Pillow` – image processing for logo display
- **GUI Framework:** `tkinter` (built-in Python library)
- **Data Source:** CSV dataset containing home appliance SKUs from Lowe's
- **Generative AI Models:**
  - `all-MiniLM-L6-v2` – Sentence-BERT model for semantic understanding (~90MB)
  - `gpt2` – Text generation model for chatbot (~500MB)

---

## 📂 Project Structure

```
product-recommendation-system/
├── app_gui.py                    # Main GUI application (entry point)
├── recommender_engine.py         # Core recommendation engine with GenAI
├── chatbot.py                    # AI chatbot with LLM support
├── chatbot_ui.py                 # Chatbot UI components (ChatGPT-style)
├── chatbot_trainer.py            # Knowledge base generator
├── config.py                     # Configuration management
├── ui_components.py              # UI component creation functions
├── ui_handlers.py                # UI event handlers and business logic
├── ui_constants.py               # UI constants (colors, fonts, etc.)
├── ui_styles.py                  # UI styling functions
├── test_recommender.py           # Unit tests
├── home appliance skus lowes.csv  # Product dataset
├── requirements.txt              # Python dependencies
├── TRAINING_GUIDE.md             # Chatbot training documentation
├── README.md                     # Project documentation
├── LICENSE                       # MIT License
├── models/                       # Cached AI models (auto-downloaded)
│   ├── all-MiniLM-L6-v2/        # Sentence Transformer model
│   └── gpt2/                     # GPT-2 chatbot model
├── training_data/                # Auto-generated training data
│   ├── knowledge_base.json       # Product knowledge base
│   └── training_prompts.json    # Chatbot training prompts
└── assets/                       # Application assets
    └── inti logo.png            # University logo
```

---

## 🧩 Key Features

### Recommendation Engine

- **Generative AI (LLM)**: Uses Sentence Transformers (BERT-based) for semantic embeddings
  - Captures meaning and context beyond keyword matching
  - Understands synonyms, related terms, and product relationships
  - Generates high-dimensional semantic vectors for products and queries
  - Model caching: Downloads and caches models locally for offline use
- **Content-Based Filtering**: Analyzes product names and brands using TF-IDF vectorization
- **Hybrid Approach**: Can use both GenAI embeddings and TF-IDF (GenAI takes precedence when enabled)
- **Cosine Similarity**: Computes similarity scores between products for accurate recommendations
- **Dynamic Brand Filtering**: Automatically filters available brands based on product type
- **Budget Constraints**: Filter recommendations by price ranges (Under $100, $300, $500, $1000, $2000)
- **Multiple Sorting Options**: Sort by similarity, price (low-to-high/high-to-low), or rating
- **Similar Products Discovery**: Find related products based on selected items

### AI Chatbot Assistant

- **Natural Language Interface**: Chat with the AI assistant using natural language
- **Conversational & Friendly**: Natural, casual responses that feel like talking to a friend
- **LLM-Powered**: Uses GPT-2 model for conversational responses with optimized parameters
- **Product Recommendations**: Chatbot can recommend products directly in conversation
- **Out-of-Scope Detection**: Intelligently detects and handles queries about non-appliance products
- **Knowledge Base Integration**: Auto-generated knowledge base from product dataset
- **Smart Suggestions**: Dynamic quick suggestions based on user interests
- **Context Awareness**: Understands product queries, brand preferences, and budget constraints
- **ChatGPT-Style UI**: Modern chat interface with message bubbles and timestamps
- **Auto-Training**: Automatically generates training data from CSV on first run

### User Interface

- **Tabbed Interface**: Two tabs - "Search Products" and "AI Assistant"
- **Modern Dark Theme**: Clean and professional desktop interface
- **Interactive Tables**: Display recommendations with product details (name, brand, price, rating, similarity)
- **Similar Products Section**: Shows related products when clicking on a recommendation
- **Product URL Integration**: Double-click to open product links in browser
- **Real-time Search**: Instant brand filtering based on product query
- **Export Functionality**: Export search results to CSV or JSON format
- **Responsive Design**: Adapts to window resizing with proper scrolling

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/TheToriqul/product-recommendation-system.git
cd product-recommendation-system
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
# On Windows:
python -m venv venv
# On macOS/Linux:
python3 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# On Windows:
pip install -r requirements.txt
# On macOS/Linux:
pip3 install -r requirements.txt
```

### Step 4: Run the Application

```bash
# On Windows:
python app_gui.py
# On macOS/Linux:
python3 app_gui.py
```

**Note:** On macOS and Linux, use `python3` and `pip3`. On Windows, `python` and `pip` should work. If `python` doesn't work on Windows, try `py` or `python3`.

### First Run Setup

On first run, the application will:

1. **Download AI Models** (if not cached):
   - Sentence Transformer model (~90MB) - for semantic search
   - GPT-2 model (~500MB) - for chatbot (optional, can be skipped)
   - Models are cached locally in `models/` directory for future use
2. **Generate Training Data** (if not present):
   - Knowledge base from CSV dataset (~10-30 seconds)
   - Training prompts for chatbot
   - Saved in `training_data/` directory

**Note:** The first run may take a few minutes to download models. Subsequent runs are instant as models are cached locally.

The GUI application will launch automatically. No browser access needed - it's a desktop application!

---

## 📖 Usage Guide

### Search Products Tab

1. **Enter Product Type**: Type the name of the appliance you're looking for (e.g., "refrigerator", "washing machine", "air conditioner")
2. **Select Brand** (Optional): Choose a specific brand from the dropdown, or leave as "All Brands"
   - Brand list automatically filters based on your product query
3. **Set Budget** (Optional): Select a maximum price range, or choose "No Limit"
   - Options: Under $100, $300, $500, $1000, $2000
4. **Choose Sort Option**: Select how you want results sorted:
   - Similarity (Default) - Most relevant first
   - Price: Low to High - Cheapest first
   - Price: High to Low - Most expensive first
   - Rating: Best First - Highest rated first
5. **Click Search**: View recommendations in the main table

### AI Assistant Tab

1. **Switch to AI Assistant Tab**: Click on the "💬 AI Assistant" tab
2. **Ask Questions**: Type your question in natural language, for example:
   - "Find me a refrigerator"
   - "Show me washing machines under $500"
   - "What brands are available?"
   - "Help me find products"
3. **Use Quick Suggestions**: Click on quick suggestion buttons for common queries
4. **Get Recommendations**: The chatbot can recommend products directly in the conversation
5. **Clear Chat**: Use "Clear Chat" button to start a new conversation

### Advanced Features

- **View Product Details**: Double-click any product row to open its URL in your browser
- **Find Similar Products**: Single-click a product to see similar items in the "You May Also Like" section
- **Dynamic Brand Filtering**: The brand dropdown automatically updates based on your product query
- **Export Results**: Export search results to CSV or JSON format (when available)
- **Smart Suggestions**: Chatbot suggestions adapt based on your search history

---

## 📊 Dataset

The system uses a custom dataset of **electrical appliances** sourced from Lowe's product catalog. Each record includes:

- **Product Name**: Full product title
- **Brand**: Manufacturer name
- **Price Current**: Current selling price
- **Price Retail**: Original retail price
- **Bestseller Rank**: Popularity ranking
- **Product URL**: Link to product page

The dataset is preprocessed automatically:

- Text cleaning and normalization
- Feature extraction from product names and brands
- TF-IDF vectorization for similarity computation

---

## 🧠 How It Works

### Recommendation Algorithm

1. **Data Loading**: The system loads product data from the CSV file
2. **Text Preprocessing**: Product names and brands are cleaned and normalized
3. **Vectorization**:
   - **GenAI Mode**: Products are converted to semantic embeddings using Sentence Transformers
   - **TF-IDF Mode**: Products are converted to TF-IDF vectors (fallback)
4. **Query Processing**: User input is transformed into a query vector/embedding
5. **Similarity Computation**: Cosine similarity is calculated between query and all products
6. **Filtering & Sorting**: Results are filtered by brand/budget and sorted according to user preference
7. **Ranking**: Top K most similar products are returned as recommendations

### Chatbot System

1. **Knowledge Base Generation**: Product information is extracted from CSV and organized
2. **Training Data Creation**: Training prompts are generated automatically
3. **Query Understanding**: User messages are analyzed for product queries, brands, and budgets
4. **Out-of-Scope Detection**: Detects queries about non-appliance products (phones, laptops, TVs, etc.) and provides helpful guidance
5. **Response Generation**:
   - **LLM Mode**: GPT-2 generates natural, conversational responses with optimized temperature and sampling
   - **Rule-Based Mode**: Friendly fallback responses for common patterns
   - **Natural Language**: Casual, friendly tone that feels like talking to a friend
6. **Product Integration**: Chatbot can call recommendation engine to provide product suggestions

### Technical Details

- **Generative AI (Sentence Transformers)**:

  - **Model**: `all-MiniLM-L6-v2` (optimized BERT-based transformer)
  - **Why This Model**: Best balance of size, speed, and accuracy for semantic similarity
  - **Free and Open-Source**: No API costs, runs completely locally
  - **Performance**:
    - Generates 384-dimensional semantic embeddings
    - Understands context, synonyms, and semantic relationships
    - Processes queries and products into dense vector representations
    - Fast inference: ~10-50ms per query
  - **Specifications**:
    - Model Size: ~90MB (small and efficient)
    - License: Apache 2.0 (free for commercial use)
    - Architecture: BERT-based transformer with 6 layers
    - Training: Fine-tuned on 1B+ sentence pairs
    - **Caching**: Models are cached locally in `models/` directory

- **Chatbot LLM (GPT-2)**:

  - **Model**: `gpt2` (small, fast text generation model)
  - **Purpose**: Natural language understanding and response generation
  - **Model Size**: ~500MB
  - **Features**:
    - Natural, conversational responses with casual, friendly tone
    - Product query understanding
    - Out-of-scope query detection and handling
    - Integration with recommendation engine
    - Knowledge base context
    - Optimized parameters: temperature=0.8, top_p=0.9 for natural variation
    - Multi-sentence responses for better conversation flow
    - **Completely Free**: Runs locally, no API costs, no usage limits
    - **Note**: Can be upgraded to better local models (Ollama, DialoGPT, Phi-2) for improved quality

- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Weights terms based on their importance (fallback when GenAI not available)
- **Cosine Similarity**: Measures the angle between vectors (0 = identical, 1 = completely different)
- **Content-Based Filtering**: Recommends items similar to what the user is looking for, not based on other users' behavior
- **Hybrid Architecture**: Seamlessly switches between GenAI embeddings and TF-IDF based on availability
- **Knowledge Base**: Auto-generated from CSV, contains product categories, brands, and price ranges

---

## 📈 Performance & Scalability

- **Efficient Processing**: Handles datasets with thousands of products
- **Fast Search**: Real-time recommendations with minimal latency
- **Memory Optimized**: Uses sparse matrices for efficient memory usage
- **Scalable Architecture**: Can be extended to handle larger datasets

---

## 🧭 Future Enhancements

- **Hybrid Recommendation**: Incorporate collaborative filtering for better accuracy
- **Real-time Learning**: Update recommendations based on user interaction data
- **Web Interface**: Deploy as a web application using Flask or Streamlit
- **API Development**: Create RESTful API for integration with other systems
- **Analytics Dashboard**: Visualize recommendation performance and user behavior
- **Multi-language Support**: Extend to support multiple languages
- **Advanced Chatbot**: Fine-tune chatbot on product-specific data for better responses
- **User Profiles**: Save user preferences and search history
- **Recommendation Explanations**: Show why products were recommended

---

## 🧪 Testing

### Running the Application

```bash
# Run the GUI application
python app_gui.py
# or
python3 app_gui.py
```

### Running Unit Tests

```bash
# Run unit tests
python -m pytest test_recommender.py
# or
python test_recommender.py
```

### Testing Recommendations

Test with different product queries:

- "refrigerator"
- "washing machine"
- "air conditioner"
- "microwave"
- "oven"
- "dishwasher"

### Testing Chatbot

Try these chatbot queries:

- "Find me a refrigerator"
- "Show me washing machines under $500"
- "What brands are available?"
- "Help me find products"
- "I need a Samsung air conditioner"

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## ⚖️ Ethical Considerations

- **Fairness**: Recommendations avoid brand or price bias
- **Transparency**: Clear explanation of how recommendations are generated
- **Privacy**: No user data is stored or tracked
- **Data Protection**: Complies with data protection regulations

---

## 🌍 Sustainable Development Goal (SDG) Alignment

This project aligns with:

- **SDG 9: Industry, Innovation, and Infrastructure** – promoting AI innovation in consumer electronics
- **SDG 12: Responsible Consumption and Production** – encouraging efficient product discovery and informed purchases

---

## 📚 References & Resources

### Academic References

1. Aggarwal, C. C. (2016). _Recommender Systems: The Textbook._ Springer.
2. Ricci, F., Rokach, L., & Shapira, B. (2015). _Recommender Systems Handbook._ Springer.
3. Sharma, M., & Pathak, D. (2021). _Content-Based Recommendation System Using TF-IDF and Cosine Similarity._ IJERT.

### Technical Documentation

4. [Scikit-learn Documentation](https://scikit-learn.org/)
5. [Pandas Documentation](https://pandas.pydata.org/)
6. [Sentence Transformers Documentation](https://www.sbert.net/)
7. [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
8. [PyTorch Documentation](https://pytorch.org/docs/)

### Model Information

- **Sentence-BERT**: [all-MiniLM-L6-v2 on Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **GPT-2**: [GPT-2 on Hugging Face](https://huggingface.co/gpt2)

### Additional Resources

- See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for information about chatbot training data generation

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact & Support

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/TheToriqul/product-recommendation-system) or open an issue.

---

## 🙏 Acknowledgments

- Dataset sourced from Lowe's product catalog
- Built as part of INT4203E course project
- Thanks to all team members for their contributions

---

**Made with ❤️ by the Product Recommendation System Team**
