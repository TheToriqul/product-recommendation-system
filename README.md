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
- 🎯 **Intelligent Recommendations**: Uses TF-IDF vectorization and cosine similarity for accurate product matching
- 🧠 **Semantic Understanding**: LLM-powered embeddings capture meaning and context beyond keyword matching
- 🖥️ **User-Friendly GUI**: Modern desktop application built with Tkinter
- 🔍 **Advanced Filtering**: Filter by product type, brand, budget, and sorting preferences
- 🔗 **Product Links**: Direct access to product URLs for easy browsing
- 📊 **Similar Products**: "You May Also Like" feature for discovering related items

---

## ⚙️ Tech Stack

- **Programming Language:** Python 3.10+
- **Core Libraries:**
  - `pandas` – data manipulation and analysis
  - `scikit-learn` – TF-IDF vectorization and cosine similarity computation
  - `numpy` – numerical operations
  - `sentence-transformers` – **Generative AI (LLM)** for semantic embeddings (BERT-based)
  - `torch` – PyTorch backend for sentence transformers
  - `nltk` – advanced text preprocessing (optional)
- **GUI Framework:** `tkinter` (built-in Python library)
- **Data Source:** CSV dataset containing home appliance SKUs from Lowe's
- **Generative AI Model:** `all-MiniLM-L6-v2` (Sentence-BERT model for semantic understanding)

---

## 📂 Project Structure

```
product-recommendation-system/
├── app_gui.py                    # Main GUI application (entry point)
├── recommender_engine.py         # Core recommendation engine
├── home appliance skus lowes.csv # Product dataset
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── LICENSE                       # MIT License
```

---

## 🧩 Key Features

### Recommendation Engine

- **Generative AI (LLM)**: Uses Sentence Transformers (BERT-based) for semantic embeddings
  - Captures meaning and context beyond keyword matching
  - Understands synonyms, related terms, and product relationships
  - Generates high-dimensional semantic vectors for products and queries
- **Content-Based Filtering**: Analyzes product names and brands using TF-IDF vectorization
- **Hybrid Approach**: Can use both GenAI embeddings and TF-IDF (GenAI takes precedence when enabled)
- **Cosine Similarity**: Computes similarity scores between products for accurate recommendations
- **Dynamic Brand Filtering**: Automatically filters available brands based on product type
- **Budget Constraints**: Filter recommendations by price ranges (Under $100, $300, $500, $1000, $2000)
- **Multiple Sorting Options**: Sort by similarity, price (low-to-high/high-to-low), or rating

### User Interface

- **Modern Dark Theme**: Clean and professional desktop interface
- **Interactive Tables**: Display recommendations with product details (name, brand, price, rating)
- **Similar Products Section**: Shows related products when clicking on a recommendation
- **Product URL Integration**: Double-click to open product links in browser
- **Real-time Search**: Instant brand filtering based on product query

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

The GUI application will launch automatically. No browser access needed - it's a desktop application!

---

## 📖 Usage Guide

### Basic Search

1. **Enter Product Type**: Type the name of the appliance you're looking for (e.g., "refrigerator", "washing machine", "air conditioner")
2. **Select Brand** (Optional): Choose a specific brand from the dropdown, or leave as "All Brands"
3. **Set Budget** (Optional): Select a maximum price range, or choose "No Limit"
4. **Choose Sort Option**: Select how you want results sorted:
   - Similarity (Default) - Most relevant first
   - Price: Low to High - Cheapest first
   - Price: High to Low - Most expensive first
   - Rating: Best First - Highest rated first
5. **Click Search**: View recommendations in the main table

### Advanced Features

- **View Product Details**: Double-click any product row to open its URL in your browser
- **Find Similar Products**: Single-click a product to see similar items in the "You May Also Like" section
- **Dynamic Brand Filtering**: The brand dropdown automatically updates based on your product query

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
2. **Text Vectorization**: Product names and brands are converted to TF-IDF vectors
3. **Query Processing**: User input is transformed into a query vector
4. **Similarity Computation**: Cosine similarity is calculated between query and all products
5. **Filtering & Sorting**: Results are filtered by brand/budget and sorted according to user preference
6. **Ranking**: Top K most similar products are returned as recommendations

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
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Weights terms based on their importance (fallback when GenAI not available)
- **Cosine Similarity**: Measures the angle between vectors (0 = identical, 1 = completely different)
- **Content-Based Filtering**: Recommends items similar to what the user is looking for, not based on other users' behavior
- **Hybrid Architecture**: Seamlessly switches between GenAI embeddings and TF-IDF based on availability

---

## 📈 Performance & Scalability

- **Efficient Processing**: Handles datasets with thousands of products
- **Fast Search**: Real-time recommendations with minimal latency
- **Memory Optimized**: Uses sparse matrices for efficient memory usage
- **Scalable Architecture**: Can be extended to handle larger datasets

---

## 🧭 Future Enhancements

- **Hybrid Recommendation**: Incorporate collaborative filtering for better accuracy
- **Deep Learning**: Integrate neural embeddings (Word2Vec, BERT) for semantic understanding
- **Real-time Learning**: Update recommendations based on user interaction data
- **Web Interface**: Deploy as a web application using Flask or Streamlit
- **API Development**: Create RESTful API for integration with other systems
- **Analytics Dashboard**: Visualize recommendation performance and user behavior
- **Multi-language Support**: Extend to support multiple languages

---

## 🧪 Testing

To test the recommendation system:

```bash
# Run the GUI application
python app_gui.py

# Test with different product queries:
# - "refrigerator"
# - "washing machine"
# - "air conditioner"
# - "microwave"
```

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

1. Aggarwal, C. C. (2016). _Recommender Systems: The Textbook._ Springer.
2. Ricci, F., Rokach, L., & Shapira, B. (2015). _Recommender Systems Handbook._ Springer.
3. Sharma, M., & Pathak, D. (2021). _Content-Based Recommendation System Using TF-IDF and Cosine Similarity._ IJERT.
4. [Scikit-learn Documentation](https://scikit-learn.org/)
5. [Pandas Documentation](https://pandas.pydata.org/)

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
