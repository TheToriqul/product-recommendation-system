# AI-Based Project Recommendation System for Electrical Appliance

## 👥 Project Team Members

* Jason Goh Lik Jhien
* Md Toriqul Islam
* Nada Ahmed Abdulwahab Shalaby
* Phuah Jun Hao
* Tan Kai Yang

---

## 🧠 Project Overview

This repository contains the implementation of an **AI-driven Product Recommendation System** specifically designed for **electrical appliances**. The system leverages **Content-Based Filtering** techniques enhanced by **Artificial Intelligence (AI)** to provide personalized recommendations based on product features and user preferences.

The goal of this project is to demonstrate how recommendation algorithms can enhance customer experience in the e-commerce domain, particularly within the electrical appliance industry.

---

## ⚙️ Tech Stack

* **Programming Language:** Python 3.10+
* **Libraries/Frameworks:**

  * `pandas`, `numpy` – data manipulation
  * `scikit-learn` – similarity computation, model building
  * `nltk`, `spacy` – text preprocessing and feature extraction
  * `flask` or `streamlit` – optional web interface for deployment

---

## 📂 Project Structure

```
📦 ai-electrical-appliance-recommendation-system
 ┣ 📂 data               # Datasets (product info, metadata, features)
 ┣ 📂 notebooks          # Jupyter notebooks for exploration & model testing
 ┣ 📂 src                # Source code for the recommender engine
 ┣ 📂 models             # Trained models or serialized pipelines
 ┣ 📂 static             # Assets for front-end (if applicable)
 ┣ 📂 templates          # Front-end HTML templates (if Flask app)
 ┣ 📜 requirements.txt   # Python dependencies
 ┣ 📜 README.md          # Project documentation
 ┗ 📜 app.py             # Entry point for demo/deployment
```

---

## 🧩 Key Features

* Intelligent **content-based filtering** using product descriptions and features.
* **TF-IDF vectorization** and **cosine similarity** for recommendation ranking.
* Support for **scalable datasets** (e.g., thousands of product listings).
* Easy integration with a **web or mobile front-end**.

---

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/<your-username>/ai-electrical-appliance-recommendation-system.git
cd ai-electrical-appliance-recommendation-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then open your browser and navigate to `http://127.0.0.1:5000/`.

---

## 📊 Dataset

A custom dataset of **electrical appliances** (e.g., refrigerators, air conditioners, washing machines) was created using publicly available product data. Each record includes:

* Product name
* Brand
* Category
* Specifications
* Description
* Price range

Data preprocessing includes text cleaning, feature extraction, and vectorization for similarity computation.

---

## 🧠 Model Development Workflow

1. **Data Collection & Cleaning:** Gathered product data from e-commerce sources.
2. **Feature Engineering:** Extracted textual and numerical attributes.
3. **Model Building:** Implemented a TF-IDF and cosine similarity-based recommender.
4. **Evaluation:** Compared recommendation relevance via precision metrics.
5. **Deployment:** Optional deployment via Flask web interface or API.

---

## 📈 Performance Metrics

* **Precision@K** – evaluates relevance of top K recommendations.
* **Recall@K** – measures the ability to find all relevant items.
* **NDCG (Normalized Discounted Cumulative Gain)** – ranks relevance in recommendation lists.

---

## 🌍 Sustainable Development Goal (SDG) Alignment

This project aligns with:

* **SDG 9: Industry, Innovation, and Infrastructure** – promoting AI innovation in consumer electronics.
* **SDG 12: Responsible Consumption and Production** – encouraging efficient product discovery and informed purchases.

---

## 🧭 Future Enhancements

* Incorporate **hybrid recommendation** with collaborative filtering.
* Integrate **deep learning embeddings** for better semantic understanding.
* Add **real-time recommendations** using user interaction data.
* Develop an interactive dashboard for analytics visualization.

---

## ⚖️ Ethical Considerations

* Ensure fairness in recommendations by avoiding brand or price bias.
* Maintain transparency in how recommendations are generated.
* Respect user data privacy and comply with data protection regulations.

---

## 📚 References

1. Aggarwal, C. C. (2016). *Recommender Systems: The Textbook.* Springer.
2. Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook.* Springer.
3. Sharma, M., & Pathak, D. (2021). *Content-Based Recommendation System Using TF-IDF and Cosine Similarity.* IJERT.
4. Amazon Product Dataset (Kaggle): [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)
5. Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
