import pandas as pd
import gzip
import json
from sentence_transformers import SentenceTransformer, util
import torch
from collections import Counter
import re

# ---------------------------------------------
# Helper: Load JSON.GZ with a record limit
# ---------------------------------------------
def load_amazon_data(file_path, max_records=20000):
    data = []
    with gzip.open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            try:
                data.append(json.loads(line))
            except:
                continue
    return pd.DataFrame(data)

# ---------------------------------------------
# Load datasets
# ---------------------------------------------
print("📥 Loading Amazon Home & Kitchen data...")
reviews_path = "Home_and_Kitchen.json.gz"
meta_path = "meta_Home_and_Kitchen.json.gz"

reviews = load_amazon_data(reviews_path, 20000)
metadata = load_amazon_data(meta_path, 20000)

print(f"✅ Reviews loaded: {reviews.shape}")
print(f"✅ Metadata loaded: {metadata.shape}\n")

# ---------------------------------------------
# Clean & Select Columns Safely
# ---------------------------------------------
review_cols = [col for col in ['reviewText', 'overall', 'asin'] if col in reviews.columns]
meta_cols = [col for col in ['asin', 'title', 'brand', 'categories', 'description'] if col in metadata.columns]

reviews = reviews[review_cols].dropna(subset=['asin'])
metadata = metadata[meta_cols].dropna(subset=['asin'])

for col in ['title', 'brand', 'categories', 'description']:
    if col not in metadata.columns:
        metadata[col] = "Unknown"

# Merge review + metadata
merged_df = pd.merge(reviews, metadata, on='asin')
merged_df.drop_duplicates(subset=['asin'], inplace=True)
merged_df.reset_index(drop=True, inplace=True)
print(f"✅ Merged dataset ready: {merged_df.shape[0]} products\n")

# ---------------------------------------------
# Explore Dataset Content
# ---------------------------------------------
print("🔍 Exploring dataset contents...\n")
print(f"📦 Total unique products: {merged_df['title'].nunique()}")
print(f"🏷️  Total unique brands: {merged_df['brand'].nunique()}\n")

# Show sample product titles
print("🧩 Random sample of 10 product titles:")
print(merged_df['title'].dropna().sample(10, random_state=42).tolist())

# Show top frequent words in titles
titles_text = " ".join(merged_df['title'].dropna().astype(str)).lower()
words = re.findall(r'\b[a-z]{4,}\b', titles_text)
common_words = Counter(words).most_common(20)

print("\n🔥 Top 20 most common words in product titles:")
for word, freq in common_words:
    print(f"{word:<15} {freq}")
print("\n---------------------------------------------\n")

# ---------------------------------------------
# Load Semantic Model
# ---------------------------------------------
print("🧠 Loading semantic model (please wait)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded successfully!\n")

# Precompute embeddings for product titles
print("🔄 Encoding product titles...")
titles = merged_df['title'].astype(str).tolist()
title_embeddings = model.encode(titles, convert_to_tensor=True, show_progress_bar=True)
print("✅ All product titles encoded.\n")

# ---------------------------------------------
# Recommend Products
# ---------------------------------------------
def recommend_products(query, brand_filter=None, top_k=5):
    # Filter by keyword in title, description, or category
    filtered = merged_df[
        merged_df['title'].str.contains(query, case=False, na=False)
    ]
    if 'categories' in merged_df.columns:
        filtered = pd.concat([
            filtered,
            merged_df[merged_df['categories'].astype(str).str.contains(query, case=False, na=False)]
        ]).drop_duplicates(subset=['asin'])
    if 'description' in merged_df.columns:
        filtered = pd.concat([
            filtered,
            merged_df[merged_df['description'].astype(str).str.contains(query, case=False, na=False)]
        ]).drop_duplicates(subset=['asin'])

    # Filter by brand
    if brand_filter:
        filtered = filtered[filtered['brand'].str.contains(brand_filter, case=False, na=False)]

    if filtered.empty:
        print(f"\n❌ No matching products found for '{query}' with brand '{brand_filter or 'any'}'. Please try another search.\n")
        return

    # Encode the query and calculate similarity
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, model.encode(filtered['title'].astype(str).tolist(), convert_to_tensor=True))[0]
    top_indices = torch.topk(similarities, k=min(top_k, len(similarities))).indices

    top_indices = [int(i) for i in top_indices]

    print(f"\n🔍 Top {len(top_indices)} recommendations for: '{query}'")
    print("-" * 100)
    for idx in top_indices:
        product = filtered.iloc[idx]
        brand = product.get('brand', 'Unknown')
        rating = product.get('overall', 'N/A')
        print(f"✅ {product['title']} (Brand: {brand}, Rating: {rating})")
    print("-" * 100 + "\n")

# ---------------------------------------------
# Interactive Loop
# ---------------------------------------------
while True:
    query = input("Enter a product type (e.g. vacuum, blender, toaster) or type 'exit' to quit: ")
    if query.lower() == 'exit':
        print("👋 Exiting recommendation system. Goodbye!")
        break
    brand = input("Enter a brand name (or leave blank to skip): ").strip()
    recommend_products(query, brand)
