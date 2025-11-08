# recommender_engine.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ProductRecommender:
    def __init__(self, csv_path="home appliance skus lowes.csv"):
        try:
            self.df = pd.read_csv(csv_path)
            self.df.columns = [c.strip().lower() for c in self.df.columns]
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            self.df = pd.DataFrame()

        if not self.df.empty:
            text_data = (
                self.df["product_name"].fillna('') + ' ' +
                self.df.get("brand", "").fillna('')
            )
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.tfidf_matrix = self.vectorizer.fit_transform(text_data)
        else:
            self.vectorizer = None
            self.tfidf_matrix = None

    def get_available_brands(self, product_query=""):
        """Get sorted list of unique brands from the dataset, optionally filtered by product type"""
        if self.df.empty:
            return []
        
        if not product_query or product_query.strip() == "":
            # Return all brands if no product query
            brands = self.df["brand"].dropna().unique().tolist()
            return sorted([b for b in brands if b and str(b).strip()])
        
        # Filter brands based on product type using similarity search
        if self.vectorizer is None:
            return []
        
        try:
            query_vec = self.vectorizer.transform([product_query.strip()])
            similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top 100 most similar products to find relevant brands
            top_indices = similarity_scores.argsort()[-100:][::-1]
            relevant_products = self.df.iloc[top_indices]
            
            # Extract unique brands from relevant products
            brands = relevant_products["brand"].dropna().unique().tolist()
            return sorted([b for b in brands if b and str(b).strip()])
        except Exception as e:
            # Fallback to all brands if there's an error
            brands = self.df["brand"].dropna().unique().tolist()
            return sorted([b for b in brands if b and str(b).strip()])
    
    def get_similar_products(self, product_name, brand, top_k=8):
        """Get similar products based on a specific product name and brand"""
        if self.df.empty or self.vectorizer is None:
            return []
        
        # Create query from product name and brand
        query_text = f"{product_name} {brand}".strip()
        query_vec = self.vectorizer.transform([query_text])
        similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top candidates (more than needed to account for duplicates and the current product)
        max_candidates = min(len(self.df), top_k * 5)
        top_indices = similarity_scores.argsort()[-max_candidates:][::-1]
        
        results = self.df.iloc[top_indices]
        if results.empty:
            return []
        
        # Track seen products and exclude the current product
        seen_products = set()
        current_product_key = (product_name.lower().strip(), brand.lower().strip())
        output = []
        
        for _, row in results.iterrows():
            name = row.get("product_name", "Unknown Product")
            brand_val = row.get("brand", "Unknown")
            
            # Convert to string and handle NaN/None values
            name = str(name) if pd.notna(name) else "Unknown Product"
            brand_val = str(brand_val) if pd.notna(brand_val) else "Unknown"
            
            product_key = (name.lower().strip(), brand_val.lower().strip())
            
            # Skip if it's the current product
            if product_key == current_product_key:
                continue
            
            # Skip if we've already seen this product
            if product_key in seen_products:
                continue
            
            # Stop if we have enough products
            if len(output) >= top_k:
                break
            
            # Get price
            price_current = row.get("price_current", "")
            price_retail = row.get("price_retail", "")
            price_value = price_current if price_current else price_retail
            
            try:
                if price_value and pd.notna(price_value):
                    price = f"${float(price_value):.2f}"
                else:
                    price = "N/A"
            except (ValueError, TypeError):
                price = "N/A"
            
            # Get rating
            bestseller_rank = row.get("bestseller_rank", "")
            try:
                if bestseller_rank and pd.notna(bestseller_rank):
                    rating = f"Rank #{int(float(bestseller_rank))}"
                else:
                    rating = "N/A"
            except (ValueError, TypeError):
                rating = "N/A"
            
            # Get URL
            product_url = row.get("product_url", "")
            product_url = str(product_url) if pd.notna(product_url) and product_url else ""
            
            seen_products.add(product_key)
            
            output.append({
                "name": name,
                "brand": brand_val,
                "price": price,
                "rating": rating,
                "url": product_url
            })
        
        return output
    
    def recommend(self, product_query, brand_query="", budget_max=None, top_k=10, sort_by="Similarity (Default)"):
        if self.df.empty or self.vectorizer is None:
            return ["Dataset not loaded properly."]

        query_text = f"{product_query} {brand_query}".strip()
        query_vec = self.vectorizer.transform([query_text])
        similarity_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get more candidates than needed to account for duplicates and budget filtering
        # Start with top_k * 10, but expand if we need more
        max_candidates = min(len(self.df), top_k * 10)
        top_indices = similarity_scores.argsort()[-max_candidates:][::-1]

        results = self.df.iloc[top_indices]
        if results.empty:
            return [f"No relevant products found for '{product_query}'."]

        # Track seen products to avoid duplicates (using product_name + brand as unique key)
        seen_products = set()
        output = []
        
        for _, row in results.iterrows():
            name = row.get("product_name", "Unknown Product")
            brand = row.get("brand", "Unknown")
            
            # Convert to string and handle NaN/None values
            name = str(name) if pd.notna(name) else "Unknown Product"
            brand = str(brand) if pd.notna(brand) else "Unknown"
            
            # Create a unique key for this product
            product_key = (name.lower().strip(), brand.lower().strip())
            
            # Skip if we've already seen this product
            if product_key in seen_products:
                continue
            
            # Stop if we have enough unique products
            if len(output) >= top_k:
                break
            
            # Use price_current, fallback to price_retail if price_current is missing
            price_current = row.get("price_current", "")
            price_retail = row.get("price_retail", "")
            price_value = price_current if price_current else price_retail
            
            # Filter by budget if specified
            if budget_max is not None:
                try:
                    if price_value and pd.notna(price_value):
                        if float(price_value) > budget_max:
                            continue
                except (ValueError, TypeError):
                    pass  # Skip budget check if price is invalid
            
            # Format price with $ if it's a number
            try:
                if price_value and pd.notna(price_value):
                    price = f"${float(price_value):.2f}"
                else:
                    price = "N/A"
            except (ValueError, TypeError):
                price = "N/A"
            
            # Use bestseller_rank as rating (lower rank = more popular/better)
            bestseller_rank = row.get("bestseller_rank", "")
            try:
                if bestseller_rank and pd.notna(bestseller_rank):
                    rating = f"Rank #{int(float(bestseller_rank))}"
                else:
                    rating = "N/A"
            except (ValueError, TypeError):
                rating = "N/A"
            
            # Mark this product as seen
            seen_products.add(product_key)
            
            # Store numeric values for sorting
            price_num = None
            try:
                if price_value and pd.notna(price_value):
                    price_num = float(price_value)
            except (ValueError, TypeError):
                price_num = float('inf')  # Put items without price at the end
            
            rating_num = None
            try:
                if bestseller_rank and pd.notna(bestseller_rank):
                    rating_num = int(float(bestseller_rank))
                else:
                    rating_num = float('inf')  # Put items without rating at the end
            except (ValueError, TypeError):
                rating_num = float('inf')
            
            # Get product URL
            product_url = row.get("product_url", "")
            product_url = str(product_url) if pd.notna(product_url) and product_url else ""
            
            output.append({
                "name": name,
                "brand": brand,
                "price": price,
                "rating": rating,
                "url": product_url,
                "_price_num": price_num,  # For sorting
                "_rating_num": rating_num  # For sorting
            })
        
        # If we don't have enough results and haven't exhausted all candidates, try to get more
        if len(output) < top_k and max_candidates < len(self.df):
            try:
                # Get more candidates, excluding ones we already processed
                expanded_candidates = min(len(self.df), top_k * 20)
                all_indices = similarity_scores.argsort()[-expanded_candidates:][::-1]
                # Get only new indices that weren't in the first batch
                processed_indices_set = set(top_indices)
                new_indices = [idx for idx in all_indices if idx not in processed_indices_set]
                
                if new_indices:
                    additional_results = self.df.iloc[new_indices]
                    
                    for _, row in additional_results.iterrows():
                        if len(output) >= top_k:
                            break
                            
                        name = row.get("product_name", "Unknown Product")
                        brand = row.get("brand", "Unknown")
                        
                        # Convert to string and handle NaN/None values
                        name = str(name) if pd.notna(name) else "Unknown Product"
                        brand = str(brand) if pd.notna(brand) else "Unknown"
                        
                        product_key = (name.lower().strip(), brand.lower().strip())
                        
                        if product_key in seen_products:
                            continue
                        
                        # Check budget first before marking as seen
                        price_current = row.get("price_current", "")
                        price_retail = row.get("price_retail", "")
                        price_value = price_current if price_current else price_retail
                        
                        if budget_max is not None:
                            try:
                                if price_value and pd.notna(price_value):
                                    if float(price_value) > budget_max:
                                        continue
                            except (ValueError, TypeError):
                                pass
                        
                        # Mark as seen only if we're going to add it
                        seen_products.add(product_key)
                        
                        try:
                            if price_value and pd.notna(price_value):
                                price = f"${float(price_value):.2f}"
                            else:
                                price = "N/A"
                        except (ValueError, TypeError):
                            price = "N/A"
                        
                        bestseller_rank = row.get("bestseller_rank", "")
                        try:
                            if bestseller_rank and pd.notna(bestseller_rank):
                                rating = f"Rank #{int(float(bestseller_rank))}"
                            else:
                                rating = "N/A"
                        except (ValueError, TypeError):
                            rating = "N/A"
                        
                        # Store numeric values for sorting
                        price_num = None
                        try:
                            if price_value and pd.notna(price_value):
                                price_num = float(price_value)
                        except (ValueError, TypeError):
                            price_num = float('inf')
                        
                        rating_num = None
                        try:
                            if bestseller_rank and pd.notna(bestseller_rank):
                                rating_num = int(float(bestseller_rank))
                            else:
                                rating_num = float('inf')
                        except (ValueError, TypeError):
                            rating_num = float('inf')
                        
                        # Get product URL
                        product_url = row.get("product_url", "")
                        product_url = str(product_url) if pd.notna(product_url) and product_url else ""
                        
                        output.append({
                            "name": name,
                            "brand": brand,
                            "price": price,
                            "rating": rating,
                            "url": product_url,
                            "_price_num": price_num,  # For sorting
                            "_rating_num": rating_num  # For sorting
                        })
            except Exception as e:
                # If there's an error in expanded search, just return what we have
                print(f"Warning: Error in expanded search: {e}")
        
        # Sort the results based on sort_by parameter
        if sort_by == "Price: Low to High":
            output.sort(key=lambda x: x.get("_price_num", float('inf')))
        elif sort_by == "Price: High to Low":
            # For high to low, use -inf for missing prices so they go to the end
            output.sort(key=lambda x: x.get("_price_num") if x.get("_price_num") != float('inf') else float('-inf'), reverse=True)
        elif sort_by == "Rating: Best First":
            output.sort(key=lambda x: x.get("_rating_num", float('inf')))
        # "Similarity (Default)" - no sorting needed, already sorted by similarity
        
        # Remove sorting helper keys before returning
        for item in output:
            item.pop("_price_num", None)
            item.pop("_rating_num", None)
        
        return output
