# app_gui_table.py
import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser
from recommender_engine import ProductRecommender


class ProdRecommendationApp:
    def __init__(self, root):
        # Initialize recommender
        self.engine = ProductRecommender()
        self.bgcolor = '#50657D'

        # Window setup
        self.root = root
        self.root.title("Product Recommendation System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#0a0a0a')

        main_frame = tk.Frame(self.root, bg=self.bgcolor)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = tk.Frame(main_frame, bg=self.bgcolor, height=80)
        header_frame.pack(fill=tk.X)
        tk.Label(
            header_frame,
            text="Product Recommendation System",
            font=('Segoe UI', 24, 'bold'),
            fg='#ffffff',
            bg=self.bgcolor
        ).pack(pady=(15, 0))

        # Input frame
        input_main_container = tk.Frame(main_frame, bg=self.bgcolor)
        input_main_container.pack(fill=tk.X, side=tk.TOP, padx=20, pady=10)
        input_frame = tk.Frame(input_main_container, bg='#2d2d2d')
        input_frame.pack(fill=tk.X, pady=(10, 0))

        # Product entry
        tk.Label(input_frame, text="Product:", fg='white', bg='#2d2d2d', font=('Segoe UI', 11)).pack(side=tk.LEFT, padx=(10, 5))
        self.product_entry = tk.Entry(input_frame, font=('Segoe UI', 11), width=25, bg='#3a3a3a', fg='white', relief=tk.FLAT)
        self.product_entry.pack(side=tk.LEFT, padx=(0, 10), pady=10)

        # Brand dropdown
        tk.Label(input_frame, text="Brand:", fg='white', bg='#2d2d2d', font=('Segoe UI', 11)).pack(side=tk.LEFT)
        self.brand_var = tk.StringVar()
        # Initialize with all brands, will update dynamically when product is typed
        all_brands = self.engine.get_available_brands("")
        brands = ["All Brands"] + all_brands
        self.brand_combo = ttk.Combobox(input_frame, textvariable=self.brand_var, values=brands, 
                                        font=('Segoe UI', 11), width=18, state='readonly')
        self.brand_combo.set("All Brands")
        self.brand_combo.pack(side=tk.LEFT, padx=(0, 10), pady=10)
        
        # Bind product entry to update brand dropdown
        self.product_entry.bind('<KeyRelease>', self.update_brand_dropdown)
        
        # Budget dropdown
        tk.Label(input_frame, text="Budget:", fg='white', bg='#2d2d2d', font=('Segoe UI', 11)).pack(side=tk.LEFT)
        self.budget_var = tk.StringVar()
        budget_options = ["No Limit", "Under $100", "Under $300", "Under $500", "Under $1000", "Under $2000"]
        self.budget_combo = ttk.Combobox(input_frame, textvariable=self.budget_var, values=budget_options,
                                         font=('Segoe UI', 11), width=15, state='readonly')
        self.budget_combo.set("No Limit")
        self.budget_combo.pack(side=tk.LEFT, padx=(0, 10), pady=10)
        
        # Sort dropdown
        tk.Label(input_frame, text="Sort By:", fg='white', bg='#2d2d2d', font=('Segoe UI', 11)).pack(side=tk.LEFT)
        self.sort_var = tk.StringVar()
        sort_options = ["Similarity (Default)", "Price: Low to High", "Price: High to Low", "Rating: Best First"]
        self.sort_combo = ttk.Combobox(input_frame, textvariable=self.sort_var, values=sort_options,
                                       font=('Segoe UI', 11), width=18, state='readonly')
        self.sort_combo.set("Similarity (Default)")
        self.sort_combo.pack(side=tk.LEFT, padx=(0, 10), pady=10)

        send_button = tk.Button(
            input_frame,
            text="Search",
            command=self.send_message,
            font=('Segoe UI', 11, 'bold'),
            bg='#0078d4',
            fg='white',
            activebackground='#106ebe',
            relief=tk.FLAT,
            bd=0,
            width=10,
            cursor='hand2'
        )
        send_button.pack(side=tk.RIGHT, padx=(0, 15), pady=10)

        # Main content container
        content_container = tk.Frame(main_frame, bg='#0a0a0a')
        content_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        
        # Table / Treeview for recommendations
        table_frame = tk.Frame(content_container, bg='#0a0a0a')
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("Product Name", "Brand", "Price", "Rating", "View Product")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)

        for col in columns:
            self.tree.heading(col, text=col)
            if col == "View Product":
                self.tree.column(col, anchor=tk.CENTER, width=120)
            else:
                self.tree.column(col, anchor=tk.W, width=200)
        
        # Bind click events
        self.tree.bind("<Button-1>", self.on_item_click)  # Single click for similar products
        self.tree.bind("<Double-1>", self.on_item_double_click)  # Double click for URL

        self.tree.pack(fill=tk.BOTH, expand=True)

        # Scrollbar for table
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # "You May Also Like" section
        similar_section_frame = tk.Frame(content_container, bg='#0a0a0a')
        similar_section_frame.pack(fill=tk.BOTH, expand=False, pady=(10, 0))
        
        # Header for similar products
        similar_header = tk.Frame(similar_section_frame, bg=self.bgcolor)
        similar_header.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(
            similar_header,
            text="You May Also Like",
            font=('Segoe UI', 16, 'bold'),
            fg='#ffffff',
            bg=self.bgcolor
        ).pack(side=tk.LEFT, padx=10, pady=5)
        
        # Similar products table
        similar_table_frame = tk.Frame(similar_section_frame, bg='#0a0a0a')
        similar_table_frame.pack(fill=tk.BOTH, expand=False)
        
        similar_columns = ("Product Name", "Brand", "Price", "Rating", "View")
        self.similar_tree = ttk.Treeview(similar_table_frame, columns=similar_columns, show="headings", height=8)
        
        for col in similar_columns:
            self.similar_tree.heading(col, text=col)
            if col == "View":
                self.similar_tree.column(col, anchor=tk.CENTER, width=100)
            else:
                self.similar_tree.column(col, anchor=tk.W, width=180)
        
        self.similar_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for similar products table
        similar_scrollbar = ttk.Scrollbar(similar_table_frame, orient=tk.VERTICAL, command=self.similar_tree.yview)
        self.similar_tree.configure(yscroll=similar_scrollbar.set)
        similar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click for similar products
        self.similar_tree.bind("<Double-1>", self.on_similar_double_click)

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background="#1e1e1e",
                        foreground="white",
                        fieldbackground="#1e1e1e",
                        font=('Segoe UI', 11))
        style.configure("Treeview.Heading", font=('Segoe UI', 12, 'bold'), background="#2d2d2d", foreground="white")
        # Style for Combobox
        style.configure("TCombobox",
                       fieldbackground="#3a3a3a",
                       background="#3a3a3a",
                       foreground="white",
                       borderwidth=0,
                       relief=tk.FLAT)
        style.map("TCombobox",
                 fieldbackground=[("readonly", "#3a3a3a")],
                 background=[("readonly", "#3a3a3a")])

    def update_brand_dropdown(self, event=None):
        """Update brand dropdown based on the product type entered"""
        product = self.product_entry.get().strip()
        
        if product:
            # Get brands available for this product type
            brands = self.engine.get_available_brands(product)
            brand_values = ["All Brands"] + brands
            self.brand_combo['values'] = brand_values
            # Reset to "All Brands" if current selection is not in new list
            current_brand = self.brand_var.get()
            if current_brand not in brand_values:
                self.brand_combo.set("All Brands")
        else:
            # If product is empty, show all brands
            all_brands = self.engine.get_available_brands("")
            brand_values = ["All Brands"] + all_brands
            self.brand_combo['values'] = brand_values
            self.brand_combo.set("All Brands")
    
    def on_item_click(self, event):
        """Show similar products below when clicking on a row"""
        item = self.tree.selection()[0] if self.tree.selection() else None
        if item:
            values = self.tree.item(item, "values")
            if len(values) >= 4:  # Check if we have product data
                product_name = values[0]
                brand = values[1]
                # Get URL from tags
                tags = self.tree.item(item, "tags")
                url = tags[0] if tags else ""
                
                # Show similar products below
                self.show_similar_products_below(product_name, brand, url)
    
    def on_similar_double_click(self, event):
        """Open product URL when double-clicking on a similar product row"""
        item = self.similar_tree.selection()[0] if self.similar_tree.selection() else None
        if item:
            tags = self.similar_tree.item(item, "tags")
            url = tags[0] if tags else ""
            if url and url.startswith("http"):
                webbrowser.open(url)
    
    def on_item_double_click(self, event):
        """Open product URL when double-clicking on a row"""
        item = self.tree.selection()[0] if self.tree.selection() else None
        if item:
            tags = self.tree.item(item, "tags")
            url = tags[0] if tags else ""
            if url and url.startswith("http"):
                webbrowser.open(url)
            elif url:
                messagebox.showinfo("No URL", "Product URL not available for this item.")
    
    def show_similar_products_below(self, product_name, brand, product_url):
        """Show similar products in the section below the main table"""
        # Clear previous similar products
        for item in self.similar_tree.get_children():
            self.similar_tree.delete(item)
        
        # Get similar products
        similar_products = self.engine.get_similar_products(product_name, brand, top_k=8)
        
        if not similar_products:
            self.similar_tree.insert("", tk.END, values=("No similar products found", "", "", "", ""))
        else:
            for product in similar_products:
                view_text = "Click to View" if product.get("url", "").startswith("http") else "N/A"
                self.similar_tree.insert(
                    "", tk.END,
                    values=(
                        product.get("name", "Unknown"),
                        product.get("brand", "Unknown"),
                        product.get("price", "N/A"),
                        product.get("rating", "N/A"),
                        view_text
                    ),
                    tags=(product.get("url", ""),)
                )

    def send_message(self):
        product = self.product_entry.get().strip()
        if not product:
            messagebox.showwarning("Input Missing", "Please enter a product type.")
            return

        # Get brand selection
        brand_selection = self.brand_var.get()
        brand = "" if brand_selection == "All Brands" else brand_selection
        
        # Get budget selection and convert to max price
        budget_selection = self.budget_var.get()
        budget_max = None
        if budget_selection != "No Limit":
            budget_map = {
                "Under $100": 100,
                "Under $300": 300,
                "Under $500": 500,
                "Under $1000": 1000,
                "Under $2000": 2000
            }
            budget_max = budget_map.get(budget_selection)
        
        # Get sort selection
        sort_selection = self.sort_var.get()

        # Clear previous results
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Clear similar products section
        for item in self.similar_tree.get_children():
            self.similar_tree.delete(item)

        recommendations = self.engine.recommend(product, brand, budget_max=budget_max, sort_by=sort_selection)
        
        # Insert into table
        for rec in recommendations:
            # Handle both dict format (new) and string format (old, for backward compatibility)
            if isinstance(rec, dict):
                name = rec.get("name", "Unknown Product")
                brand_part = rec.get("brand", "Unknown")
                price = rec.get("price", "N/A")
                rating = rec.get("rating", "N/A")
                url = rec.get("url", "")
            elif isinstance(rec, str) and rec.startswith("✅"):
                # Legacy string format parsing
                try:
                    parts = rec[2:].split(" (Brand: ")
                    name = parts[0].strip()
                    brand_part = parts[1].split(")")[0]
                    price = parts[1].split(")")[1].strip() if len(parts[1].split(")")) > 1 else "N/A"
                    rating = "N/A"
                    url = ""
                except:
                    name, brand_part, price, rating, url = rec, "", "N/A", "N/A", ""
            else:
                name, brand_part, price, rating, url = rec, "", "N/A", "N/A", ""
            
            # Display "Click to View" if URL exists, otherwise "N/A"
            view_text = "Click to View" if url and url.startswith("http") else "N/A"
            
            # Insert with URL stored in tags for retrieval
            item_id = self.tree.insert("", tk.END, values=(name, brand_part, price, rating, view_text), tags=(url,))


def main():
    root = tk.Tk()
    app = ProdRecommendationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
