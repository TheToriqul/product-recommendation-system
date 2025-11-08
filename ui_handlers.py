"""
UI Event Handlers for Product Recommendation System GUI

This module contains event handler functions and business logic for the GUI.
"""

import logging
import tkinter as tk
from tkinter import messagebox, filedialog
from typing import List, Dict, Optional, Callable
import webbrowser
import json
import csv
from datetime import datetime
import os

from ui_constants import BUDGET_MAP

logger = logging.getLogger(__name__)


def handle_item_click(
    event: tk.Event,
    tree: tk.ttk.Treeview,
    use_advanced: bool,
    show_similar_callback
) -> None:
    """
    Handle click event on a table row to show similar products.
    
    Args:
        event: Tkinter click event
        tree: Treeview widget
        use_advanced: Whether advanced features are enabled
        show_similar_callback: Callback to show similar products
    """
    item = tree.selection()[0] if tree.selection() else None
    if item:
        values = tree.item(item, "values")
        min_values = 5 if use_advanced else 4
        if len(values) >= min_values:
            product_name = values[0]
            brand = values[1]
            tags = tree.item(item, "tags")
            url = tags[0] if tags else ""
            show_similar_callback(product_name, brand, url)


def handle_item_double_click(event: tk.Event, tree: tk.ttk.Treeview) -> None:
    """
    Handle double-click event to open product URL.
    
    Args:
        event: Tkinter double-click event
        tree: Treeview widget
    """
    item = tree.selection()[0] if tree.selection() else None
    if item:
        tags = tree.item(item, "tags")
        url = tags[0] if tags else ""
        if url and url.startswith("http"):
            webbrowser.open(url)
            logger.info(f"Opened URL: {url}")
        elif url:
            messagebox.showinfo("No URL", "Product URL not available for this item.")


def handle_similar_double_click(event: tk.Event, similar_tree: tk.ttk.Treeview) -> None:
    """
    Handle double-click on similar products table.
    
    Args:
        event: Tkinter double-click event
        similar_tree: Similar products treeview widget
    """
    item = similar_tree.selection()[0] if similar_tree.selection() else None
    if item:
        tags = similar_tree.item(item, "tags")
        url = tags[0] if tags else ""
        if url and url.startswith("http"):
            webbrowser.open(url)
            logger.info(f"Opened URL: {url}")


def show_similar_products(
    similar_tree: tk.ttk.Treeview,
    engine,
    product_name: str,
    brand: str,
    product_url: str
) -> None:
    """
    Show similar products in the similar products section.
    
    Args:
        similar_tree: Similar products treeview widget
        engine: ProductRecommender instance
        product_name: Name of the selected product
        brand: Brand of the selected product
        product_url: URL of the selected product (unused but kept for compatibility)
    """
    # Clear previous similar products
    for item in similar_tree.get_children():
        similar_tree.delete(item)
    
    try:
        similar_products = engine.get_similar_products(product_name, brand, top_k=8)
        
        if not similar_products:
            similar_tree.insert("", tk.END, values=("No similar products found", "", "", "", ""))
        else:
            for product in similar_products:
                view_text = "Click to View" if product.get("url", "").startswith("http") else "N/A"
                similar_tree.insert(
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
    except Exception as e:
        logger.error(f"Error showing similar products: {e}", exc_info=True)
        messagebox.showerror("Error", f"Failed to load similar products:\n{e}")


def perform_search(
    product_entry_var: tk.StringVar,
    brand_var: tk.StringVar,
    budget_var: tk.StringVar,
    sort_var: tk.StringVar,
    tree: tk.ttk.Treeview,
    similar_tree: tk.ttk.Treeview,
    engine,
    use_advanced: bool,
    update_export_button_state: Optional[Callable[[bool], None]] = None
) -> List[Dict[str, str]]:
    """
    Perform product search and update the results table.
    
    Args:
        product_entry_var: Product entry StringVar
        brand_var: Brand selection StringVar
        budget_var: Budget selection StringVar
        sort_var: Sort selection StringVar
        tree: Main results treeview widget
        similar_tree: Similar products treeview widget
        engine: ProductRecommender instance
        use_advanced: Whether advanced features are enabled
        update_export_button_state: Optional callback to update export button state
        
    Returns:
        List of current search results
    """
    product = product_entry_var.get().strip()
    if not product:
        messagebox.showwarning("Input Missing", "Please enter a product type.")
        return []
    
    # Get selections
    brand_selection = brand_var.get()
    brand = "" if brand_selection == "All Brands" else brand_selection
    
    budget_selection = budget_var.get()
    budget_max = None
    if budget_selection != "No Limit":
        budget_max = BUDGET_MAP.get(budget_selection)
    
    sort_selection = sort_var.get()
    
    # Clear previous results
    for item in tree.get_children():
        tree.delete(item)
    for item in similar_tree.get_children():
        similar_tree.delete(item)
    
    # Show loading indicator
    tree.insert("", tk.END, values=("Searching...", "Please wait", "", "", ""))
    tree.update()
    
    try:
        # Use advanced recommender with diversity if available
        if use_advanced:
            recommendations = engine.recommend(
                product, 
                brand, 
                budget_max=budget_max, 
                sort_by=sort_selection,
                diversity_weight=0.2
            )
        else:
            recommendations = engine.recommend(
                product, 
                brand, 
                budget_max=budget_max, 
                sort_by=sort_selection
            )
        
        # Clear loading indicator
        for item in tree.get_children():
            tree.delete(item)
        
        # Handle error messages
        if recommendations and isinstance(recommendations[0], str):
            if recommendations[0].startswith("No relevant"):
                messagebox.showinfo("No Results", recommendations[0])
            else:
                messagebox.showerror("Error", recommendations[0])
            return []
        
        # Insert into table
        if not recommendations:
            messagebox.showinfo("No Results", "No products found matching your criteria.")
            return []
        
        current_results = [rec for rec in recommendations if isinstance(rec, dict)]
        
        for rec in recommendations:
            if isinstance(rec, dict):
                name = rec.get("name", "Unknown Product")
                brand_part = rec.get("brand", "Unknown")
                price = rec.get("price", "N/A")
                rating = rec.get("rating", "N/A")
                url = rec.get("url", "")
                similarity = rec.get("similarity_score", "N/A") if use_advanced else None
            else:
                name, brand_part, price, rating, url = str(rec), "", "N/A", "N/A", ""
                similarity = None
            
            view_text = "Click to View" if url and url.startswith("http") else "N/A"
            
            if use_advanced and similarity:
                tree.insert("", tk.END, values=(name, brand_part, price, rating, similarity, view_text), tags=(url,))
            else:
                tree.insert("", tk.END, values=(name, brand_part, price, rating, view_text), tags=(url,))
        
        logger.info(f"Displayed {len(recommendations)} recommendations")
        if update_export_button_state:
            update_export_button_state(len(current_results) > 0)
        return current_results
        
    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        messagebox.showerror("Error", f"An error occurred during search:\n{e}")
        return []
    finally:
        # Clear loading indicator
        for item in tree.get_children():
            if tree.item(item, "values")[0] == "Searching...":
                tree.delete(item)


def export_results(
    current_results: List[Dict[str, str]],
    use_advanced: bool,
    root: tk.Tk
) -> None:
    """
    Export current search results to CSV or JSON file.
    
    Args:
        current_results: List of result dictionaries to export
        use_advanced: Whether advanced features are enabled
        root: Tkinter root window
    """
    if not current_results:
        messagebox.showwarning(
            "No Data to Export", 
            "No search results available to export.\n\nPlease perform a search first."
        )
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"product_recommendations_{timestamp}.csv"
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        initialfile=default_filename,
        filetypes=[
            ("CSV files (Excel compatible)", "*.csv"),
            ("JSON files", "*.json"),
            ("All files", "*.*")
        ],
        title="Export Search Results",
        confirmoverwrite=True
    )
    
    if not file_path:
        return
    
    try:
        root.config(cursor="wait")
        root.update()
        
        if file_path.endswith('.json'):
            _export_json(file_path, current_results, use_advanced)
            file_type = "JSON"
        else:
            _export_csv(file_path, current_results, use_advanced)
            file_type = "CSV"
        
        file_name = os.path.basename(file_path)
        file_dir = os.path.dirname(file_path)
        
        messagebox.showinfo(
            "Export Successful",
            f"✓ Successfully exported {len(current_results)} results!\n\n"
            f"File: {file_name}\n"
            f"Location: {file_dir}\n"
            f"Format: {file_type}"
        )
        logger.info(f"Exported {len(current_results)} results to {file_path} ({file_type})")
        
    except PermissionError:
        error_msg = (
            "Permission denied!\n\n"
            "The file may be open in another program.\n"
            "Please close it and try again."
        )
        messagebox.showerror("Export Failed", error_msg)
        logger.error(f"Permission denied when exporting to {file_path}")
    except OSError as e:
        error_msg = (
            "File system error!\n\n"
            f"Could not write to the selected location.\n"
            f"Error: {str(e)}"
        )
        messagebox.showerror("Export Failed", error_msg)
        logger.error(f"OS error when exporting: {e}", exc_info=True)
    except Exception as e:
        error_msg = (
            "An unexpected error occurred!\n\n"
            f"Error: {str(e)}\n\n"
            "Please try again or contact support."
        )
        messagebox.showerror("Export Error", error_msg)
        logger.error(f"Error exporting results: {e}", exc_info=True)
    finally:
        root.config(cursor="")
        root.update()


def _export_csv(file_path: str, current_results: List[Dict[str, str]], use_advanced: bool) -> None:
    """Export results to CSV file."""
    fieldnames = ['Product Name', 'Brand', 'Price', 'Rating', 'Product URL']
    if use_advanced:
        fieldnames.insert(4, 'Similarity Score')
    
    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in current_results:
            row = {
                'Product Name': str(result.get('name', 'N/A')).strip(),
                'Brand': str(result.get('brand', 'N/A')).strip(),
                'Price': str(result.get('price', 'N/A')).strip(),
                'Rating': str(result.get('rating', 'N/A')).strip(),
                'Product URL': str(result.get('url', 'N/A')).strip()
            }
            
            if use_advanced and 'similarity_score' in result:
                similarity = result.get('similarity_score', 'N/A')
                if isinstance(similarity, (int, float)):
                    row['Similarity Score'] = f"{similarity:.4f}"
                else:
                    row['Similarity Score'] = str(similarity)
            
            writer.writerow(row)


def _export_json(file_path: str, current_results: List[Dict[str, str]], use_advanced: bool) -> None:
    """Export results to JSON file."""
    export_data = {
        'export_info': {
            'export_date': datetime.now().isoformat(),
            'export_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'application': 'AI-Powered Product Recommendation System',
            'version': '1.0',
            'total_results': len(current_results),
            'features_enabled': {
                'advanced_ai': use_advanced,
                'generative_ai': use_advanced  # Simplified for now
            }
        },
        'results': current_results
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, sort_keys=False)

