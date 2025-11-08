"""
UI Components for Product Recommendation System GUI

This module contains functions for creating UI components like header, input section,
results section, and similar products section.
"""

import logging
import tkinter as tk
from tkinter import ttk
from typing import Optional
from PIL import Image, ImageTk
import os

from ui_constants import (
    BG_COLOR_INPUT, BG_COLOR_CARD, BG_COLOR_ENTRY, BORDER_COLOR,
    FG_COLOR_WHITE, FG_COLOR_TEXT, FG_COLOR_SECONDARY, SUCCESS_COLOR,
    FONT_FAMILY, FONT_SIZE_TITLE, FONT_SIZE_HEADING, FONT_SIZE_NORMAL, FONT_SIZE_BOLD,
    BUTTON_PRIMARY, BUTTON_PRIMARY_HOVER, BUTTON_PRIMARY_ACTIVE,
    BUTTON_SECONDARY, BUTTON_SECONDARY_HOVER, BUTTON_SECONDARY_ACTIVE,
    BUDGET_OPTIONS, SORT_OPTIONS
)

logger = logging.getLogger(__name__)


def create_header(
    parent: tk.Frame,
    use_advanced: bool,
    logo_path: Optional[str] = None
) -> None:
    """
    Create the application header with modern design and university logo.
    
    Args:
        parent: Parent frame to attach header to
        use_advanced: Whether advanced features are enabled
        logo_path: Optional path to logo image file
    """
    header_frame = tk.Frame(parent, bg=BG_COLOR_INPUT, height=110)
    header_frame.pack(fill=tk.X, padx=0, pady=0)
    
    # Header content container
    header_content = tk.Frame(header_frame, bg=BG_COLOR_INPUT)
    header_content.pack(fill=tk.BOTH, expand=True, padx=30, pady=15)
    
    # University logo (left side)
    logo_frame = tk.Frame(header_content, bg=BG_COLOR_INPUT)
    logo_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 25))
    
    if logo_path and os.path.exists(logo_path):
        try:
            logo_img = Image.open(logo_path)
            orig_width, orig_height = logo_img.size
            
            target_width = 280
            target_height = 100
            aspect_ratio = orig_width / orig_height
            
            if aspect_ratio > (target_width / target_height):
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            
            resample_method = Image.Resampling.LANCZOS
            if logo_img.mode != 'RGB' and logo_img.mode != 'RGBA':
                logo_img = logo_img.convert('RGBA')
            
            logo_img_resized = logo_img.resize((new_width, new_height), resample=resample_method)
            if logo_img_resized.mode != 'RGBA' and logo_img_resized.mode != 'RGB':
                logo_img_resized = logo_img_resized.convert('RGB')
            
            logo_photo = ImageTk.PhotoImage(logo_img_resized)
            logo_label = tk.Label(logo_frame, image=logo_photo, bg=BG_COLOR_INPUT)
            logo_label.image = logo_photo  # Keep a reference
            logo_label.pack(anchor=tk.W)
            
            logger.info(f"Logo loaded successfully: {new_width}x{new_height}")
        except Exception as e:
            logger.warning(f"Could not load logo: {e}")
    
    # Title with subtitle (center)
    title_frame = tk.Frame(header_content, bg=BG_COLOR_INPUT)
    title_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    title_text = "AI-Powered Product Recommendation"
    subtitle_text = "Discover products with intelligent semantic search"
    
    tk.Label(
        title_frame,
        text=title_text,
        font=(FONT_FAMILY, FONT_SIZE_TITLE, 'bold'),
        fg=FG_COLOR_WHITE,
        bg=BG_COLOR_INPUT
    ).pack(anchor=tk.W)
    
    tk.Label(
        title_frame,
        text=subtitle_text,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        fg=FG_COLOR_SECONDARY,
        bg=BG_COLOR_INPUT
    ).pack(anchor=tk.W, pady=(2, 0))
    
    # Status indicator (right side)
    status_frame = tk.Frame(header_content, bg=BG_COLOR_INPUT)
    status_frame.pack(side=tk.RIGHT, fill=tk.Y)
    
    if use_advanced:
        status_text = "✓ AI Enhanced"
        status_color = SUCCESS_COLOR
    else:
        status_text = "Basic Mode"
        status_color = FG_COLOR_SECONDARY
        
    tk.Label(
        status_frame,
        text=status_text,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'),
        fg=status_color,
        bg=BG_COLOR_INPUT
    ).pack(anchor=tk.E)


def create_input_section(
    parent: tk.Frame,
    engine,
    product_entry_var: tk.StringVar,
    brand_var: tk.StringVar,
    budget_var: tk.StringVar,
    sort_var: tk.StringVar,
    search_callback
) -> tuple:
    """
    Create the input controls section with modern card design.
    
    Args:
        parent: Parent frame
        engine: ProductRecommender instance
        product_entry_var: StringVar for product entry
        brand_var: StringVar for brand selection
        budget_var: StringVar for budget selection
        sort_var: StringVar for sort selection
        search_callback: Callback function for search button
        
    Returns:
        Tuple of (product_entry widget, brand_combo widget, update_brand_callback)
    """
    input_main_container = tk.Frame(parent, bg=BG_COLOR_CARD)
    input_main_container.pack(fill=tk.X, side=tk.TOP, padx=25, pady=15)
    
    input_frame = tk.Frame(input_main_container, bg=BG_COLOR_CARD, relief=tk.FLAT, bd=0)
    input_frame.pack(fill=tk.X, pady=0)
    
    input_inner = tk.Frame(input_frame, bg=BG_COLOR_CARD)
    input_inner.pack(fill=tk.X, padx=20, pady=15)
    
    # Product entry
    tk.Label(
        input_inner, 
        text="Product:", 
        fg=FG_COLOR_TEXT, 
        bg=BG_COLOR_CARD, 
        font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold')
    ).pack(side=tk.LEFT, padx=(0, 8))
    
    product_entry = tk.Entry(
        input_inner, 
        font=(FONT_FAMILY, FONT_SIZE_NORMAL), 
        width=28, 
        bg=BG_COLOR_ENTRY, 
        fg=FG_COLOR_WHITE, 
        relief=tk.SOLID,
        insertbackground=FG_COLOR_WHITE,
        bd=1,
        highlightthickness=2,
        highlightbackground=BORDER_COLOR,
        highlightcolor=BUTTON_PRIMARY,
        textvariable=product_entry_var
    )
    # Pack with consistent padding to match combobox height
    product_entry.pack(side=tk.LEFT, padx=(0, 15), pady=10, ipady=2)
    
    # Brand dropdown
    tk.Label(
        input_inner, 
        text="Brand:", 
        fg=FG_COLOR_TEXT, 
        bg=BG_COLOR_CARD, 
        font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold')
    ).pack(side=tk.LEFT, padx=(0, 8))
    
    all_brands = engine.get_available_brands("")
    brands = ["All Brands"] + all_brands
    brand_combo = ttk.Combobox(
        input_inner, 
        textvariable=brand_var, 
        values=brands, 
        font=(FONT_FAMILY, FONT_SIZE_NORMAL), 
        width=20, 
        state='readonly'
    )
    brand_combo.set("All Brands")
    brand_combo.pack(side=tk.LEFT, padx=(0, 15), pady=10)
    
    # Budget dropdown
    tk.Label(
        input_inner, 
        text="Budget:", 
        fg=FG_COLOR_TEXT, 
        bg=BG_COLOR_CARD, 
        font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold')
    ).pack(side=tk.LEFT, padx=(0, 8))
    
    budget_combo = ttk.Combobox(
        input_inner, 
        textvariable=budget_var, 
        values=BUDGET_OPTIONS,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL), 
        width=16, 
        state='readonly'
    )
    budget_combo.set("No Limit")
    budget_combo.pack(side=tk.LEFT, padx=(0, 15), pady=10)
    
    # Sort dropdown
    tk.Label(
        input_inner, 
        text="Sort By:", 
        fg=FG_COLOR_TEXT, 
        bg=BG_COLOR_CARD, 
        font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold')
    ).pack(side=tk.LEFT, padx=(0, 8))
    
    sort_combo = ttk.Combobox(
        input_inner, 
        textvariable=sort_var, 
        values=SORT_OPTIONS,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL), 
        width=20, 
        state='readonly'
    )
    sort_combo.set("Similarity (Default)")
    sort_combo.pack(side=tk.LEFT, padx=(0, 15), pady=10)
    
    # Search button - Enhanced design with vibrant colors
    send_button = tk.Button(
        input_inner,
        text="🔍 Search",
        command=search_callback,
        font=(FONT_FAMILY, FONT_SIZE_BOLD, 'bold'),
        bg=BUTTON_PRIMARY,
        fg=FG_COLOR_WHITE,
        activebackground=BUTTON_PRIMARY_ACTIVE,
        activeforeground=FG_COLOR_WHITE,
        highlightbackground=BUTTON_PRIMARY,
        highlightcolor=BUTTON_PRIMARY,
        highlightthickness=0,
        relief=tk.FLAT,
        bd=0,
        cursor='hand2',
        padx=24,
        pady=10
    )
    
    # Force button color on macOS
    try:
        send_button.configure(bg=BUTTON_PRIMARY)
    except:
        pass
    
    def on_enter(e):
        send_button.config(
            bg=BUTTON_PRIMARY_HOVER, 
            highlightbackground=BUTTON_PRIMARY_HOVER
        )
    def on_leave(e):
        send_button.config(
            bg=BUTTON_PRIMARY, 
            highlightbackground=BUTTON_PRIMARY
        )
    send_button.bind("<Enter>", on_enter)
    send_button.bind("<Leave>", on_leave)
    send_button.pack(side=tk.RIGHT, padx=(15, 0), pady=8)
    
    def update_brand_dropdown(event: Optional[tk.Event] = None) -> None:
        """Update brand dropdown based on product entry."""
        product = product_entry_var.get().strip()
        try:
            if product:
                brands = engine.get_available_brands(product)
                brand_values = ["All Brands"] + brands
                brand_combo['values'] = brand_values
                current_brand = brand_var.get()
                if current_brand not in brand_values:
                    brand_combo.set("All Brands")
            else:
                all_brands = engine.get_available_brands("")
                brand_values = ["All Brands"] + all_brands
                brand_combo['values'] = brand_values
                brand_combo.set("All Brands")
        except Exception as e:
            logger.error(f"Error updating brand dropdown: {e}", exc_info=True)
    
    product_entry.bind('<KeyRelease>', update_brand_dropdown)
    
    return product_entry, brand_combo, send_button


def create_results_section(parent: tk.Frame, use_advanced: bool) -> tuple:
    """
    Create the main results table section with modern design.
    
    Args:
        parent: Parent frame
        use_advanced: Whether advanced features are enabled (affects columns)
        
    Returns:
        Tuple of (content_container frame, tree widget)
    """
    content_container = tk.Frame(parent, bg=BG_COLOR_CARD)
    content_container.pack(fill=tk.BOTH, expand=True, padx=25, pady=(0, 10))
    
    results_card = tk.Frame(content_container, bg=BG_COLOR_CARD, relief=tk.FLAT, bd=0)
    results_card.pack(fill=tk.BOTH, expand=True)
    
    results_header = tk.Frame(results_card, bg=BG_COLOR_CARD)
    results_header.pack(fill=tk.X, padx=20, pady=(15, 10))
    
    tk.Label(
        results_header,
        text="Search Results",
        font=(FONT_FAMILY, FONT_SIZE_HEADING, 'bold'),
        fg=FG_COLOR_WHITE,
        bg=BG_COLOR_CARD
    ).pack(side=tk.LEFT)
    
    table_frame = tk.Frame(results_card, bg=BG_COLOR_CARD)
    table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
    
    if use_advanced:
        columns = ("Product Name", "Brand", "Price", "Rating", "Similarity", "View Product")
    else:
        columns = ("Product Name", "Brand", "Price", "Rating", "View Product")
        
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview.Heading", background=BG_COLOR_CARD, foreground=FG_COLOR_WHITE, font=(FONT_FAMILY, 10, 'bold'))
    style.map("Treeview.Heading", background=[("active", BUTTON_SECONDARY_HOVER)])
    
    tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
    
    for col in columns:
        tree.heading(col, text=col)
        if col == "View Product":
            tree.column(col, anchor=tk.CENTER, width=130)
        elif col == "Similarity":
            tree.column(col, anchor=tk.CENTER, width=110)
        elif col == "Product Name":
            tree.column(col, anchor=tk.W, width=300)
        else:
            tree.column(col, anchor=tk.W, width=150)
    
    tree.pack(fill=tk.BOTH, expand=True)
    
    scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    return content_container, tree


def create_similar_section(parent: tk.Frame) -> ttk.Treeview:
    """
    Create the similar products section with modern design.
    
    Args:
        parent: Parent frame (content_container)
        
    Returns:
        Similar products tree widget
    """
    similar_card = tk.Frame(parent, bg=BG_COLOR_CARD, relief=tk.FLAT, bd=0)
    similar_card.pack(fill=tk.BOTH, expand=False, pady=(5, 15))
    
    similar_header = tk.Frame(similar_card, bg=BG_COLOR_CARD)
    similar_header.pack(fill=tk.X, padx=20, pady=(10, 8))
    
    tk.Label(
        similar_header,
        text="You May Also Like",
        font=(FONT_FAMILY, FONT_SIZE_HEADING, 'bold'),
        fg=FG_COLOR_WHITE,
        bg=BG_COLOR_CARD
    ).pack(side=tk.LEFT)
    
    similar_table_frame = tk.Frame(similar_card, bg=BG_COLOR_CARD)
    similar_table_frame.pack(fill=tk.BOTH, expand=False, padx=20, pady=(0, 15))
    
    similar_columns = ("Product Name", "Brand", "Price", "Rating", "View")
    similar_tree = ttk.Treeview(
        similar_table_frame, 
        columns=similar_columns, 
        show="headings", 
        height=7
    )
    
    for col in similar_columns:
        similar_tree.heading(col, text=col)
        if col == "View":
            similar_tree.column(col, anchor=tk.CENTER, width=100)
        else:
            similar_tree.column(col, anchor=tk.W, width=180)
    
    similar_tree.pack(fill=tk.BOTH, expand=True)
    
    similar_scrollbar = ttk.Scrollbar(
        similar_table_frame, 
        orient=tk.VERTICAL, 
        command=similar_tree.yview
    )
    similar_tree.configure(yscroll=similar_scrollbar.set)
    similar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    return similar_tree


def create_export_button(parent: tk.Frame, export_callback) -> tk.Button:
    """
    Create export button for results.
    
    Args:
        parent: Parent frame
        export_callback: Callback function for export button
        
    Returns:
        Export button widget
    """
    export_frame = tk.Frame(parent, bg=BG_COLOR_CARD)
    export_frame.pack(fill=tk.X, padx=25, pady=(0, 15))
    
    export_button = tk.Button(
        export_frame,
        text="📥 Export to CSV/JSON",
        command=export_callback,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'),
        bg=BUTTON_SECONDARY,
        fg=FG_COLOR_WHITE,
        activebackground=BUTTON_SECONDARY_ACTIVE,
        activeforeground=FG_COLOR_WHITE,
        highlightbackground=BUTTON_SECONDARY,
        highlightcolor=BUTTON_SECONDARY,
        highlightthickness=0,
        relief=tk.FLAT,
        bd=0,
        cursor='hand2',
        state=tk.DISABLED,
        padx=24,
        pady=12
    )
    
    # Force button color on macOS
    try:
        export_button.configure(bg=BUTTON_SECONDARY)
    except:
        pass
    
    def on_enter_export(e):
        if export_button['state'] != 'disabled':
            export_button.config(
                bg=BUTTON_SECONDARY_HOVER, 
                highlightbackground=BUTTON_SECONDARY_HOVER
            )
    def on_leave_export(e):
        if export_button['state'] != 'disabled':
            export_button.config(
                bg=BUTTON_SECONDARY, 
                highlightbackground=BUTTON_SECONDARY
            )
    export_button.bind("<Enter>", on_enter_export)
    export_button.bind("<Leave>", on_leave_export)
    export_button.pack(side=tk.RIGHT, padx=(0, 0))
    
    return export_button

