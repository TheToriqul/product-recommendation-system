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

from src.ui.ui_constants import (
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
    
    # Debouncing for brand dropdown update (professional best practice)
    # Prevents excessive updates while user is typing
    _update_timer_id = None
    
    def update_brand_dropdown(event: Optional[tk.Event] = None, immediate: bool = False) -> None:
        """Update brand dropdown based on product entry with debouncing."""
        nonlocal _update_timer_id
        
        def do_update():
            try:
                product = product_entry_var.get().strip()
                if product:
                    # Filter brands to show only those with matching products
                    brands = engine.get_available_brands(product)
                    if brands:  # Only update if we got brands
                        brand_values = ["All Brands"] + brands
                        brand_combo['values'] = brand_values
                        current_brand = brand_var.get()
                        # Reset to "All Brands" if current selection is not in filtered list
                        if current_brand not in brand_values:
                            brand_combo.set("All Brands")
                    else:
                        # If no brands found, show all brands as fallback
                        all_brands = engine.get_available_brands("")
                        if all_brands:
                            brand_values = ["All Brands"] + all_brands
                            brand_combo['values'] = brand_values
                            brand_combo.set("All Brands")
                else:
                    # Show all brands when product field is empty
                    all_brands = engine.get_available_brands("")
                    if all_brands:  # Only update if we got brands
                        brand_values = ["All Brands"] + all_brands
                        brand_combo['values'] = brand_values
                        brand_combo.set("All Brands")
            except Exception as e:
                logger.error(f"Error updating brand dropdown: {e}", exc_info=True)
        
        # If immediate update requested (e.g., field cleared), update right away
        if immediate:
            # Cancel any pending updates
            if _update_timer_id is not None:
                try:
                    product_entry.after_cancel(_update_timer_id)
                except:
                    pass
            do_update()
            return
        
        # Cancel previous timer if user is still typing
        if _update_timer_id is not None:
            try:
                product_entry.after_cancel(_update_timer_id)
            except:
                pass  # Timer might have already fired
        
        # Schedule update after 200ms of no typing (debouncing - reduced for better responsiveness)
        _update_timer_id = product_entry.after(200, do_update)
    
    # Bind to both KeyRelease and when field loses focus for immediate update
    product_entry.bind('<KeyRelease>', update_brand_dropdown)
    
    def on_focus_out(event):
        """Update brand dropdown when field loses focus - immediate update."""
        update_brand_dropdown(event, immediate=True)
    
    product_entry.bind('<FocusOut>', on_focus_out)
    
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
    content_container.pack(fill=tk.BOTH, expand=True, padx=25, pady=(0, 8))
    
    results_card = tk.Frame(content_container, bg=BG_COLOR_CARD, relief=tk.FLAT, bd=0)
    results_card.pack(fill=tk.BOTH, expand=True)
    
    results_header = tk.Frame(results_card, bg=BG_COLOR_CARD)
    results_header.pack(fill=tk.X, padx=20, pady=(12, 8))
    
    tk.Label(
        results_header,
        text="Search Results",
        font=(FONT_FAMILY, FONT_SIZE_HEADING, 'bold'),
        fg=FG_COLOR_WHITE,
        bg=BG_COLOR_CARD
    ).pack(side=tk.LEFT)
    
    # Create a frame for the table with scrollbar
    table_frame = tk.Frame(results_card, bg=BG_COLOR_CARD)
    table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
    
    # Create a container for treeview and scrollbar using grid for better control
    table_container = tk.Frame(table_frame, bg=BG_COLOR_CARD)
    table_container.pack(fill=tk.BOTH, expand=True)
    
    if use_advanced:
        columns = ("Product Name", "Brand", "Price", "Rating", "Similarity", "View Product")
    else:
        columns = ("Product Name", "Brand", "Price", "Rating", "View Product")
        
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview.Heading", background=BG_COLOR_CARD, foreground=FG_COLOR_WHITE, font=(FONT_FAMILY, 10, 'bold'))
    style.map("Treeview.Heading", background=[("active", BUTTON_SECONDARY_HOVER)])
    
    # Set a fixed height to show ~8 rows within 1280x720 window
    # This ensures rows are visible and fits within the window size
    tree = ttk.Treeview(table_container, columns=columns, show="headings", height=8)
    
    for col in columns:
        tree.heading(col, text=col)
        if col == "View Product":
            tree.column(col, anchor=tk.CENTER, width=130, minwidth=100)
        elif col == "Similarity":
            tree.column(col, anchor=tk.CENTER, width=110, minwidth=80)
        elif col == "Product Name":
            tree.column(col, anchor=tk.W, width=300, minwidth=200)
        else:
            tree.column(col, anchor=tk.W, width=150, minwidth=100)
    
    # Use grid layout for better control over expansion
    tree.grid(row=0, column=0, sticky="nsew")
    
    # Configure grid weights for proper expansion
    table_container.grid_rowconfigure(0, weight=1)
    table_container.grid_columnconfigure(0, weight=1)
    
    # Create scrollbar
    scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.grid(row=0, column=1, sticky="ns")
    
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
    similar_card.pack(fill=tk.BOTH, expand=False, pady=(3, 10))
    
    similar_header = tk.Frame(similar_card, bg=BG_COLOR_CARD)
    similar_header.pack(fill=tk.X, padx=20, pady=(8, 6))
    
    tk.Label(
        similar_header,
        text="You May Also Like",
        font=(FONT_FAMILY, FONT_SIZE_HEADING, 'bold'),
        fg=FG_COLOR_WHITE,
        bg=BG_COLOR_CARD
    ).pack(side=tk.LEFT)
    
    similar_table_frame = tk.Frame(similar_card, bg=BG_COLOR_CARD)
    similar_table_frame.pack(fill=tk.BOTH, expand=False, padx=20, pady=(0, 10))
    
    # Create container for similar tree with scrollbar
    similar_table_container = tk.Frame(similar_table_frame, bg=BG_COLOR_CARD)
    similar_table_container.pack(fill=tk.BOTH, expand=True)
    
    similar_columns = ("Product Name", "Brand", "Price", "Rating", "View")
    # Set height for similar products to fit within 1280x720 window (shows ~4-5 rows)
    # Reduced to ensure everything fits without cutting
    similar_tree = ttk.Treeview(
        similar_table_container, 
        columns=similar_columns, 
        show="headings",
        height=5
    )
    
    for col in similar_columns:
        similar_tree.heading(col, text=col)
        if col == "View":
            similar_tree.column(col, anchor=tk.CENTER, width=100, minwidth=80)
        else:
            similar_tree.column(col, anchor=tk.W, width=180, minwidth=120)
    
    # Use grid layout for better control
    similar_tree.grid(row=0, column=0, sticky="nsew")
    
    # Configure grid weights
    similar_table_container.grid_rowconfigure(0, weight=1)
    similar_table_container.grid_columnconfigure(0, weight=1)
    
    similar_scrollbar = ttk.Scrollbar(
        similar_table_container, 
        orient=tk.VERTICAL, 
        command=similar_tree.yview
    )
    similar_tree.configure(yscroll=similar_scrollbar.set)
    similar_scrollbar.grid(row=0, column=1, sticky="ns")
    
    return similar_tree


def create_evaluation_tab(parent: tk.Frame) -> tuple:
    """
    Create the evaluation tab with metrics display and controls.
    
    Args:
        parent: Parent frame (tab)
        
    Returns:
        Tuple of (metrics_canvas, buttons_frame, status_label, scrollable_frame)
    """
    from src.ui.ui_constants import (
        BG_COLOR_CARD, BG_COLOR_ENTRY, BG_COLOR_MAIN, BG_COLOR_HOVER, FG_COLOR_WHITE, FG_COLOR_TEXT,
        FG_COLOR_SECONDARY, SUCCESS_COLOR, FONT_FAMILY, FONT_SIZE_NORMAL,
        FONT_SIZE_HEADING, BUTTON_PRIMARY, BUTTON_PRIMARY_HOVER, BORDER_COLOR
    )
    
    # Main container
    main_container = tk.Frame(parent, bg=BG_COLOR_MAIN)
    main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
    
    # Header
    header_frame = tk.Frame(main_container, bg=BG_COLOR_MAIN)
    header_frame.pack(fill=tk.X, pady=(0, 15))
    
    tk.Label(
        header_frame,
        text="📊 System Evaluation & Performance Metrics",
        font=(FONT_FAMILY, FONT_SIZE_HEADING + 2, 'bold'),
        fg=FG_COLOR_WHITE,
        bg=BG_COLOR_MAIN
    ).pack(side=tk.LEFT)
    
    # Buttons frame
    buttons_frame = tk.Frame(main_container, bg=BG_COLOR_MAIN)
    buttons_frame.pack(fill=tk.X, pady=(0, 15))
    
    # Status label
    status_label = tk.Label(
        main_container,
        text="No evaluation results loaded. Click 'Run Quick Evaluation' to generate metrics.",
        bg=BG_COLOR_MAIN,
        fg=FG_COLOR_SECONDARY,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        wraplength=1000,
        justify=tk.LEFT
    )
    status_label.pack(fill=tk.X, pady=(0, 15))
    
    # Create scrollable canvas for metrics cards
    canvas_frame = tk.Frame(main_container, bg=BG_COLOR_MAIN)
    canvas_frame.pack(fill=tk.BOTH, expand=True)
    
    # Canvas with scrollbar
    canvas = tk.Canvas(
        canvas_frame,
        bg=BG_COLOR_MAIN,
        highlightthickness=0,
        bd=0
    )
    
    # Scrollbar
    scrollbar = tk.Scrollbar(
        canvas_frame,
        orient=tk.VERTICAL,
        command=canvas.yview,
        bg=BG_COLOR_ENTRY,
        troughcolor=BG_COLOR_MAIN,
        activebackground=BG_COLOR_HOVER
    )
    
    # Scrollable frame inside canvas
    scrollable_frame = tk.Frame(canvas, bg=BG_COLOR_MAIN)
    canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)
    
    # Configure scrollbar
    def configure_scroll_region(event=None):
        canvas.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
    
    def configure_canvas_width(event):
        canvas_width = event.width
        canvas.itemconfig(canvas_window, width=canvas_width)
    
    scrollable_frame.bind("<Configure>", configure_scroll_region)
    canvas.bind("<Configure>", configure_canvas_width)
    canvas.config(yscrollcommand=scrollbar.set)
    
    # Pack canvas and scrollbar
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Mouse wheel scrolling
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    canvas.bind_all("<MouseWheel>", on_mousewheel)
    
    return canvas, buttons_frame, status_label, scrollable_frame