"""
UI Styles Configuration for Product Recommendation System GUI

This module handles styling configuration for Tkinter ttk widgets.
"""

import tkinter as tk
from tkinter import ttk
from ui_constants import (
    BG_COLOR_ENTRY, BG_COLOR_INPUT, BG_COLOR_DARK, BORDER_COLOR,
    FG_COLOR_WHITE, FG_COLOR_TEXT, ACCENT_COLOR,
    FONT_FAMILY, FONT_SIZE_NORMAL, FONT_SIZE_BOLD
)


def apply_styles() -> None:
    """
    Apply custom styles to UI components with modern design.
    
    This function configures ttk.Style for Treeview, Combobox, and Scrollbar widgets.
    """
    style = ttk.Style()
    style.theme_use("clam")
    
    # Modern table styling
    style.configure(
        "Treeview",
        background=BG_COLOR_ENTRY,
        foreground=FG_COLOR_WHITE,
        fieldbackground=BG_COLOR_ENTRY,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        rowheight=25,
        borderwidth=0
    )
    style.configure(
        "Treeview.Heading", 
        font=(FONT_FAMILY, FONT_SIZE_BOLD, 'bold'), 
        background=BG_COLOR_INPUT, 
        foreground=FG_COLOR_WHITE,
        relief=tk.FLAT,
        borderwidth=0
    )
    style.map(
        "Treeview",
        background=[("selected", ACCENT_COLOR)],
        foreground=[("selected", FG_COLOR_WHITE)]
    )
    
    # Modern combobox styling
    style.configure(
        "TCombobox",
        fieldbackground=BG_COLOR_ENTRY,
        background=BG_COLOR_ENTRY,
        foreground=FG_COLOR_WHITE,
        borderwidth=1,
        relief=tk.SOLID,
        padding=(5, 6)  # (horizontal, vertical) padding for consistent height
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", BG_COLOR_ENTRY)],
        background=[("readonly", BG_COLOR_ENTRY)],
        arrowcolor=[("readonly", FG_COLOR_TEXT)],
        bordercolor=[("readonly", BORDER_COLOR), ("focus", ACCENT_COLOR)],
        lightcolor=[("readonly", BORDER_COLOR)],
        darkcolor=[("readonly", BORDER_COLOR)]
    )
    
    # Scrollbar styling
    style.configure(
        "Vertical.TScrollbar",
        background=BG_COLOR_INPUT,
        troughcolor=BG_COLOR_DARK,
        borderwidth=0,
        arrowcolor=FG_COLOR_TEXT,
        darkcolor=BG_COLOR_INPUT,
        lightcolor=BG_COLOR_INPUT
    )

