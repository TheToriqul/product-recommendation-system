"""
UI Constants for Product Recommendation System GUI

This module contains all UI-related constants including colors, fonts, sizes, and options.
"""

# Window Configuration
WINDOW_TITLE = "AI-Powered Product Recommendation System"
WINDOW_SIZE = "1280x720"  # Optimized to show all UI elements and table rows within this size

# Color Scheme - Modern Dark Theme
BG_COLOR_MAIN = '#1a1a2e'  # Modern dark blue-gray
BG_COLOR_DARK = '#0f0f1e'  # Deeper dark
BG_COLOR_INPUT = '#16213e'  # Card background
BG_COLOR_ENTRY = '#505b6c'  # Input field / Box color
BG_COLOR_CARD = '#1f2937'  # Card background
BG_COLOR_HOVER = '#2a3a4e'  # Hover state
BORDER_COLOR = '#374151'  # Border color for input fields
FG_COLOR_WHITE = '#ffffff'
FG_COLOR_TEXT = '#e5e7eb'  # Soft white
FG_COLOR_SECONDARY = '#9ca3af'  # Secondary text
ACCENT_COLOR = '#3b82f6'  # Modern blue
SUCCESS_COLOR = '#10b981'

# Button Colors - Custom Brown/Tan Theme
BUTTON_PRIMARY = '#97865e'  # Custom brown/tan button color
BUTTON_PRIMARY_HOVER = '#7a6b4a'  # Darker brown on hover
BUTTON_PRIMARY_ACTIVE = '#5d5137'  # Even darker when clicked
BUTTON_SECONDARY = '#06b6d4'  # Vibrant cyan/teal
BUTTON_SECONDARY_HOVER = '#0891b2'  # Darker cyan on hover
BUTTON_SECONDARY_ACTIVE = '#0e7490'  # Even darker when clicked

# Typography
FONT_FAMILY = 'Segoe UI'
FONT_SIZE_TITLE = 24
FONT_SIZE_HEADING = 16
FONT_SIZE_NORMAL = 11
FONT_SIZE_BOLD = 12

# Budget Options
BUDGET_OPTIONS = ["No Limit", "Under $100", "Under $300", "Under $500", "Under $1000", "Under $2000"]
BUDGET_MAP = {
    "Under $100": 100,
    "Under $300": 300,
    "Under $500": 500,
    "Under $1000": 1000,
    "Under $2000": 2000
}

# Sort Options
SORT_OPTIONS = ["Similarity (Default)", "Price: Low to High", "Price: High to Low", "Rating: Best First"]

