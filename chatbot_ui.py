"""
Chatbot UI Component for Product Recommendation System

This module provides the UI components for the chatbot interface with modern ChatGPT-style design.
"""

import logging
import tkinter as tk
from tkinter import ttk, scrolledtext, Canvas, Frame
from typing import Optional, Callable, List, Dict
from datetime import datetime
import webbrowser
import re

from ui_constants import (
    BG_COLOR_CARD, BG_COLOR_ENTRY, BG_COLOR_HOVER, BORDER_COLOR,
    FG_COLOR_WHITE, FG_COLOR_TEXT, FG_COLOR_SECONDARY,
    FONT_FAMILY, FONT_SIZE_HEADING, FONT_SIZE_NORMAL, FONT_SIZE_BOLD,
    BUTTON_PRIMARY, BUTTON_PRIMARY_HOVER, BUTTON_PRIMARY_ACTIVE
)

logger = logging.getLogger(__name__)

# Modern chat colors
USER_BUBBLE_COLOR = '#97865e'  # User message bubble (brown/tan)
BOT_BUBBLE_COLOR = '#374151'   # Bot message bubble (dark gray)
CHAT_BG_COLOR = '#1a1a2e'      # Chat background

# Suggestion button colors (modern gradient-like)
SUGGESTION_COLORS = {
    'primary': '#4f46e5',      # Indigo
    'primary_hover': '#6366f1',
    'secondary': '#06b6d4',    # Cyan
    'secondary_hover': '#0891b2',
    'accent': '#97865e',        # Brown/tan
    'accent_hover': '#7a6b4a',
    'success': '#10b981',       # Green
    'success_hover': '#059669',
    'warning': '#f59e0b',       # Amber
    'warning_hover': '#d97706',
    'info': '#3b82f6',          # Blue
    'info_hover': '#2563eb'
}


class UserInterestTracker:
    """Tracks user interests and preferences for dynamic suggestions."""
    
    def __init__(self):
        self.product_types_searched: List[str] = []
        self.brands_searched: List[str] = []
        self.budget_preferences: List[str] = []
        self.recent_queries: List[str] = []
        self.max_history = 10
    
    def add_query(self, query: str) -> None:
        """Add a user query to history."""
        self.recent_queries.append(query.lower())
        if len(self.recent_queries) > self.max_history:
            self.recent_queries.pop(0)
        
        # Extract product type
        product_type = self._extract_product_type(query)
        if product_type:
            if product_type not in self.product_types_searched:
                self.product_types_searched.append(product_type)
            if len(self.product_types_searched) > 5:
                self.product_types_searched.pop(0)
        
        # Extract brand
        brand = self._extract_brand(query)
        if brand:
            if brand not in self.brands_searched:
                self.brands_searched.append(brand)
            if len(self.brands_searched) > 5:
                self.brands_searched.pop(0)
        
        # Extract budget
        budget = self._extract_budget(query)
        if budget:
            if budget not in self.budget_preferences:
                self.budget_preferences.append(budget)
            if len(self.budget_preferences) > 3:
                self.budget_preferences.pop(0)
    
    def _extract_product_type(self, query: str) -> Optional[str]:
        """Extract product type from query."""
        query_lower = query.lower()
        product_types = {
            'refrigerator': ['refrigerator', 'fridge'],
            'washing machine': ['washing machine', 'washer'],
            'air conditioner': ['air conditioner', 'ac'],
            'microwave': ['microwave'],
            'oven': ['oven'],
            'dishwasher': ['dishwasher'],
            'dryer': ['dryer'],
            'vacuum': ['vacuum', 'vacuum cleaner'],
            'blender': ['blender'],
            'toaster': ['toaster']
        }
        for product_type, keywords in product_types.items():
            if any(keyword in query_lower for keyword in keywords):
                return product_type
        return None
    
    def _extract_brand(self, query: str) -> Optional[str]:
        """Extract brand from query."""
        # Common brands
        brands = ['samsung', 'lg', 'whirlpool', 'ge', 'frigidaire', 'maytag', 'bosch', 'kitchenaid', 'kenmore', 'haier']
        query_lower = query.lower()
        for brand in brands:
            if brand in query_lower:
                return brand.title()
        return None
    
    def _extract_budget(self, query: str) -> Optional[str]:
        """Extract budget preference from query."""
        query_lower = query.lower()
        budget_keywords = {
            'under $100': ['under 100', 'under $100', 'less than 100'],
            'under $300': ['under 300', 'under $300', 'less than 300'],
            'under $500': ['under 500', 'under $500', 'less than 500'],
            'under $1000': ['under 1000', 'under $1000', 'less than 1000'],
            'under $2000': ['under 2000', 'under $2000', 'less than 2000']
        }
        for budget, keywords in budget_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return budget
        return None
    
    def get_suggestions(self) -> List[tuple]:
        """Generate dynamic suggestions based on user interests."""
        suggestions = []
        
        # If user has searched for specific product types, suggest related ones
        if self.product_types_searched:
            # Suggest similar or related products
            recent_type = self.product_types_searched[-1]
            related_products = {
                'refrigerator': ['washing machine', 'dishwasher', 'oven'],
                'washing machine': ['dryer', 'refrigerator'],
                'air conditioner': ['refrigerator', 'microwave'],
                'microwave': ['oven', 'refrigerator'],
                'oven': ['microwave', 'dishwasher'],
                'dishwasher': ['refrigerator', 'washing machine']
            }
            
            if recent_type in related_products:
                for related in related_products[recent_type][:2]:
                    suggestions.append((f"🔍 {related.title()}", f"find me a {related}"))
        
        # Suggest brands if user has shown interest
        if self.brands_searched:
            brand = self.brands_searched[-1]
            if self.product_types_searched:
                product_type = self.product_types_searched[-1]
                suggestions.append((f"🏷️ {brand} {product_type.title()}", f"find me a {brand} {product_type}"))
        
        # Suggest budget options if user has used budgets
        if self.budget_preferences:
            budget = self.budget_preferences[-1]
            if self.product_types_searched:
                product_type = self.product_types_searched[-1]
                suggestions.append((f"💰 {budget}", f"find me a {product_type} {budget.lower()}"))
        
        # Fill with default suggestions if needed
        default_suggestions = [
            ("🔍 Refrigerator", "find me a refrigerator"),
            ("🔍 Washing Machine", "find me a washing machine"),
            ("🔍 Air Conditioner", "find me an air conditioner"),
            ("⭐ Top Rated", "show me top rated products"),
            ("💰 Under $500", "show me products under $500"),
            ("❓ Help", "help me find products")
        ]
        
        # Add defaults that aren't already in suggestions
        for default in default_suggestions:
            if len(suggestions) >= 6:
                break
            if not any(default[1] == s[1] for s in suggestions):
                suggestions.append(default)
        
        return suggestions[:6]  # Max 6 suggestions


def create_chatbot_panel(parent: tk.Frame, send_message_callback: Callable[[str], str]) -> tuple:
    """
    Create a modern chatbot panel with ChatGPT-style interface.
    
    Args:
        parent: Parent frame to attach chatbot to
        send_message_callback: Callback function that takes user message and returns bot response
        
    Returns:
        Tuple of (chat_container widget, message_entry widget, send_button widget)
    """
    # Chatbot container
    chatbot_card = tk.Frame(parent, bg=BG_COLOR_CARD, relief=tk.FLAT, bd=0)
    chatbot_card.pack(fill=tk.BOTH, expand=True, padx=25, pady=(0, 15))
    
    # Header
    chatbot_header = tk.Frame(chatbot_card, bg=BG_COLOR_CARD)
    chatbot_header.pack(fill=tk.X, padx=20, pady=(15, 10))
    
    tk.Label(
        chatbot_header,
        text="💬 AI Assistant",
        font=(FONT_FAMILY, FONT_SIZE_HEADING, 'bold'),
        fg=FG_COLOR_WHITE,
        bg=BG_COLOR_CARD
    ).pack(side=tk.LEFT)
    
    # Clear button
    clear_button = tk.Button(
        chatbot_header,
        text="Clear Chat",
        command=lambda: clear_chat(chat_container=None),
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        bg=BG_COLOR_ENTRY,
        fg=FG_COLOR_WHITE,
        activebackground=BG_COLOR_HOVER,
        activeforeground=FG_COLOR_WHITE,
        highlightbackground=BG_COLOR_ENTRY,
        highlightcolor=BG_COLOR_ENTRY,
        highlightthickness=0,
        relief=tk.FLAT,
        bd=0,
        cursor='hand2',
        padx=12,
        pady=4
    )
    clear_button.pack(side=tk.RIGHT)
    
    # Chat display area with Canvas for scrolling
    chat_frame = tk.Frame(chatbot_card, bg=CHAT_BG_COLOR)
    chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
    
    # Canvas and scrollbar for chat messages
    chat_canvas = Canvas(
        chat_frame,
        bg=CHAT_BG_COLOR,
        highlightthickness=0,
        bd=0
    )
    
    scrollbar = ttk.Scrollbar(chat_frame, orient=tk.VERTICAL, command=chat_canvas.yview)
    chat_canvas.configure(yscrollcommand=scrollbar.set)
    
    # Scrollable frame inside canvas
    chat_container = tk.Frame(chat_canvas, bg=CHAT_BG_COLOR)
    chat_canvas_window = chat_canvas.create_window((0, 0), window=chat_container, anchor=tk.NW)
    
    # Configure scrolling
    def configure_scroll_region(event=None):
        chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))
    
    def configure_canvas_width(event=None):
        canvas_width = event.width
        chat_canvas.itemconfig(chat_canvas_window, width=canvas_width)
    
    chat_container.bind("<Configure>", configure_scroll_region)
    chat_canvas.bind("<Configure>", configure_canvas_width)
    
    # Mouse wheel scrolling
    def on_mousewheel(event):
        chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    chat_canvas.bind_all("<MouseWheel>", on_mousewheel)
    
    chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Welcome message
    welcome_msg = "👋 Hello! I'm your AI assistant. I can help you find products and answer questions about appliances.\n\nTry asking:\n• 'Help me find a refrigerator'\n• 'What brands are available?'\n• 'Show me products under $500'"
    _add_bot_message(chat_container, welcome_msg, chat_canvas)
    
    # User interest tracker
    interest_tracker = UserInterestTracker()
    
    # Quick Suggestions section with modern styling
    quick_suggestions_frame = tk.Frame(chatbot_card, bg=BG_COLOR_CARD)
    quick_suggestions_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
    
    suggestions_label = tk.Label(
        quick_suggestions_frame,
        text="💡 Quick Suggestions:",
        font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'),
        fg=FG_COLOR_WHITE,
        bg=BG_COLOR_CARD
    )
    suggestions_label.pack(side=tk.LEFT, padx=(0, 12))
    
    suggestions_buttons_frame = tk.Frame(quick_suggestions_frame, bg=BG_COLOR_CARD)
    suggestions_buttons_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    # Get initial suggestions
    quick_suggestions = interest_tracker.get_suggestions()
    
    # Color scheme for suggestions (rotating colors)
    suggestion_color_schemes = [
        (SUGGESTION_COLORS['primary'], SUGGESTION_COLORS['primary_hover']),
        (SUGGESTION_COLORS['secondary'], SUGGESTION_COLORS['secondary_hover']),
        (SUGGESTION_COLORS['accent'], SUGGESTION_COLORS['accent_hover']),
        (SUGGESTION_COLORS['success'], SUGGESTION_COLORS['success_hover']),
        (SUGGESTION_COLORS['warning'], SUGGESTION_COLORS['warning_hover']),
        (SUGGESTION_COLORS['info'], SUGGESTION_COLORS['info_hover'])
    ]
    
    quick_buttons = []
    for i, (label, message) in enumerate(quick_suggestions):
        color, hover_color = suggestion_color_schemes[i % len(suggestion_color_schemes)]
        
        btn = tk.Button(
            suggestions_buttons_frame,
            text=label,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'),
            bg=color,
            fg=FG_COLOR_WHITE,
            activebackground=hover_color,
            activeforeground=FG_COLOR_WHITE,
            highlightbackground=color,
            highlightcolor=color,
            highlightthickness=0,
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            padx=16,
            pady=8,
            borderwidth=0
        )
        
        def make_command(msg, btn_ref=btn):
            def command():
                interest_tracker.add_query(msg)
                _send_quick_suggestion(msg, message_entry, send_message_callback, chat_container, chat_canvas)
                _update_quick_suggestions(suggestions_buttons_frame, quick_buttons, interest_tracker, message_entry, send_message_callback, chat_container, chat_canvas)
            return command
        
        btn.config(command=make_command(message))
        
        def make_hover_effect(button, hover_clr=hover_color, normal_clr=color):
            def on_enter(e):
                button.config(bg=hover_clr, highlightbackground=hover_clr)
            def on_leave(e):
                button.config(bg=normal_clr, highlightbackground=normal_clr)
            button.bind("<Enter>", on_enter)
            button.bind("<Leave>", on_leave)
        
        make_hover_effect(btn)
        btn.pack(side=tk.LEFT, padx=(0, 10))
        quick_buttons.append(btn)
    
    # Store references for updates
    chat_container.interest_tracker = interest_tracker
    chat_container.quick_buttons_frame = suggestions_buttons_frame
    chat_container.quick_buttons = quick_buttons
    
    # Input area
    input_frame = tk.Frame(chatbot_card, bg=BG_COLOR_CARD)
    input_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
    
    message_entry = tk.Entry(
        input_frame,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        bg=BG_COLOR_ENTRY,
        fg=FG_COLOR_WHITE,
        relief=tk.SOLID,
        insertbackground=FG_COLOR_WHITE,
        bd=1,
        highlightthickness=2,
        highlightbackground=BORDER_COLOR,
        highlightcolor=BUTTON_PRIMARY,
    )
    message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), pady=10, ipady=2)
    
    # Send button
    send_button = tk.Button(
        input_frame,
        text="Send",
        command=lambda: send_chat_message(chat_container, message_entry, send_message_callback, chat_canvas),
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
        padx=20,
        pady=10
    )
    
    def on_enter(e):
        send_button.config(bg=BUTTON_PRIMARY_HOVER, highlightbackground=BUTTON_PRIMARY_HOVER)
    def on_leave(e):
        send_button.config(bg=BUTTON_PRIMARY, highlightbackground=BUTTON_PRIMARY)
    send_button.bind("<Enter>", on_enter)
    send_button.bind("<Leave>", on_leave)
    send_button.pack(side=tk.RIGHT)
    
    # Bind Enter key
    message_entry.bind('<Return>', lambda e: send_chat_message(chat_container, message_entry, send_message_callback, chat_canvas))
    
    # Store references
    clear_button.config(command=lambda: clear_chat(chat_container))
    chat_container.chat_canvas = chat_canvas
    
    return chat_container, message_entry, send_button


def _update_quick_suggestions(
    suggestions_frame: tk.Frame,
    current_buttons: List[tk.Button],
    interest_tracker: UserInterestTracker,
    message_entry: tk.Entry,
    send_message_callback: Callable[[str], str],
    chat_container: tk.Frame,
    chat_canvas: Canvas
) -> None:
    """Update quick suggestions based on user interests."""
    # Destroy old buttons
    for btn in current_buttons:
        btn.destroy()
    
    # Get new suggestions
    new_suggestions = interest_tracker.get_suggestions()
    
    # Color scheme for suggestions
    suggestion_color_schemes = [
        (SUGGESTION_COLORS['primary'], SUGGESTION_COLORS['primary_hover']),
        (SUGGESTION_COLORS['secondary'], SUGGESTION_COLORS['secondary_hover']),
        (SUGGESTION_COLORS['accent'], SUGGESTION_COLORS['accent_hover']),
        (SUGGESTION_COLORS['success'], SUGGESTION_COLORS['success_hover']),
        (SUGGESTION_COLORS['warning'], SUGGESTION_COLORS['warning_hover']),
        (SUGGESTION_COLORS['info'], SUGGESTION_COLORS['info_hover'])
    ]
    
    # Create new buttons
    new_buttons = []
    for i, (label, message) in enumerate(new_suggestions):
        color, hover_color = suggestion_color_schemes[i % len(suggestion_color_schemes)]
        
        btn = tk.Button(
            suggestions_frame,
            text=label,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'),
            bg=color,
            fg=FG_COLOR_WHITE,
            activebackground=hover_color,
            activeforeground=FG_COLOR_WHITE,
            highlightbackground=color,
            highlightcolor=color,
            highlightthickness=0,
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            padx=16,
            pady=8,
            borderwidth=0
        )
        
        def make_command(msg, btn_ref=btn):
            def command():
                interest_tracker.add_query(msg)
                _send_quick_suggestion(msg, message_entry, send_message_callback, chat_container, chat_canvas)
                _update_quick_suggestions(suggestions_frame, new_buttons, interest_tracker, message_entry, send_message_callback, chat_container, chat_canvas)
            return command
        
        btn.config(command=make_command(message))
        
        def make_hover_effect(button, hover_clr=hover_color, normal_clr=color):
            def on_enter(e):
                button.config(bg=hover_clr, highlightbackground=hover_clr)
            def on_leave(e):
                button.config(bg=normal_clr, highlightbackground=normal_clr)
            button.bind("<Enter>", on_enter)
            button.bind("<Leave>", on_leave)
        
        make_hover_effect(btn)
        btn.pack(side=tk.LEFT, padx=(0, 10))
        new_buttons.append(btn)
    
    # Update reference
    chat_container.quick_buttons = new_buttons


def _create_message_bubble(parent: tk.Frame, message: str, is_user: bool, timestamp: str = "") -> tk.Frame:
    """
    Create a modern message bubble.
    
    Args:
        parent: Parent frame
        message: Message text
        is_user: True if user message, False if bot message
        timestamp: Optional timestamp
        
    Returns:
        Frame containing the message bubble
    """
    # Container for the entire message row
    message_row = tk.Frame(parent, bg=CHAT_BG_COLOR)
    
    if is_user:
        # User message - right side
        bubble_container = tk.Frame(message_row, bg=CHAT_BG_COLOR)
        bubble_container.pack(side=tk.RIGHT, anchor=tk.E, padx=(100, 20), pady=8)
        
        # Message bubble frame
        bubble = tk.Frame(
            bubble_container,
            bg=USER_BUBBLE_COLOR,
            relief=tk.FLAT,
            bd=0
        )
        bubble.pack()
        
        # Message text - use Label for simple text (no scrolling)
        # For multi-line, we'll use Text but disable scrolling
        if '\n' in message or len(message) > 50:
            text_widget = tk.Text(
                bubble,
                wrap=tk.WORD,
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                bg=USER_BUBBLE_COLOR,
                fg=FG_COLOR_WHITE,
                relief=tk.FLAT,
                bd=0,
                padx=14,
                pady=10,
                state=tk.DISABLED,
                width=40,
                height=1,
                highlightthickness=0,
                insertwidth=0
            )
            text_widget.pack()
            
            # Insert message
            text_widget.config(state=tk.NORMAL)
            text_widget.insert(tk.END, message)
            if timestamp:
                text_widget.insert(tk.END, f"\n{timestamp}", "timestamp")
            text_widget.config(state=tk.DISABLED)
            
            # Configure tags
            text_widget.tag_config("timestamp", foreground=FG_COLOR_SECONDARY, font=(FONT_FAMILY, 9))
            
            # Auto-resize height - no scrolling, just expand
            text_widget.update_idletasks()
            lines = int(text_widget.index('end-1c').split('.')[0])
            text_widget.config(height=lines)
        else:
            # Simple label for short messages
            text_label = tk.Label(
                bubble,
                text=message + (f"\n{timestamp}" if timestamp else ""),
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                bg=USER_BUBBLE_COLOR,
                fg=FG_COLOR_WHITE,
                justify=tk.LEFT,
                anchor=tk.W,
                padx=14,
                pady=10,
                wraplength=400
            )
            text_label.pack()
        
    else:
        # Bot message - left side
        bubble_container = tk.Frame(message_row, bg=CHAT_BG_COLOR)
        bubble_container.pack(side=tk.LEFT, anchor=tk.W, padx=(20, 100), pady=8)
        
        # Avatar
        avatar = tk.Label(
            bubble_container,
            text="🤖",
            font=(FONT_FAMILY, 18),
            bg=CHAT_BG_COLOR,
            fg=FG_COLOR_WHITE
        )
        avatar.pack(side=tk.LEFT, padx=(0, 10))
        
        # Message bubble frame
        bubble = tk.Frame(
            bubble_container,
            bg=BOT_BUBBLE_COLOR,
            relief=tk.FLAT,
            bd=0
        )
        bubble.pack(side=tk.LEFT)
        
        # Message text - always use Text for bot messages (for formatting and links)
        # But disable scrolling - make it expand to fit content
        text_widget = tk.Text(
            bubble,
            wrap=tk.WORD,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL),
            bg=BOT_BUBBLE_COLOR,
            fg=FG_COLOR_WHITE,
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=10,
            state=tk.DISABLED,
            width=50,
            height=1,
            highlightthickness=0,
            insertwidth=0,
            exportselection=False
        )
        text_widget.pack()
        
        # Insert message
        text_widget.config(state=tk.NORMAL)
        text_widget.insert(tk.END, message)
        if timestamp:
            text_widget.insert(tk.END, f"\n{timestamp}", "timestamp")
        text_widget.config(state=tk.DISABLED)
        
        # Configure tags
        text_widget.tag_config("timestamp", foreground=FG_COLOR_SECONDARY, font=(FONT_FAMILY, 9))
        text_widget.tag_config("product_name", foreground=FG_COLOR_WHITE, font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'))
        text_widget.tag_config("product_detail", foreground="#a78bfa", font=(FONT_FAMILY, FONT_SIZE_NORMAL))
        text_widget.tag_config("product_link", foreground="#3b82f6", underline=True, font=(FONT_FAMILY, FONT_SIZE_NORMAL))
        
        # Bind link clicks
        text_widget.tag_bind("product_link", "<Button-1>", lambda e: _open_url_from_text(e, text_widget))
        text_widget.tag_bind("product_link", "<Enter>", lambda e: text_widget.config(cursor="hand2"))
        text_widget.tag_bind("product_link", "<Leave>", lambda e: text_widget.config(cursor=""))
        
        # Auto-resize height - expand to fit content, no scrolling
        text_widget.update_idletasks()
        lines = int(text_widget.index('end-1c').split('.')[0])
        text_widget.config(height=lines)
    
    return message_row


def _add_user_message(chat_container: tk.Frame, message: str, chat_canvas: Canvas) -> None:
    """Add a user message bubble to the chat."""
    timestamp = datetime.now().strftime("%H:%M")
    bubble = _create_message_bubble(chat_container, message, is_user=True, timestamp=timestamp)
    bubble.pack(fill=tk.X, pady=4)
    chat_container.update_idletasks()
    chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))
    chat_canvas.yview_moveto(1.0)


def _add_bot_message(chat_container: tk.Frame, message: str, chat_canvas: Canvas, formatted: bool = False) -> None:
    """Add a bot message bubble to the chat."""
    bubble = _create_message_bubble(chat_container, message, is_user=False)
    
    # Format message if needed (for products)
    if formatted:
        # Find the text widget in the bubble
        for widget in bubble.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Frame):  # bubble frame
                        for text_widget in child.winfo_children():
                            if isinstance(text_widget, tk.Text):
                                text_widget.config(state=tk.NORMAL)
                                text_widget.delete(1.0, tk.END)
                                _insert_formatted_text(text_widget, message)
                                text_widget.config(state=tk.DISABLED)
                                
                                # Update height
                                text_widget.update_idletasks()
                                lines = int(text_widget.index('end-1c').split('.')[0])
                                text_widget.config(height=lines)
                                break
    
    bubble.pack(fill=tk.X, pady=4)
    chat_container.update_idletasks()
    chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))
    chat_canvas.yview_moveto(1.0)


def _insert_formatted_text(text_widget: tk.Text, text: str) -> None:
    """Insert formatted text with product styling."""
    lines = text.split('\n')
    for line in lines:
        if line.strip().startswith('📦') or (line.strip() and line.strip()[0].isdigit() and '.' in line[:3]):
            # Product name
            product_line = line.strip()
            if '📦' in product_line:
                product_line = product_line.split('📦', 1)[1].strip()
            if product_line and product_line[0].isdigit():
                parts = product_line.split('.', 1)
                if len(parts) > 1:
                    product_name = parts[1].strip()
                else:
                    product_name = product_line
            else:
                product_name = product_line
            text_widget.insert(tk.END, f"{product_name}\n", "product_name")
        elif '🔗' in line and ('http://' in line or 'https://' in line):
            # Link
            url_match = re.search(r'https?://[^\s]+', line)
            if url_match:
                url = url_match.group(0)
                text_widget.insert(tk.END, f"   🔗 ", "product_detail")
                text_widget.insert(tk.END, f"{url}\n", "product_link")
        elif line.strip().startswith('   ') and (line.strip().startswith('Brand:') or 
                                                 line.strip().startswith('Price:') or 
                                                 line.strip().startswith('Rating:')):
            text_widget.insert(tk.END, f"{line}\n", "product_detail")
        elif line.strip().startswith('='):
            text_widget.insert(tk.END, f"{line}\n")
        else:
            text_widget.insert(tk.END, f"{line}\n")


def _open_url_from_text(event, text_widget: tk.Text) -> None:
    """Open URL when clicked in text widget."""
    try:
        index = text_widget.index(f"@{event.x},{event.y}")
        tags = text_widget.tag_names(index)
        if "product_link" in tags:
            ranges = text_widget.tag_ranges("product_link")
            for i in range(0, len(ranges), 2):
                start = ranges[i]
                end = ranges[i + 1]
                if text_widget.compare(start, "<=", index) and text_widget.compare(index, "<=", end):
                    url_text = text_widget.get(start, end)
                    url_match = re.search(r'https?://[^\s]+', url_text)
                    if url_match:
                        webbrowser.open(url_match.group(0))
                        logger.info(f"Opening URL: {url_match.group(0)}")
                        return
    except Exception as e:
        logger.error(f"Error opening URL: {e}")


def send_chat_message(
    chat_container: tk.Frame,
    message_entry: tk.Entry,
    send_message_callback: Callable[[str], str],
    chat_canvas: Canvas
) -> None:
    """Send a chat message and display response."""
    user_message = message_entry.get().strip()
    if not user_message:
        return
    
    message_entry.delete(0, tk.END)
    
    # Track user interest
    if hasattr(chat_container, 'interest_tracker'):
        chat_container.interest_tracker.add_query(user_message)
        # Update suggestions after a short delay
        chat_container.after(500, lambda: _update_quick_suggestions(
            chat_container.quick_buttons_frame,
            chat_container.quick_buttons,
            chat_container.interest_tracker,
            message_entry,
            send_message_callback,
            chat_container,
            chat_canvas
        ))
    
    # Add user message
    _add_user_message(chat_container, user_message, chat_canvas)
    
    # Show typing indicator
    typing_bubble = _create_message_bubble(chat_container, "Thinking...", is_user=False)
    typing_bubble.pack(fill=tk.X, pady=4)
    chat_container.update_idletasks()
    chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))
    chat_canvas.yview_moveto(1.0)
    
    try:
        # Get bot response
        bot_response = send_message_callback(user_message)
        
        # Remove typing indicator (last message)
        if chat_container.winfo_children():
            last_widget = chat_container.winfo_children()[-1]
            last_widget.destroy()
        
        # Add bot response
        _add_bot_message(chat_container, bot_response, chat_canvas, formatted=True)
        
    except Exception as e:
        logger.error(f"Error getting chatbot response: {e}", exc_info=True)
        # Remove typing indicator
        if chat_container.winfo_children():
            last_widget = chat_container.winfo_children()[-1]
            last_widget.destroy()
        _add_bot_message(chat_container, "Sorry, I encountered an error. Please try again.", chat_canvas)


def _send_quick_suggestion(
    message: str,
    message_entry: tk.Entry,
    send_message_callback: Callable[[str], str],
    chat_container: tk.Frame,
    chat_canvas: Canvas
) -> None:
    """Send a quick suggestion message."""
    message_entry.delete(0, tk.END)
    message_entry.insert(0, message)
    send_chat_message(chat_container, message_entry, send_message_callback, chat_canvas)


def clear_chat(chat_container: Optional[tk.Frame]) -> None:
    """Clear the chat display."""
    if chat_container:
        # Clear all messages
        for widget in chat_container.winfo_children():
            widget.destroy()
        
        # Add welcome message
        welcome_msg = "👋 Hello! I'm your AI assistant. I can help you find products and answer questions about appliances.\n\nTry asking:\n• 'Help me find a refrigerator'\n• 'What brands are available?'\n• 'Show me products under $500'"
        _add_bot_message(chat_container, welcome_msg, chat_container.chat_canvas)
        logger.info("Chat cleared")
