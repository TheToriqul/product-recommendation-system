"""
Chatbot UI Component for Product Recommendation System

This module provides the UI components for the chatbot interface.
"""

import logging
import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Optional, Callable
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


def create_chatbot_panel(parent: tk.Frame, send_message_callback: Callable[[str], str]) -> tuple:
    """
    Create a chatbot panel with chat interface.
    
    Args:
        parent: Parent frame to attach chatbot to
        send_message_callback: Callback function that takes user message and returns bot response
        
    Returns:
        Tuple of (chat_display widget, message_entry widget, send_button widget)
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
        command=lambda: clear_chat(chat_display=None),
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
    
    # Chat display area
    chat_frame = tk.Frame(chatbot_card, bg=BG_COLOR_CARD)
    chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
    
    chat_display = scrolledtext.ScrolledText(
        chat_frame,
        wrap=tk.WORD,
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        bg=BG_COLOR_ENTRY,
        fg=FG_COLOR_WHITE,
        insertbackground=FG_COLOR_WHITE,
        relief=tk.SOLID,
        bd=1,
        highlightthickness=1,
        highlightbackground=BORDER_COLOR,
        highlightcolor=BORDER_COLOR,
        state=tk.DISABLED,
        padx=15,
        pady=10,
        tabs=('0.5c', '40c', '80c')  # Tab stops for alignment
    )
    chat_display.pack(fill=tk.BOTH, expand=True)
    
    # Welcome message (left-aligned, bot message)
    welcome_msg = "👋 Hello! I'm your AI assistant. I can help you find products and answer questions about appliances.\n\nTry asking:\n• 'Help me find a refrigerator'\n• 'What brands are available?'\n• 'Show me products under $500'\n\n"
    chat_display.config(state=tk.NORMAL)
    chat_display.insert(tk.END, "🤖 Assistant: ", "bot_label")
    chat_display.insert(tk.END, welcome_msg, "bot")
    chat_display.config(state=tk.DISABLED)
    
    # Configure text tags for styling
    # User messages (right-aligned) - blue color
    chat_display.tag_config("user", foreground="#60a5fa", font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'))
    chat_display.tag_config("user_msg", foreground="#60a5fa", font=(FONT_FAMILY, FONT_SIZE_NORMAL))
    chat_display.tag_config("user_timestamp", foreground=FG_COLOR_SECONDARY, font=(FONT_FAMILY, 9))
    
    # Bot messages (left-aligned) - white/purple color
    chat_display.tag_config("bot", foreground=FG_COLOR_WHITE)
    chat_display.tag_config("bot_label", foreground="#a78bfa", font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'))
    chat_display.tag_config("timestamp", foreground=FG_COLOR_SECONDARY, font=(FONT_FAMILY, 9))
    
    # Product formatting - bold black for titles
    chat_display.tag_config("product_name", foreground="#000000", font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'))
    chat_display.tag_config("product_detail", foreground="#a78bfa", font=(FONT_FAMILY, FONT_SIZE_NORMAL))
    
    # Link formatting - blue, underlined, clickable
    chat_display.tag_config("product_link", foreground="#3b82f6", underline=True, font=(FONT_FAMILY, FONT_SIZE_NORMAL))
    chat_display.tag_config("product_link_hover", foreground="#2563eb", underline=True, font=(FONT_FAMILY, FONT_SIZE_NORMAL))
    
    # Bind click event for links
    chat_display.tag_bind("product_link", "<Button-1>", lambda e: _open_url_from_event(e, chat_display))
    chat_display.tag_bind("product_link", "<Enter>", lambda e: _on_link_enter(e, chat_display))
    chat_display.tag_bind("product_link", "<Leave>", lambda e: _on_link_leave(e, chat_display))
    
    # Input area (create first so we can reference it in quick suggestions)
    input_frame = tk.Frame(chatbot_card, bg=BG_COLOR_CARD)
    input_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
    
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
    
    # Quick Suggestions section (placed above input)
    quick_suggestions_frame = tk.Frame(chatbot_card, bg=BG_COLOR_CARD)
    quick_suggestions_frame.pack(fill=tk.X, padx=20, pady=(0, 10), before=input_frame)
    
    # Quick Suggestions label
    suggestions_label = tk.Label(
        quick_suggestions_frame,
        text="Quick Suggestions:",
        font=(FONT_FAMILY, FONT_SIZE_NORMAL),
        fg=FG_COLOR_SECONDARY,
        bg=BG_COLOR_CARD
    )
    suggestions_label.pack(side=tk.LEFT, padx=(0, 10))
    
    # Quick suggestions buttons container
    suggestions_buttons_frame = tk.Frame(quick_suggestions_frame, bg=BG_COLOR_CARD)
    suggestions_buttons_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    # Quick suggestion messages
    quick_suggestions = [
        ("🔍 Find Refrigerator", "find me a refrigerator"),
        ("🔍 Find Washing Machine", "find me a washing machine"),
        ("🔍 Find Air Conditioner", "find me an air conditioner"),
        ("⭐ Top Rated", "show me top rated products"),
        ("💰 Under $500", "show me products under $500"),
        ("❓ Help", "help me find products")
    ]
    
    quick_buttons = []
    for i, (label, message) in enumerate(quick_suggestions):
        btn = tk.Button(
            suggestions_buttons_frame,
            text=label,
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
            pady=6
        )
        
        # Bind click event to send message (using closure to capture message)
        def make_command(msg):
            return lambda: _send_quick_suggestion(msg, message_entry, send_message_callback, chat_display)
        
        btn.config(command=make_command(message))
        
        # Hover effects
        def make_hover_effect(button):
            def on_enter(e):
                button.config(bg=BG_COLOR_HOVER)
            def on_leave(e):
                button.config(bg=BG_COLOR_ENTRY)
            button.bind("<Enter>", on_enter)
            button.bind("<Leave>", on_leave)
        
        make_hover_effect(btn)
        btn.pack(side=tk.LEFT, padx=(0, 8))
        quick_buttons.append(btn)
    
    # Send button
    send_button = tk.Button(
        input_frame,
        text="Send",
        command=lambda: send_chat_message(chat_display, message_entry, send_message_callback),
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
    
    # Hover effects
    def on_enter(e):
        send_button.config(bg=BUTTON_PRIMARY_HOVER, highlightbackground=BUTTON_PRIMARY_HOVER)
    def on_leave(e):
        send_button.config(bg=BUTTON_PRIMARY, highlightbackground=BUTTON_PRIMARY)
    send_button.bind("<Enter>", on_enter)
    send_button.bind("<Leave>", on_leave)
    send_button.pack(side=tk.RIGHT)
    
    # Bind Enter key to send message
    message_entry.bind('<Return>', lambda e: send_chat_message(chat_display, message_entry, send_message_callback))
    
    # Store reference for clear button
    clear_button.config(command=lambda: clear_chat(chat_display))
    
    # Store quick buttons reference for potential future use
    chat_display.quick_buttons = quick_buttons
    
    return chat_display, message_entry, send_button


def send_chat_message(
    chat_display: scrolledtext.ScrolledText,
    message_entry: tk.Entry,
    send_message_callback: Callable[[str], str]
) -> None:
    """
    Send a chat message and display response.
    
    Args:
        chat_display: ScrolledText widget for displaying messages
        message_entry: Entry widget for user input
        send_message_callback: Callback function to get bot response
    """
    user_message = message_entry.get().strip()
    if not user_message:
        return
    
    # Clear input
    message_entry.delete(0, tk.END)
    
    # Get current time
    timestamp = datetime.now().strftime("%H:%M")
    
    # Display user message (right-aligned using tabs)
    chat_display.config(state=tk.NORMAL)
    # Use tab to push message to the right, then display message
    chat_display.insert(tk.END, "\t\t", "user")  # Tab to right side
    chat_display.insert(tk.END, f"{user_message} ", "user_msg")
    chat_display.insert(tk.END, f"[{timestamp}]\n", "user_timestamp")
    chat_display.insert(tk.END, "\n", "bot")  # Add spacing
    chat_display.config(state=tk.DISABLED)
    chat_display.see(tk.END)
    chat_display.update()
    
    # Show typing indicator (left-aligned)
    chat_display.config(state=tk.NORMAL)
    chat_display.insert(tk.END, "🤖 Assistant: ", "bot_label")
    chat_display.insert(tk.END, "Thinking...\n\n", "bot")
    chat_display.config(state=tk.DISABLED)
    chat_display.see(tk.END)
    chat_display.update()
    
    try:
        # Get bot response
        bot_response = send_message_callback(user_message)
        
        # Remove "Thinking..." and add actual response
        chat_display.config(state=tk.NORMAL)
        # Remove "Thinking..." line
        chat_display.delete("end-2l", "end-1l")
        
        # Format response with special handling for product recommendations (left-aligned)
        _insert_formatted_response(chat_display, bot_response)
        
        chat_display.insert(tk.END, "\n", "bot")  # Add spacing after bot message
        chat_display.config(state=tk.DISABLED)
        chat_display.see(tk.END)
    except Exception as e:
        logger.error(f"Error getting chatbot response: {e}", exc_info=True)
        chat_display.config(state=tk.NORMAL)
        # Remove "Thinking..." and add error message
        chat_display.delete("end-2l", "end-1l")
        chat_display.insert(tk.END, "Sorry, I encountered an error. Please try again.\n\n", "bot")
        chat_display.config(state=tk.DISABLED)
        chat_display.see(tk.END)


def _send_quick_suggestion(
    message: str,
    message_entry: tk.Entry,
    send_message_callback: Callable[[str], str],
    chat_display: scrolledtext.ScrolledText
) -> None:
    """
    Send a quick suggestion message.
    
    Args:
        message: The message to send
        message_entry: Entry widget (to clear it)
        send_message_callback: Callback to send message
        chat_display: Chat display widget
    """
    # Set the message in the entry
    message_entry.delete(0, tk.END)
    message_entry.insert(0, message)
    
    # Send the message
    send_chat_message(chat_display, message_entry, send_message_callback)


def _open_url_from_event(event, chat_display: scrolledtext.ScrolledText) -> None:
    """
    Open URL when link is clicked.
    
    Args:
        event: Click event
        chat_display: ScrolledText widget
    """
    try:
        # Get the character position of the click
        index = chat_display.index(f"@{event.x},{event.y}")
        
        # Find all link tags at this position
        tags = chat_display.tag_names(index)
        if "product_link" in tags:
            # Get the text range of the link
            ranges = chat_display.tag_ranges("product_link")
            for i in range(0, len(ranges), 2):
                start = ranges[i]
                end = ranges[i + 1]
                # Check if click is within this range
                if chat_display.compare(start, "<=", index) and chat_display.compare(index, "<=", end):
                    # Extract URL from the text
                    url_text = chat_display.get(start, end)
                    # Extract URL using regex
                    url_match = re.search(r'https?://[^\s]+', url_text)
                    if url_match:
                        url = url_match.group(0)
                        webbrowser.open(url)
                        logger.info(f"Opening URL: {url}")
                        return
    except Exception as e:
        logger.error(f"Error opening URL: {e}")


def _on_link_enter(event, chat_display: scrolledtext.ScrolledText) -> None:
    """Change cursor to hand when hovering over link."""
    chat_display.config(cursor="hand2")


def _on_link_leave(event, chat_display: scrolledtext.ScrolledText) -> None:
    """Change cursor back to normal when leaving link."""
    chat_display.config(cursor="")


def _insert_formatted_response(chat_display: scrolledtext.ScrolledText, response: str) -> None:
    """
    Insert a formatted response into the chat display with special formatting for products.
    Product titles are bold black, links are clickable.
    
    Args:
        chat_display: ScrolledText widget
        response: Response text to insert
    """
    lines = response.split('\n')
    for line in lines:
        # Check if line contains product information
        if line.strip().startswith('📦') or (line.strip() and line.strip()[0].isdigit() and '.' in line[:3]):
            # Product name line - extract just the name (remove emoji and number)
            # Format: "📦 1. Product Name" -> "Product Name" in bold black
            product_line = line.strip()
            # Remove emoji and number prefix
            if '📦' in product_line:
                product_line = product_line.split('📦', 1)[1].strip()
            if product_line and product_line[0].isdigit():
                # Remove number and dot (e.g., "1. ")
                parts = product_line.split('.', 1)
                if len(parts) > 1:
                    product_name = parts[1].strip()
                else:
                    product_name = product_line
            else:
                product_name = product_line
            
            # Insert product name in bold black
            chat_display.insert(tk.END, f"{product_name}\n", "product_name")
        elif '🔗' in line and ('http://' in line or 'https://' in line):
            # Product link line - make it clickable
            link_text = line.strip()
            # Extract URL
            url_match = re.search(r'https?://[^\s]+', link_text)
            if url_match:
                url = url_match.group(0)
                # Insert link text with clickable tag
                chat_display.insert(tk.END, f"   🔗 ", "product_detail")
                start_pos = chat_display.index(tk.END + "-1c")
                chat_display.insert(tk.END, f"{url}\n", "product_link")
            else:
                chat_display.insert(tk.END, f"{line}\n", "product_detail")
        elif line.strip().startswith('   ') and (line.strip().startswith('Brand:') or 
                                                   line.strip().startswith('Price:') or 
                                                   line.strip().startswith('Rating:')):
            # Product detail line
            chat_display.insert(tk.END, f"{line}\n", "product_detail")
        elif line.strip().startswith('='):
            # Separator line
            chat_display.insert(tk.END, f"{line}\n", "bot")
        else:
            # Regular text
            chat_display.insert(tk.END, f"{line}\n", "bot")
    chat_display.insert(tk.END, "\n", "bot")


def clear_chat(chat_display: Optional[scrolledtext.ScrolledText]) -> None:
    """
    Clear the chat display.
    
    Args:
        chat_display: ScrolledText widget to clear
    """
    if chat_display:
        chat_display.config(state=tk.NORMAL)
        chat_display.delete(1.0, tk.END)
        welcome_msg = "👋 Hello! I'm your AI assistant. I can help you find products and answer questions about appliances.\n\nTry asking:\n• 'Help me find a refrigerator'\n• 'What brands are available?'\n• 'Show me products under $500'\n\n"
        # Bot message (left-aligned, no tab)
        chat_display.insert(tk.END, "🤖 Assistant: ", "bot_label")
        chat_display.insert(tk.END, welcome_msg, "bot")
        chat_display.config(state=tk.DISABLED)
        logger.info("Chat cleared")

