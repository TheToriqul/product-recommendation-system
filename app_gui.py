"""
Product Recommendation System GUI

A desktop application for recommending electrical appliances using a content-based
recommendation engine with a modern dark-themed interface.
"""

import logging
import tkinter as tk
from tkinter import messagebox
from typing import List, Dict
import os
import threading

from recommender_engine import ProductRecommender, GENAI_AVAILABLE

# Import UI modules
from ui_constants import (
    WINDOW_TITLE, WINDOW_SIZE, BG_COLOR_MAIN, BG_COLOR_DARK,
    BG_COLOR_INPUT, BG_COLOR_CARD,
    FG_COLOR_WHITE, FG_COLOR_SECONDARY, SUCCESS_COLOR,
    FONT_FAMILY, FONT_SIZE_NORMAL,
    BUTTON_PRIMARY
)
from ui_styles import apply_styles
from ui_components import (
    create_header, create_input_section, create_results_section,
    create_similar_section
)
from ui_handlers import (
    handle_item_click, handle_item_double_click, handle_similar_double_click,
    show_similar_products, perform_search
)

# Configure logging FIRST (before any logger usage)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import chatbot modules
try:
    from chatbot import ProductChatbot
    from chatbot_ui import create_chatbot_panel
    CHATBOT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Chatbot modules not available: {e}")
    CHATBOT_AVAILABLE = False

# Check if NLTK is available for advanced features
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class ProdRecommendationApp:
    """
    Main application class for the Product Recommendation System GUI.
    
    This class creates and manages the Tkinter-based user interface for searching
    and displaying product recommendations.
    """
    
    def __init__(self, root: tk.Tk, use_advanced: bool = True) -> None:
        """
        Initialize the application.
        
        Args:
            root: Tkinter root window
            use_advanced: Whether to use advanced recommender (if available)
        """
        self.root = root
        self.bgcolor = BG_COLOR_MAIN
        # Enable advanced mode if NLTK or GenAI is available
        self.use_advanced = use_advanced and (NLTK_AVAILABLE or GENAI_AVAILABLE)
        
        # Initialize recommender with error handling (load dataset only, models in background)
        try:
            if self.use_advanced:
                logger.info("Initializing product recommender (dataset only, models will load in background)...")
                self.engine = ProductRecommender(
                    use_ngrams=True,
                    use_advanced_preprocessing=True,
                    use_genai=True,  # Enable Generative AI
                    feature_weights={'product_name': 0.7, 'brand': 0.3},
                    load_models_immediately=False  # Defer model loading to background
                )
                logger.info("Dataset loaded - models will load in background")
            else:
                self.engine = ProductRecommender(load_models_immediately=False)
                logger.info("Dataset loaded - models will load in background")
        except FileNotFoundError as e:
            logger.error(f"Failed to initialize recommender: {e}")
            messagebox.showerror(
                "Error", 
                f"Failed to load dataset:\n{e}\n\nPlease ensure the CSV file exists."
            )
            root.destroy()
            return
        except Exception as e:
            logger.error(f"Unexpected error initializing recommender: {e}", exc_info=True)
            # Fallback to basic if advanced fails
            if self.use_advanced:
                logger.info("Falling back to basic recommender...")
                try:
                    self.engine = ProductRecommender()
                    self.use_advanced = False
                    logger.info("Basic recommender initialized as fallback")
                except Exception as e2:
                    messagebox.showerror(
                        "Error", 
                        f"Failed to initialize recommender:\n{e2}\n\nThe application will close."
                    )
                    root.destroy()
                    return
            else:
                messagebox.showerror(
                    "Error", 
                    f"An unexpected error occurred:\n{e}\n\nThe application will close."
                )
                root.destroy()
                return
        
        # Initialize chatbot lazily (only when needed) to speed up startup
        # Chatbot will be initialized when user switches to chat tab
        self.chatbot = None
        self._chatbot_initialized = False
        
        # Track model loading status
        self._models_loading = False
        self._models_loaded = False
        
        self._setup_window()
        self._create_ui()
        
        # Bind Enter key to search
        self.product_entry.bind('<Return>', lambda e: self.send_message())
        logger.info("Application initialized successfully")
        
        # Start background loading of models/embeddings
        self._start_background_model_loading()
    
    def _setup_window(self) -> None:
        """Configure the main window."""
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=BG_COLOR_DARK)
        # Set minimum window size - optimized for 1280x720 default size
        # Minimum width: 1100px, Minimum height: 650px (ensures all elements are visible)
        self.root.minsize(1100, 650)
        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_ui(self) -> None:
        """Create and layout all UI components."""
        main_frame = tk.Frame(self.root, bg=self.bgcolor)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "inti logo.png")
        create_header(main_frame, self.use_advanced, logo_path)
        
        # Create notebook for tabs (Search and Chat)
        from tkinter import ttk as ttk_module
        self.notebook = ttk_module.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Search tab
        search_tab = tk.Frame(self.notebook, bg=self.bgcolor)
        self.notebook.add(search_tab, text="🔍 Search Products")
        
        # Create status label for model loading (shown at top of search tab)
        self.status_label = tk.Label(
            search_tab,
            text="⏳ Loading AI models in background... (Search available with basic mode)",
            bg=self.bgcolor,
            fg=FG_COLOR_SECONDARY,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL - 1),
            pady=8
        )
        self.status_label.pack(fill=tk.X, padx=20, pady=(10, 0))
        
        # Create input section
        self.product_entry_var = tk.StringVar()
        self.brand_var = tk.StringVar()
        self.budget_var = tk.StringVar()
        self.sort_var = tk.StringVar()
        
        self.product_entry, self.brand_combo, self.send_button = create_input_section(
            search_tab,
            self.engine,
            self.product_entry_var,
            self.brand_var,
            self.budget_var,
            self.sort_var,
            self.send_message
        )
        
        # Create results section
        self.content_container, self.tree = create_results_section(
            search_tab,
            self.use_advanced
        )
        
        # Bind click events for results table
        self.tree.bind("<Button-1>", lambda e: handle_item_click(
            e, self.tree, self.use_advanced, self.show_similar_products_below
        ))
        self.tree.bind("<Double-1>", lambda e: handle_item_double_click(e, self.tree))
        
        # Create similar products section
        self.similar_tree = create_similar_section(self.content_container)
        self.similar_tree.bind("<Double-1>", lambda e: handle_similar_double_click(e, self.similar_tree))
        
        # Store current results for chatbot context
        self.current_results: List[Dict[str, str]] = []
        
        # Chatbot tab (if available) - lazy initialization
        if CHATBOT_AVAILABLE:
            chat_tab = tk.Frame(self.notebook, bg=self.bgcolor)
            self.notebook.add(chat_tab, text="💬 AI Assistant")
            
            # Create chatbot panel (chatbot will be initialized when tab is accessed)
            self.chat_display, self.chat_entry, self.chat_send_button = create_chatbot_panel(
                chat_tab,
                self._get_chatbot_response
            )
            
            # Bind tab change event to lazy-load chatbot
            def on_tab_changed(event):
                if event.widget.index("current") == 1:  # Chat tab is index 1
                    self._initialize_chatbot_lazy()
            self.notebook.bind("<<NotebookTabChanged>>", on_tab_changed)
        
        # Apply styles
        apply_styles()
        
        # Style the notebook tabs
        self._style_notebook()
        
        # Force button colors on macOS after window is created
        self.root.after(100, self._force_button_colors)
    
    def _force_button_colors(self) -> None:
        """Force button colors on macOS (workaround for system color override)."""
        try:
            if hasattr(self, 'send_button'):
                self.send_button.configure(
                    bg=BUTTON_PRIMARY,
                    highlightbackground=BUTTON_PRIMARY,
                    highlightcolor=BUTTON_PRIMARY
                )
        except Exception as e:
            logger.debug(f"Could not force button colors: {e}")
    
    def _start_background_model_loading(self) -> None:
        """Start background thread to load AI models and embeddings."""
        if self._models_loading or self._models_loaded:
            return
        
        self._models_loading = True
        
        def load_models():
            """Background thread function to load models."""
            try:
                logger.info("Starting background loading of AI models and embeddings...")
                self.engine.load_models_and_embeddings()
                self._models_loaded = True
                self._models_loading = False
                logger.info("✓ Background model loading completed successfully")
                
                # Update UI in main thread
                self.root.after(0, self._on_models_loaded)
            except Exception as e:
                logger.error(f"Error loading models in background: {e}", exc_info=True)
                self._models_loading = False
                # Update UI to show error
                self.root.after(0, lambda: self._on_models_load_error(str(e)))
        
        # Start background thread
        thread = threading.Thread(target=load_models, daemon=True)
        thread.start()
        logger.info("Background model loading thread started")
    
    def _on_models_loaded(self) -> None:
        """Update UI when models are loaded successfully."""
        if hasattr(self, 'status_label'):
            self.status_label.config(
                text="✓ AI models loaded successfully! Enhanced search is now available.",
                fg=SUCCESS_COLOR
            )
            # Hide status label after 3 seconds
            self.root.after(3000, lambda: self.status_label.pack_forget())
        logger.info("UI updated: Models loaded successfully")
    
    def _on_models_load_error(self, error_msg: str) -> None:
        """Update UI when model loading fails."""
        if hasattr(self, 'status_label'):
            self.status_label.config(
                text=f"⚠️ Model loading failed. Using basic search mode. Error: {error_msg[:50]}...",
                fg=FG_COLOR_SECONDARY
            )
        logger.warning(f"Model loading failed: {error_msg}")
    
    def show_similar_products_below(
        self, 
        product_name: str, 
        brand: str, 
        product_url: str
    ) -> None:
        """
        Show similar products in the section below the main table.
        
        Args:
            product_name: Name of the selected product
            brand: Brand of the selected product
            product_url: URL of the selected product (unused but kept for compatibility)
        """
        show_similar_products(
            self.similar_tree,
            self.engine,
            product_name,
            brand,
            product_url
        )
    
    def send_message(self) -> None:
        """Handle search button click and display recommendations."""
        self.current_results = perform_search(
            self.product_entry_var,
            self.brand_var,
            self.budget_var,
            self.sort_var,
            self.tree,
            self.similar_tree,
            self.engine,
            self.use_advanced
        )
    
    def _initialize_chatbot_lazy(self) -> None:
        """Lazy initialization of chatbot - only when chat tab is accessed."""
        if self._chatbot_initialized:
            return
        
        if not CHATBOT_AVAILABLE:
            return
        
        try:
            logger.info("Initializing chatbot (lazy load)...")
            self.chatbot = ProductChatbot(use_llm=True, recommender_engine=self.engine)
            self._chatbot_initialized = True
            logger.info("Chatbot initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize chatbot: {e}")
            self.chatbot = None
    
    def _get_chatbot_response(self, user_message: str) -> str:
        """
        Get chatbot response for user message.
        
        Args:
            user_message: User's message/question
            
        Returns:
            Chatbot response string
        """
        # Initialize chatbot if not already initialized
        if not self._chatbot_initialized:
            self._initialize_chatbot_lazy()
        
        if self.chatbot:
            # Get context from current search if available
            context = None
            if hasattr(self, 'current_results') and self.current_results:
                # Provide context about current search results
                context = f"User is searching for products. Found {len(self.current_results)} results."
            
            return self.chatbot.generate_response(user_message, context)
        else:
            return "Chatbot is not available. Please ensure transformers library is installed."
    
    def _style_notebook(self) -> None:
        """Style the notebook tabs to match the dark theme."""
        try:
            from tkinter import ttk as ttk_module
            style = ttk_module.Style()
            
            # Configure notebook style
            style.configure(
                "TNotebook",
                background=BG_COLOR_DARK,
                borderwidth=0
            )
            style.configure(
                "TNotebook.Tab",
                background=BG_COLOR_INPUT,
                foreground=FG_COLOR_WHITE,
                padding=[20, 10],
                font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold')
            )
            style.map(
                "TNotebook.Tab",
                background=[("selected", BG_COLOR_CARD)],
                expand=[("selected", [1, 1, 1, 0])]
            )
        except Exception as e:
            logger.debug(f"Could not style notebook: {e}")


def main() -> None:
    """Main entry point for the application."""
    root = tk.Tk()
    app = ProdRecommendationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
