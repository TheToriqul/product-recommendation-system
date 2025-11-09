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

# Set environment variable to suppress tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from src.core.recommender_engine import ProductRecommender, GENAI_AVAILABLE

# Import UI modules
from src.ui.ui_constants import (
    WINDOW_TITLE, WINDOW_SIZE, BG_COLOR_MAIN, BG_COLOR_DARK,
    BG_COLOR_INPUT, BG_COLOR_CARD, BG_COLOR_ENTRY, BG_COLOR_HOVER,
    FG_COLOR_WHITE, FG_COLOR_SECONDARY, SUCCESS_COLOR,
    FONT_FAMILY, FONT_SIZE_NORMAL,
    BUTTON_PRIMARY
)
from src.ui.ui_styles import apply_styles
from src.ui.ui_components import (
    create_header, create_input_section, create_results_section,
    create_similar_section, create_evaluation_tab
)
from src.ui.ui_handlers import (
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
    from src.chatbot.chatbot import ProductChatbot
    from src.chatbot.chatbot_ui import create_chatbot_panel
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
        tab_index = 1
        if CHATBOT_AVAILABLE:
            chat_tab = tk.Frame(self.notebook, bg=self.bgcolor)
            self.notebook.add(chat_tab, text="💬 AI Assistant")
            
            # Create chatbot panel (chatbot will be initialized when tab is accessed)
            self.chat_display, self.chat_entry, self.chat_send_button = create_chatbot_panel(
                chat_tab,
                self._get_chatbot_response
            )
            
            tab_index = 2
        
        # Evaluation tab
        eval_tab = tk.Frame(self.notebook, bg=self.bgcolor)
        self.notebook.add(eval_tab, text="📊 Evaluation")
        
        # Create evaluation tab components
        self.eval_canvas, self.eval_buttons_frame, self.eval_status_label, self.eval_scrollable_frame = create_evaluation_tab(eval_tab)
        self._setup_evaluation_buttons()
        self._load_evaluation_results()
        
        # Bind tab change event to lazy-load chatbot
        def on_tab_changed(event):
            current_tab = event.widget.index("current")
            if CHATBOT_AVAILABLE and current_tab == 1:  # Chat tab
                self._initialize_chatbot_lazy()
            elif current_tab == tab_index:  # Evaluation tab
                self._load_evaluation_results()  # Refresh results when tab is opened
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
    
    def _setup_evaluation_buttons(self) -> None:
        """Setup buttons for evaluation tab."""
        from src.ui.evaluation_ui import run_quick_evaluation, open_evaluation_report, open_evaluation_folder
        from src.ui.ui_constants import BUTTON_PRIMARY, BUTTON_PRIMARY_HOVER, FG_COLOR_WHITE, FONT_FAMILY, FONT_SIZE_NORMAL
        
        # Run Quick Evaluation button
        run_btn = tk.Button(
            self.eval_buttons_frame,
            text="Run Quick Evaluation",
            command=self._run_evaluation,
            font=(FONT_FAMILY, FONT_SIZE_NORMAL, 'bold'),
            bg=BUTTON_PRIMARY,
            fg=FG_COLOR_WHITE,
            activebackground=BUTTON_PRIMARY_HOVER,
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
        run_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # View Report button
        view_btn = tk.Button(
            self.eval_buttons_frame,
            text="View Full Report",
            command=self._view_report,
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
            padx=20,
            pady=10
        )
        view_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Open Folder button
        folder_btn = tk.Button(
            self.eval_buttons_frame,
            text="Open Results Folder",
            command=open_evaluation_folder,
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
            padx=20,
            pady=10
        )
        folder_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Refresh button
        refresh_btn = tk.Button(
            self.eval_buttons_frame,
            text="Refresh Results",
            command=self._load_evaluation_results,
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
            padx=20,
            pady=10
        )
        refresh_btn.pack(side=tk.LEFT)
    
    def _load_evaluation_results(self) -> None:
        """Load and display latest evaluation results."""
        from src.ui.evaluation_ui import load_latest_evaluation_results, render_evaluation_results
        from src.ui.ui_constants import FG_COLOR_SECONDARY, SUCCESS_COLOR
        
        results = load_latest_evaluation_results()
        
        if results:
            # Render results in beautiful card-based UI
            render_evaluation_results(self.eval_scrollable_frame, results)
            
            # Update canvas scroll region
            self.eval_canvas.update_idletasks()
            self.eval_canvas.config(scrollregion=self.eval_canvas.bbox("all"))
            
            self.eval_status_label.config(
                text="✓ Latest evaluation results loaded successfully.",
                fg=SUCCESS_COLOR
            )
        else:
            # Clear and show empty state
            for widget in self.eval_scrollable_frame.winfo_children():
                widget.destroy()
            
            from src.ui.ui_constants import BG_COLOR_MAIN, FG_COLOR_SECONDARY, FONT_FAMILY, FONT_SIZE_HEADING, FONT_SIZE_NORMAL
            
            empty_frame = tk.Frame(self.eval_scrollable_frame, bg=BG_COLOR_MAIN)
            empty_frame.pack(fill=tk.BOTH, expand=True, pady=50)
            
            tk.Label(
                empty_frame,
                text="📊",
                font=(FONT_FAMILY, 48),
                fg=FG_COLOR_SECONDARY,
                bg=BG_COLOR_MAIN
            ).pack(pady=20)
            
            tk.Label(
                empty_frame,
                text="No evaluation results available",
                font=(FONT_FAMILY, FONT_SIZE_HEADING),
                fg=FG_COLOR_SECONDARY,
                bg=BG_COLOR_MAIN
            ).pack(pady=10)
            
            tk.Label(
                empty_frame,
                text="Click 'Run Quick Evaluation' to generate metrics",
                font=(FONT_FAMILY, FONT_SIZE_NORMAL),
                fg=FG_COLOR_SECONDARY,
                bg=BG_COLOR_MAIN
            ).pack()
            
            self.eval_canvas.update_idletasks()
            self.eval_canvas.config(scrollregion=self.eval_canvas.bbox("all"))
            
            self.eval_status_label.config(
                text="No evaluation results loaded. Click 'Run Quick Evaluation' to generate metrics.",
                fg=FG_COLOR_SECONDARY
            )
    
    def _run_evaluation(self) -> None:
        """Run quick evaluation in background."""
        from src.ui.evaluation_ui import run_quick_evaluation
        from src.ui.ui_constants import FG_COLOR_SECONDARY
        
        self.eval_status_label.config(
            text="⏳ Running evaluation... This may take a few minutes. Please wait...",
            fg=FG_COLOR_SECONDARY
        )
        
        def on_complete(success: bool, message: str):
            self.root.after(0, lambda: self._on_evaluation_complete(success, message))
        
        run_quick_evaluation(on_complete)
    
    def _on_evaluation_complete(self, success: bool, message: str) -> None:
        """Handle evaluation completion."""
        from src.ui.ui_constants import SUCCESS_COLOR, FG_COLOR_SECONDARY
        
        if success:
            self.eval_status_label.config(
                text=f"✓ {message}",
                fg=SUCCESS_COLOR
            )
            # Reload results
            self._load_evaluation_results()
        else:
            self.eval_status_label.config(
                text=f"⚠️ {message}",
                fg=FG_COLOR_SECONDARY
            )
    
    def _view_report(self) -> None:
        """Open the full evaluation report."""
        from src.ui.evaluation_ui import open_evaluation_report
        
        report_path = open_evaluation_report()
        if report_path:
            self.eval_status_label.config(
                text=f"✓ Opened report: {os.path.basename(report_path)}",
                fg=SUCCESS_COLOR
            )
        else:
            self.eval_status_label.config(
                text="⚠️ No evaluation report found. Run an evaluation first.",
                fg=FG_COLOR_SECONDARY
            )
    
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
