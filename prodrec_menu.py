import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

class ProdRecommendation:
    def __init__(self, root):
        
        # Initialize variables
        self.conversation_history = []
        self.is_typing = False
        self.message_count = 0
        self.placeholder_active = True
        self.bgcolor = '#50657D'
        
        self.root = root
        self.root.title("Product Recommendation System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#0a0a0a')
        self.root.resizable(True, True)
        
        main_frame = tk.Frame(self.root, bg=self.bgcolor)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        """Setup modern header"""
        header_frame = tk.Frame(main_frame, bg=self.bgcolor, height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header_frame, 
            text="Product Recommendation System", 
            font=('Segoe UI', 24, 'bold'), 
            fg='#ffffff', 
            bg=self.bgcolor
        )
        title_label.pack(pady=(15, 0))
        
        # Subtitle
        """
        subtitle_label = tk.Label(
            header_frame, 
            text="Product Recommendation System", 
            font=('Segoe UI', 10), 
            fg='#888888', 
            bg='#1a1a2e'
        )
        subtitle_label.pack(pady=(0, 15))
        """
        """Setup status bar
        status_frame = tk.Frame(main_frame, bg='#1a1a2e', height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="Loading dataset...",
            font=('Segoe UI', 9),
            fg='#ffaa00',
            bg='#1a1a2e'
        )
        self.status_label.pack(side=tk.LEFT, padx=15, pady=5)
        
        self.counter_label = tk.Label(
            status_frame,
            text="0 messages",
            font=('Segoe UI', 9),
            fg='#888888',
            bg='#1a1a2e'
        )
        self.counter_label.pack(side=tk.RIGHT, padx=15, pady=5)
        """
        """Setup chat area"""
        # Chat container - fill remaining space
        chat_container = tk.Frame(main_frame, bg='#0a0a0a')
        chat_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 0))
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            chat_container,
            wrap=tk.WORD,
            font=('Segoe UI', 11),
            bg='#1e1e1e',
            fg='#ffffff',
            insertbackground='#ffffff',
            selectbackground='#0078d4',
            selectforeground='#ffffff',
            relief=tk.FLAT,
            bd=0,
            padx=20,
            pady=20,
            spacing1=8,
            spacing2=4,
            spacing3=8
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        self.chat_display.tag_config("user_name", foreground="#00d4aa", font=('Segoe UI', 11, 'bold'), spacing1=10)
        self.chat_display.tag_config("user_message", foreground="#ffffff", font=('Segoe UI', 11), lmargin1=20, lmargin2=20, spacing3=10)
        self.chat_display.tag_config("bot_name", foreground="#0078d4", font=('Segoe UI', 11, 'bold'), spacing1=10)
        self.chat_display.tag_config("bot_message", foreground="#e6e6e6", font=('Segoe UI', 11), lmargin1=20, lmargin2=20, spacing3=10)
        self.chat_display.tag_config("typing", foreground="#888888", font=('Segoe UI', 10, 'italic'))
        
        """Setup input area with fixed layout"""
        # Main input container - ensure it's always visible
        input_main_container = tk.Frame(main_frame, bg=self.bgcolor)
        input_main_container.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=10)
        
        # Quick suggestions section
        """
        suggestions_frame = tk.Frame(input_main_container, bg=self.bgcolor)
        suggestions_frame.pack(fill=tk.X, pady=(0, 10))
        
        suggestions_label = tk.Label(
            suggestions_frame,
            text="Quick Suggestions:",
            font=('Segoe UI', 10, 'bold'),
            fg='#888888',
            bg='#0a0a0a'
        )
        suggestions_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Suggestion buttons
        buttons_frame = tk.Frame(suggestions_frame, bg='#0a0a0a')
        buttons_frame.pack(fill=tk.X)
        
        suggestions = [
            ("Cat1", "comm1"),
            ("Cat2", "comm2"),
            ("Cat3", "comm3"),
            ("Top Rated", "highest rated"),
            ("Recent", "recent"),
            ("Surprise", "recommend something good")
        ]
        
        for text, query in suggestions:
            btn = tk.Button(
                buttons_frame,
                text=text,
                #command=lambda q=query: self.quick_search(q),
                font=('Segoe UI', 9),
                bg='#2d2d2d',
                fg='#ffffff',
                activebackground='#404040',
                activeforeground='#ffffff',
                relief=tk.FLAT,
                bd=0,
                padx=12,
                pady=6,
                cursor='hand2'
            )
            btn.pack(side=tk.LEFT, padx=(0, 8), pady=2)
            
            # Hover effects
            btn.bind('<Enter>', lambda e, b=btn: b.config(bg='#404040'))
            btn.bind('<Leave>', lambda e, b=btn: b.config(bg='#2d2d2d'))
        """
        # Input section
        input_frame = tk.Frame(input_main_container, bg='#2d2d2d')
        input_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Text input
        self.user_input = tk.Text(
            input_frame,
            height=3,
            font=('Segoe UI', 12),
            bg='#2d2d2d',
            fg='#ffffff',
            insertbackground='#ffffff',
            selectbackground='#0078d4',
            relief=tk.FLAT,
            bd=0,
            padx=15,
            pady=10,
            wrap=tk.WORD
        )
        self.user_input.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(15, 10), pady=10)
        #self.user_input.bind('<Return>', self.handle_enter_key)
        #self.user_input.bind('<KeyPress>', self.on_key_press)
        #self.user_input.bind('<FocusIn>', self.on_focus_in)
        
        # Send button
        send_button = tk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            font=('Segoe UI', 11, 'bold'),
            bg='#0078d4',
            fg='white',
            activebackground='#106ebe',
            activeforeground='white',
            relief=tk.FLAT,
            bd=0,
            width=8,
            cursor='hand2'
        )
        send_button.pack(side=tk.RIGHT, padx=(0, 15), pady=10)
        
        # Hover effects
        send_button.bind('<Enter>', lambda e: send_button.config(bg='#106ebe'))
        send_button.bind('<Leave>', lambda e: send_button.config(bg='#0078d4'))
        
    def send_message(self):
        print('send message')

def main():
    root = tk.Tk()
    app = ProdRecommendation(root)
    root.mainloop()

if __name__ == "__main__":
    main()