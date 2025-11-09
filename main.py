"""
Main Entry Point for Product Recommendation System

Run this script to launch the GUI application.
Usage: python main.py
"""

import sys
import os

# Set environment variable to suppress tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.ui.app_gui import main

if __name__ == "__main__":
    main()

