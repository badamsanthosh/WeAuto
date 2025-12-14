#!/usr/bin/env python3
"""
WeAuto Entry Point
Runs the main application from the project root
"""
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run main
from main import main

if __name__ == '__main__':
    main()
