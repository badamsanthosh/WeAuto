#!/usr/bin/env python3
"""
WeAuto Desktop Application Launcher
Starts the desktop GUI application
"""
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gui.desktop_app import main

if __name__ == '__main__':
    main()
