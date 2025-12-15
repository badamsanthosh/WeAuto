#!/usr/bin/env python3
"""
WeAuto Desktop Application Launcher
Starts the desktop GUI application
"""
import sys
import os

# Check for tkinter before importing
# Need to check both tkinter and _tkinter (the C extension)
TKINTER_AVAILABLE = False
tkinter_error = None
try:
    import tkinter as tk
    # Try to actually use tkinter to ensure _tkinter is available
    test_root = tk.Tk()
    test_root.withdraw()  # Hide it immediately
    test_root.destroy()
    TKINTER_AVAILABLE = True
except Exception as e:
    TKINTER_AVAILABLE = False
    tkinter_error = str(e)
    print("=" * 80)
    print("❌ tkinter is not available for this Python version")
    print("=" * 80)
    print(f"\nCurrent Python: {sys.executable}")
    print(f"Python version: {sys.version}")
    if tkinter_error:
        print(f"Error: {tkinter_error}")
    
    # Check if system Python has tkinter
    import subprocess
    try:
        result = subprocess.run(
            ['/usr/bin/python3', '-c', 'import tkinter'],
            capture_output=True,
            timeout=2
        )
        if result.returncode == 0:
            print("\n✅ System Python has tkinter available!")
            print("\nSolution: Use system Python instead:")
            print("    /usr/bin/python3 desktop.py")
            print("\nOr install tkinter for Homebrew Python:")
            print("    brew install python-tk")
    except:
        pass
    
    print("\nInstallation instructions:")
    print("\n  macOS (Homebrew Python):")
    print("    brew install python-tk")
    print("    OR use system Python: /usr/bin/python3 desktop.py")
    print("\n  macOS (System Python):")
    print("    tkinter should be pre-installed")
    print("\n  Linux (Ubuntu/Debian):")
    print("    sudo apt-get install python3-tk")
    print("\n  Linux (Fedora):")
    print("    sudo dnf install python3-tkinter")
    print("\nAlternatively, use the command-line interface:")
    print("    python3 run.py --mode scan")
    print("    python3 predict.py --tickers AAPL,MSFT,GOOGL")
    print("\n" + "=" * 80)
    sys.exit(1)

# Only import if tkinter is available
if not TKINTER_AVAILABLE:
    sys.exit(1)

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gui.desktop_app import main

if __name__ == '__main__':
    main()
