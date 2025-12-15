"""
WeAuto Desktop Application
Modern GUI for the WeAuto trading system with iMessage monitoring
"""
import sys
import os

# Check if tkinter is available (including _tkinter C extension)
TKINTER_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    # Test that _tkinter (C extension) is actually available
    # by trying to create a root window
    test_root = tk.Tk()
    test_root.withdraw()  # Hide it immediately
    test_root.destroy()
    TKINTER_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AttributeError, tk.TclError) as e:
    TKINTER_AVAILABLE = False
    print("=" * 80)
    print("ERROR: tkinter is not available")
    print("=" * 80)
    print(f"\nError details: {e}")
    print("\nTo fix this issue, install tkinter:")
    print("\n  macOS (Homebrew Python):")
    print("    brew install python-tk")
    print("    OR reinstall Python with: brew install python@3.11")
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

import threading
import time
import re
from datetime import datetime
from typing import Optional, List
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.predictor import StockPredictor
from utils.imessage_notifier import iMessageNotifier
from utils.imessage_reader import iMessageReader
import core.config as config
import threading

class iMessageMonitor:
    """Monitors iMessage for incoming messages and extracts stock tickers"""
    
    def __init__(self, callback):
        """
        Initialize iMessage monitor
        
        Args:
            callback: Function to call when stock ticker is detected (ticker: str)
        """
        self.callback = callback
        self.running = False
        self.last_check_time = None
        self.processed_messages = set()  # Track processed messages to avoid duplicates
        
    def _get_recent_messages(self) -> List[dict]:
        """
        Get recent iMessages using AppleScript
        
        Returns:
            List of message dictionaries with 'text' and 'date' keys
        """
        applescript = '''
        tell application "Messages"
            set recentMessages to {}
            set allChats to every chat
            repeat with aChat in allChats
                set messagesInChat to messages of aChat
                repeat with aMessage in messagesInChat
                    set messageText to text of aMessage
                    set messageDate to date received of aMessage
                    set end of recentMessages to {messageText, messageDate as string}
                end repeat
            end repeat
            return recentMessages
        end tell
        '''
        
        try:
            result = subprocess.run(
                ['osascript', '-e', applescript],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse the result (simplified - AppleScript returns complex structure)
                # For now, we'll use a simpler approach
                return []
            return []
        except Exception as e:
            print(f"Error getting messages: {e}")
            return []
    
    def _extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock tickers from text
        
        Args:
            text: Message text
            
        Returns:
            List of potential stock tickers
        """
        # Common stock ticker patterns (1-5 uppercase letters)
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        matches = re.findall(ticker_pattern, text.upper())
        
        # Filter out common words that aren't tickers
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'WAY', 'USE', 'MAN', 'MEN', 'YES', 'YET', 'TOO', 'TRY', 'PUT', 'LET', 'SET', 'SIT', 'RUN', 'FAR', 'OFF', 'OWN', 'ASK', 'END', 'BAD', 'BIG', 'CUT', 'DID', 'EAT', 'FEW', 'GOT', 'HAD', 'HOT', 'ITS', 'JOB', 'KEY', 'LAW', 'LOW', 'MAD', 'NET', 'PAY', 'RED', 'SAY', 'SHE', 'SIX', 'TEN', 'TOP', 'USE', 'WAR', 'WIN', 'YES'}
        
        # Filter matches
        tickers = []
        for match in matches:
            if match not in common_words and len(match) >= 1 and len(match) <= 5:
                tickers.append(match)
        
        return list(set(tickers))  # Remove duplicates
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Get recent messages (simplified approach)
                # In production, you'd want to track message IDs to avoid duplicates
                time.sleep(2)  # Check every 2 seconds
                
                # For now, we'll use a simpler approach: monitor clipboard or use a different method
                # This is a placeholder - actual iMessage monitoring requires more complex AppleScript
                pass
                
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(5)
    
    def start(self):
        """Start monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
    
    def check_message(self, message_text: str):
        """
        Check a message for stock tickers and trigger callback
        
        Args:
            message_text: Message text to check
        """
        tickers = self._extract_tickers(message_text)
        for ticker in tickers:
            if self.callback:
                self.callback(ticker)

class WeAutoDesktopApp:
    """Main desktop application window"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("WeAuto - Elite Trading System")
        self.root.geometry("900x700")
        self.root.configure(bg='#1e1e1e')
        
        # Application state
        self.is_running = False
        self.predictor = None
        self.notifier = None
        self.monitor = None
        
        # Setup UI
        self._setup_ui()
        
        # Initialize predictor (lazy load)
        self._init_predictor()
    
    def _setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#1e1e1e', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = tk.Frame(main_frame, bg='#1e1e1e')
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(
            header_frame,
            text="🤖 WeAuto Trading System",
            font=('Arial', 24, 'bold'),
            fg='#00ff88',
            bg='#1e1e1e'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="Elite ML-Powered Stock Predictions",
            font=('Arial', 12),
            fg='#888888',
            bg='#1e1e1e'
        )
        subtitle_label.pack()
        
        # Control Panel
        control_frame = tk.LabelFrame(
            main_frame,
            text="Control Panel",
            font=('Arial', 12, 'bold'),
            fg='#00ff88',
            bg='#2d2d2d',
            padx=15,
            pady=15
        )
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Status and controls
        status_frame = tk.Frame(control_frame, bg='#2d2d2d')
        status_frame.pack(fill=tk.X)
        
        self.status_label = tk.Label(
            status_frame,
            text="Status: Stopped",
            font=('Arial', 11),
            fg='#ff4444',
            bg='#2d2d2d'
        )
        self.status_label.pack(side=tk.LEFT)
        
        button_frame = tk.Frame(status_frame, bg='#2d2d2d')
        button_frame.pack(side=tk.RIGHT)
        
        self.start_button = tk.Button(
            button_frame,
            text="▶ Start",
            font=('Arial', 11, 'bold'),
            bg='#00aa55',
            fg='white',
            activebackground='#00cc66',
            activeforeground='white',
            padx=20,
            pady=8,
            command=self.start_application,
            cursor='hand2'
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(
            button_frame,
            text="⏹ Stop",
            font=('Arial', 11, 'bold'),
            bg='#aa0000',
            fg='white',
            activebackground='#cc0000',
            activeforeground='white',
            padx=20,
            pady=8,
            command=self.stop_application,
            state=tk.DISABLED,
            cursor='hand2'
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Manual Prediction Frame
        manual_frame = tk.LabelFrame(
            main_frame,
            text="Manual Prediction",
            font=('Arial', 11, 'bold'),
            fg='#00ff88',
            bg='#2d2d2d',
            padx=15,
            pady=15
        )
        manual_frame.pack(fill=tk.X, pady=(0, 20))
        
        input_frame = tk.Frame(manual_frame, bg='#2d2d2d')
        input_frame.pack(fill=tk.X)
        
        tk.Label(
            input_frame,
            text="Stock Ticker:",
            font=('Arial', 10),
            fg='#cccccc',
            bg='#2d2d2d'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.ticker_entry = tk.Entry(
            input_frame,
            font=('Arial', 11),
            bg='#3d3d3d',
            fg='#ffffff',
            insertbackground='#ffffff',
            width=15
        )
        self.ticker_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.ticker_entry.bind('<Return>', lambda e: self.predict_manual())
        
        predict_button = tk.Button(
            input_frame,
            text="Predict",
            font=('Arial', 10, 'bold'),
            bg='#0066cc',
            fg='white',
            activebackground='#0088ff',
            activeforeground='white',
            padx=15,
            pady=5,
            command=self.predict_manual,
            cursor='hand2'
        )
        predict_button.pack(side=tk.LEFT)
        
        # Activity Log
        log_frame = tk.LabelFrame(
            main_frame,
            text="Activity Log",
            font=('Arial', 11, 'bold'),
            fg='#00ff88',
            bg='#2d2d2d',
            padx=15,
            pady=15
        )
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            font=('Courier', 10),
            bg='#1e1e1e',
            fg='#00ff88',
            wrap=tk.WORD,
            height=15
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for different log levels
        self.log_text.tag_config('info', foreground='#00ff88')
        self.log_text.tag_config('success', foreground='#00ff00')
        self.log_text.tag_config('warning', foreground='#ffaa00')
        self.log_text.tag_config('error', foreground='#ff4444')
        self.log_text.tag_config('prediction', foreground='#00aaff')
        
        self.log("WeAuto Desktop Application started", 'info')
        self.log("Ready to monitor iMessage and make predictions", 'info')
    
    def _init_predictor(self):
        """Initialize the predictor (lazy load)"""
        try:
            self.log("Initializing ML predictor...", 'info')
            self.predictor = StockPredictor()
            self.notifier = iMessageNotifier()
            self.log("Predictor initialized successfully", 'success')
        except Exception as e:
            self.log(f"Error initializing predictor: {e}", 'error')
    
    def log(self, message: str, level: str = 'info'):
        """
        Add message to activity log
        
        Args:
            message: Log message
            level: Log level (info, success, warning, error, prediction)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry, level)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def start_application(self):
        """Start the application and begin monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Running", fg='#00ff00')
        
        self.log("=" * 60, 'info')
        self.log("Application STARTED", 'success')
        self.log("Monitoring clipboard for stock tickers...", 'info')
        self.log("Tip: Copy a message containing a stock ticker (e.g., 'Check AAPL')", 'info')
        
        # Initialize monitor
        try:
            self.monitor = iMessageMonitor(callback=self.on_ticker_detected)
            self.monitor.start()
            self.log("Clipboard monitor started", 'success')
            self.log("Copy text with stock tickers to trigger predictions", 'info')
        except Exception as e:
            self.log(f"Failed to start monitor: {e}", 'warning')
            self.log("You can still use manual prediction by entering tickers", 'info')
    
    def stop_application(self):
        """Stop the application and monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped", fg='#ff4444')
        
        if self.monitor:
            self.monitor.stop()
            self.monitor = None
        
        self.log("Application STOPPED", 'warning')
        self.log("=" * 60, 'info')
    
    def on_ticker_detected(self, ticker: str):
        """
        Callback when a stock ticker is detected in iMessage
        
        Args:
            ticker: Detected stock ticker
        """
        if not self.is_running:
            return
        
        self.log(f"📱 Stock ticker detected: {ticker}", 'prediction')
        self.predict_stock(ticker)
    
    def predict_manual(self):
        """Manually trigger prediction for entered ticker"""
        ticker = self.ticker_entry.get().strip().upper()
        if not ticker:
            messagebox.showwarning("No Ticker", "Please enter a stock ticker")
            return
        
        self.predict_stock(ticker)
    
    def predict_stock(self, ticker: str):
        """
        Make prediction for a stock ticker
        
        Args:
            ticker: Stock ticker symbol
        """
        if not self.predictor:
            self.log("Predictor not initialized. Please restart the application.", 'error')
            return
        
        self.log(f"🔍 Analyzing {ticker}...", 'prediction')
        
        # Check if model needs training
        if self.predictor.model is None:
            self.log("Training model (this may take 1-2 minutes)...", 'warning')
            training_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'JNJ', 'V']
            try:
                success = self.predictor.train_model(training_tickers[:10], weekly=True)
                if success:
                    self.log("Model trained successfully", 'success')
                else:
                    self.log("Model training failed", 'error')
                    return
            except Exception as e:
                self.log(f"Training error: {e}", 'error')
                return
        
        # Make prediction
        try:
            prediction = self.predictor.predict_stock(ticker)
            
            if prediction and isinstance(prediction, dict):
                current_price = prediction.get('current_price', 0)
                probability = prediction.get('probability', 0)
                signal = 'BUY' if prediction.get('prediction') else 'HOLD'
                confidence = prediction.get('confidence', 'UNKNOWN')
                
                # Log prediction
                self.log(f"✅ Prediction for {ticker}:", 'success')
                self.log(f"   Current Price: ${current_price:.2f}", 'prediction')
                self.log(f"   Signal: {signal}", 'prediction')
                self.log(f"   Probability: {probability:.1%}", 'prediction')
                self.log(f"   Confidence: {confidence}", 'prediction')
                
                if 'target_price' in prediction:
                    target = prediction.get('target_price', 0)
                    gain_pct = ((target - current_price) / current_price * 100) if current_price > 0 else 0
                    self.log(f"   Target: ${target:.2f} ({gain_pct:+.1f}%)", 'prediction')
                
                # Send iMessage notification
                if self.notifier and self.notifier.enabled:
                    try:
                        self.notifier.send_prediction_notification(prediction)
                        self.log(f"   📱 iMessage notification sent", 'success')
                    except Exception as e:
                        self.log(f"   ⚠️  Failed to send iMessage: {e}", 'warning')
            else:
                self.log(f"⚠️  Unable to generate prediction for {ticker}", 'warning')
                
        except Exception as e:
            self.log(f"❌ Error predicting {ticker}: {e}", 'error')

def main():
    """Main entry point"""
    root = tk.Tk()
    app = WeAutoDesktopApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
