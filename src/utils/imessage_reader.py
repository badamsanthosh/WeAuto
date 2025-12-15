"""
iMessage Reader Module
Reads incoming iMessages and extracts stock tickers
"""
import subprocess
import re
from typing import List, Dict, Callable, Optional
from datetime import datetime
import time
import threading

class iMessageReader:
    """Reads and monitors iMessages for stock tickers"""
    
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        """
        Initialize iMessage reader
        
        Args:
            callback: Function to call when stock ticker is detected (ticker: str)
        """
        self.callback = callback
        self.running = False
        self.processed_messages = {}  # Track processed messages by ID
        self.last_check_time = datetime.now()
        
    def _get_recent_messages(self) -> List[Dict]:
        """
        Get recent iMessages using AppleScript
        
        Returns:
            List of message dictionaries
        """
        # AppleScript to get recent messages
        applescript = '''
        tell application "Messages"
            set messageList to {}
            set allChats to every chat
            repeat with aChat in allChats
                try
                    set messagesInChat to messages of aChat
                    repeat with aMessage in messagesInChat
                        try
                            set messageText to text of aMessage
                            set messageDate to date received of aMessage
                            set messageID to id of aMessage as string
                            set end of messageList to {messageID, messageText, messageDate as string}
                        end try
                    end repeat
                end try
            end repeat
            return messageList
        end tell
        '''
        
        try:
            result = subprocess.run(
                ['osascript', '-e', applescript],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Parse result - AppleScript returns a list structure
                # This is simplified - actual parsing would need more work
                messages = []
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        # Parse the message data
                        # Format is complex from AppleScript, so we'll use a simpler approach
                        pass
                return messages
            return []
        except Exception as e:
            print(f"Error reading messages: {e}")
            return []
    
    def _extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock tickers from text
        
        Args:
            text: Message text
            
        Returns:
            List of potential stock tickers
        """
        if not text:
            return []
        
        # Common stock ticker patterns (1-5 uppercase letters, possibly with $ prefix)
        patterns = [
            r'\$([A-Z]{1,5})\b',  # $AAPL format
            r'\b([A-Z]{1,5})\b',  # AAPL format
        ]
        
        tickers = []
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            tickers.extend(matches)
        
        # Filter out common words that aren't tickers
        common_words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER',
            'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW',
            'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'WAY', 'USE',
            'MAN', 'MEN', 'YES', 'YET', 'TOO', 'TRY', 'PUT', 'LET', 'SET', 'SIT',
            'RUN', 'FAR', 'OFF', 'OWN', 'ASK', 'END', 'BAD', 'BIG', 'CUT', 'DID',
            'EAT', 'FEW', 'GOT', 'HAD', 'HOT', 'JOB', 'KEY', 'LAW', 'LOW', 'MAD',
            'NET', 'PAY', 'RED', 'SAY', 'SHE', 'SIX', 'TEN', 'TOP', 'WAR', 'WIN',
            'I', 'A', 'AN', 'AT', 'BE', 'DO', 'GO', 'IF', 'IN', 'IS', 'IT', 'ME',
            'MY', 'NO', 'OF', 'ON', 'OR', 'SO', 'TO', 'UP', 'US', 'WE'
        }
        
        # Filter and validate tickers
        valid_tickers = []
        for ticker in tickers:
            ticker = ticker.strip()
            if (ticker and 
                ticker not in common_words and 
                1 <= len(ticker) <= 5 and
                ticker.isalpha()):
                valid_tickers.append(ticker)
        
        return list(set(valid_tickers))  # Remove duplicates
    
    def check_text_for_tickers(self, text: str) -> List[str]:
        """
        Check text for stock tickers and trigger callback
        
        Args:
            text: Text to check
            
        Returns:
            List of detected tickers
        """
        tickers = self._extract_tickers(text)
        for ticker in tickers:
            if self.callback:
                self.callback(ticker)
        return tickers
    
    def monitor_clipboard(self):
        """
        Monitor clipboard for stock tickers (simpler alternative to reading iMessages)
        This can be used when user copies a message
        """
        import pyperclip
        
        last_clipboard = ""
        while self.running:
            try:
                current_clipboard = pyperclip.paste()
                if current_clipboard != last_clipboard:
                    last_clipboard = current_clipboard
                    tickers = self.check_text_for_tickers(current_clipboard)
                    if tickers:
                        print(f"Detected tickers in clipboard: {tickers}")
                time.sleep(1)
            except Exception as e:
                print(f"Clipboard monitor error: {e}")
                time.sleep(5)
    
    def start(self, use_clipboard: bool = True):
        """
        Start monitoring
        
        Args:
            use_clipboard: If True, monitor clipboard instead of iMessages (simpler)
        """
        self.running = True
        if use_clipboard:
            self.monitor_thread = threading.Thread(target=self.monitor_clipboard, daemon=True)
        else:
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
    
    def _monitor_loop(self):
        """Main monitoring loop for iMessages"""
        while self.running:
            try:
                messages = self._get_recent_messages()
                for msg in messages:
                    msg_id = msg.get('id', '')
                    if msg_id and msg_id not in self.processed_messages:
                        self.processed_messages[msg_id] = True
                        text = msg.get('text', '')
                        if text:
                            self.check_text_for_tickers(text)
                time.sleep(2)  # Check every 2 seconds
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(5)
