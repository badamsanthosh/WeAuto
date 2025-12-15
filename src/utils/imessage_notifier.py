"""
iMessage Notification Module
Sends notifications via Apple's iMessage when predictions are made
"""
import subprocess
import os
from typing import Dict, Optional, List
import core.config as config

class iMessageNotifier:
    """Sends iMessages via AppleScript on macOS"""
    
    def __init__(self):
        self.enabled = getattr(config, 'IMESSAGE_ENABLED', False)
        self.recipient_apple_id = getattr(config, 'IMESSAGE_APPLE_ID', '')
        self.recipient_phone = getattr(config, 'IMESSAGE_PHONE', '')
        
        # Parse recipients - support multiple phone numbers (comma-separated)
        self.recipients: List[str] = []
        
        if self.recipient_phone:
            # Split by comma and strip whitespace
            phone_list = [p.strip() for p in self.recipient_phone.split(',') if p.strip()]
            self.recipients.extend(phone_list)
        
        if self.recipient_apple_id and not self.recipient_phone:
            # Only add Apple ID if no phone numbers are set
            self.recipients.append(self.recipient_apple_id)
        
        # For backward compatibility
        self.recipient = self.recipients[0] if self.recipients else ''
        
    def _send_imessage(self, message: str, recipient: str) -> bool:
        """
        Send iMessage using AppleScript
        
        Args:
            message: Message text to send
            recipient: Phone number or Apple ID (email)
            
        Returns:
            True if successful, False otherwise
        """
        if not recipient:
            return False
            
        # Escape special characters for AppleScript
        escaped_message = message.replace('\\', '\\\\').replace('"', '\\"')
        escaped_recipient = recipient.replace('\\', '\\\\').replace('"', '\\"')
        
        # AppleScript to send iMessage
        applescript = f'''
        tell application "Messages"
            set targetService to 1st account whose service type = iMessage
            set targetBuddy to participant "{escaped_recipient}" of targetService
            send "{escaped_message}" to targetBuddy
        end tell
        '''
        
        try:
            # Run AppleScript
            result = subprocess.run(
                ['osascript', '-e', applescript],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return True
            else:
                print(f"⚠️  iMessage error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠️  iMessage timeout - Messages app may not be responding")
            return False
        except Exception as e:
            print(f"⚠️  iMessage error: {e}")
            return False
    
    def send_prediction_notification(self, prediction: Dict) -> bool:
        """
        Send notification when a prediction is made to all configured recipients
        
        Args:
            prediction: Dictionary with prediction data
            
        Returns:
            True if at least one notification sent successfully
        """
        if not self.enabled:
            return False
            
        if not self.recipients:
            print("⚠️  iMessage recipients not configured")
            return False
        
        # Format message
        ticker = prediction.get('ticker', 'UNKNOWN')
        current_price = prediction.get('current_price', 0)
        probability = prediction.get('probability', 0)
        confidence = prediction.get('confidence', 'UNKNOWN')
        signal = 'BUY' if prediction.get('prediction') else 'HOLD'
        
        # Build message
        message = f"📊 WeAuto Prediction Alert\n\n"
        message += f"Ticker: {ticker}\n"
        message += f"Current Price: ${current_price:.2f}\n"
        message += f"Signal: {signal}\n"
        message += f"Probability: {probability:.1%}\n"
        message += f"Confidence: {confidence}\n"
        
        # Add target price if available
        if 'target_price' in prediction:
            target = prediction.get('target_price', 0)
            gain_pct = ((target - current_price) / current_price * 100) if current_price > 0 else 0
            message += f"Target: ${target:.2f} ({gain_pct:+.1f}%)\n"
        
        # Add RSI if available
        if 'rsi' in prediction and prediction['rsi']:
            message += f"RSI: {prediction['rsi']:.1f}\n"
        
        # Add MA signal if available
        if 'ma_signal' in prediction:
            message += f"MA Signal: {prediction.get('ma_signal', 'N/A')}\n"
        
        message += f"\n🤖 WeAuto Trading System"
        
        # Send message to all recipients
        return self._send_to_all(message)
    
    def send_trade_alert(self, ticker: str, action: str, price: float, 
                        quantity: int = None, reason: str = None) -> bool:
        """
        Send notification for trade execution to all configured recipients
        
        Args:
            ticker: Stock ticker
            action: BUY or SELL
            price: Execution price
            quantity: Number of shares (optional)
            reason: Reason for trade (optional)
            
        Returns:
            True if at least one notification sent successfully
        """
        if not self.enabled:
            return False
            
        if not self.recipients:
            return False
        
        message = f"🔔 WeAuto Trade Alert\n\n"
        message += f"Action: {action}\n"
        message += f"Ticker: {ticker}\n"
        message += f"Price: ${price:.2f}\n"
        
        if quantity:
            message += f"Quantity: {quantity}\n"
            message += f"Total: ${price * quantity:.2f}\n"
        
        if reason:
            message += f"Reason: {reason}\n"
        
        message += f"\n🤖 WeAuto Trading System"
        
        return self._send_to_all(message)
    
    def _send_to_all(self, message: str) -> bool:
        """
        Send message to all configured recipients
        
        Args:
            message: Message text to send
            
        Returns:
            True if at least one message sent successfully
        """
        if not self.recipients:
            return False
        
        success_count = 0
        total_recipients = len(self.recipients)
        
        for recipient in self.recipients:
            if self._send_imessage(message, recipient):
                success_count += 1
            else:
                print(f"⚠️  Failed to send to {recipient}")
        
        if success_count > 0:
            print(f"📱 Sent to {success_count}/{total_recipients} recipient(s)")
            return True
        else:
            print(f"❌ Failed to send to all {total_recipients} recipient(s)")
            return False
    
    def test_connection(self) -> bool:
        """
        Test iMessage connection and configuration
        
        Returns:
            True if test message sent to at least one recipient successfully
        """
        if not self.enabled:
            print("ℹ️  iMessage notifications are disabled")
            return False
            
        if not self.recipients:
            print("⚠️  iMessage recipients not configured")
            print("   Please set IMESSAGE_APPLE_ID or IMESSAGE_PHONE in config")
            print("   For multiple recipients, use comma-separated phone numbers:")
            print("   IMESSAGE_PHONE=+1234567890,+19876543210")
            return False
        
        test_message = "🧪 WeAuto iMessage Test\n\nThis is a test message from WeAuto Trading System.\n\nIf you received this, your iMessage notifications are working correctly! ✅"
        
        print(f"📱 Sending test message to {len(self.recipients)} recipient(s)...")
        print(f"   Recipients: {', '.join(self.recipients)}")
        
        success = self._send_to_all(test_message)
        
        if success:
            print("✅ Test message(s) sent successfully!")
            print("   Please check your Messages app")
        else:
            print("❌ Failed to send test message(s)")
            print("   Make sure:")
            print("   • Messages app is open and signed in")
            print("   • All recipients are in your contacts")
            print("   • You have permission to send messages")
        
        return success
