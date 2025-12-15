#!/usr/bin/env python3
"""
Test iMessage Notification
Quick script to test iMessage configuration
"""
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.imessage_notifier import iMessageNotifier

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🧪 Testing iMessage Configuration")
    print("="*80 + "\n")
    
    notifier = iMessageNotifier()
    
    if not notifier.enabled:
        print("⚠️  iMessage notifications are disabled")
        print("\nTo enable:")
        print("1. Set IMESSAGE_ENABLED=true in .env file")
        print("2. Set IMESSAGE_APPLE_ID or IMESSAGE_PHONE")
        print("\nExample .env file:")
        print("  IMESSAGE_ENABLED=true")
        print("  IMESSAGE_PHONE=+12345678900")
        sys.exit(1)
    
    if not notifier.recipients:
        print("⚠️  No recipients configured")
        print("\nPlease set either:")
        print("  IMESSAGE_APPLE_ID=recipient@example.com")
        print("  OR")
        print("  IMESSAGE_PHONE=+12345678900")
        print("\nFor multiple recipients, use comma-separated phone numbers:")
        print("  IMESSAGE_PHONE=+12345678900,+19876543210,+15551234567")
        sys.exit(1)
    
    print(f"📱 Recipients ({len(notifier.recipients)}): {', '.join(notifier.recipients)}")
    print(f"✅ Enabled: {notifier.enabled}\n")
    
    print("Sending test message...")
    success = notifier.test_connection()
    
    if success:
        print("\n" + "="*80)
        print("✅ Test successful! Check your Messages app.")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("❌ Test failed. See error messages above.")
        print("="*80 + "\n")
        sys.exit(1)
