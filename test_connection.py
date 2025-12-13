"""
Test script to verify Moomoo API connection
"""
from moomoo_integration import MoomooIntegration
import config

def test_connection():
    """Test Moomoo API connection"""
    print("Testing Moomoo API Connection...")
    print(f"Host: {config.MOOMOO_HOST}")
    print(f"Port: {config.MOOMOO_PORT}")
    print(f"Trading Environment: {config.TRADING_ENV}")
    print("-" * 60)
    
    moomoo = MoomooIntegration()
    
    # Test connection
    if moomoo.connect():
        print("✅ Successfully connected to Moomoo API!")
        
        # Test account info
        print("\nFetching account information...")
        account_info = moomoo.get_account_info()
        if account_info:
            print("✅ Account info retrieved:")
            print(f"   {account_info}")
        else:
            print("⚠️  Could not retrieve account info")
        
        # Test market data
        print("\nTesting market data retrieval...")
        test_ticker = "AAPL"
        price = moomoo.get_current_price(test_ticker)
        if price:
            print(f"✅ Current price for {test_ticker}: ${price:.2f}")
        else:
            print(f"⚠️  Could not retrieve price for {test_ticker}")
        
        # Test positions
        print("\nFetching current positions...")
        positions = moomoo.get_positions()
        if positions:
            print(f"✅ Found {len(positions)} open positions:")
            for pos in positions:
                print(f"   {pos}")
        else:
            print("✅ No open positions (or positions query successful)")
        
        moomoo.disconnect()
        print("\n✅ Connection test completed successfully!")
        return True
    else:
        print("❌ Failed to connect to Moomoo API")
        print("\nTroubleshooting:")
        print("1. Make sure OpenD is running on your machine")
        print("2. Check that OpenD is listening on the correct host/port")
        print("3. Verify your Moomoo credentials in .env file")
        print("4. Ensure you have OpenAPI access enabled in your Moomoo account")
        return False

if __name__ == '__main__':
    test_connection()


