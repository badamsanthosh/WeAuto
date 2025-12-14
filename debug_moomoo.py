"""
Debug script to diagnose Moomoo API connection issues
"""
import sys

print("=" * 60)
print("Moomoo API Debug Script")
print("=" * 60)

# Step 1: Check if moomoo module can be imported
print("\n1. Checking Moomoo module import...")
try:
    import moomoo as ft
    print("✅ Moomoo module imported successfully")
    print(f"   Module location: {ft.__file__ if hasattr(ft, '__file__') else 'N/A'}")
except ImportError:
    try:
        from moomoo_api import moomoo as ft
        print("✅ Moomoo module imported from moomoo_api")
    except ImportError:
        print("❌ ERROR: Cannot import moomoo module")
        print("   Please install: pip install moomoo-api")
        sys.exit(1)

# Step 2: Check for required constants
print("\n2. Checking for required constants...")
constants_to_check = ['RET_OK', 'TrdEnv', 'OrderType', 'TrdSide', 'ModifyOrderOp']
for const in constants_to_check:
    if hasattr(ft, const):
        print(f"   ✅ {const} found")
        try:
            value = getattr(ft, const)
            if const == 'RET_OK':
                print(f"      Value: {value}")
            elif const == 'TrdEnv':
                print(f"      REAL: {value.REAL if hasattr(value, 'REAL') else 'N/A'}")
                print(f"      SIMULATE: {value.SIMULATE if hasattr(value, 'SIMULATE') else 'N/A'}")
        except Exception as e:
            print(f"      ⚠️  Error accessing: {e}")
    else:
        print(f"   ❌ {const} NOT found")

# Step 3: Check OpenD connection
print("\n3. Testing OpenD connection...")
import config

try:
    quote_ctx = ft.OpenQuoteContext(host=config.MOOMOO_HOST, port=config.MOOMOO_PORT)
    print(f"   ✅ QuoteContext created (host={config.MOOMOO_HOST}, port={config.MOOMOO_PORT})")
    
    # Try to start
    print("   Attempting to start quote context...")
    try:
        result = quote_ctx.start()
        print(f"   ✅ start() returned: {result}")
        print(f"      Type: {type(result)}")
        
        if result is None:
            print("      ⚠️  start() returned None (might be void function)")
        elif isinstance(result, tuple):
            print(f"      ✅ Tuple with {len(result)} elements")
            for i, item in enumerate(result):
                print(f"         [{i}]: {item} (type: {type(item)})")
        elif hasattr(result, '__dict__'):
            print(f"      ✅ Object with attributes: {dir(result)}")
        else:
            print(f"      ⚠️  Unexpected type: {type(result)}")
            
    except Exception as e:
        print(f"   ❌ Error calling start(): {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"   ❌ Error creating QuoteContext: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Test TradeContext
print("\n4. Testing TradeContext...")
try:
    trade_ctx = ft.OpenTradeContext(host=config.MOOMOO_HOST, port=config.MOOMOO_PORT)
    print(f"   ✅ TradeContext created")
    
    # Try to start
    print("   Attempting to start trade context...")
    try:
        result = trade_ctx.start()
        print(f"   ✅ start() returned: {result}")
        print(f"      Type: {type(result)}")
        
        if result is None:
            print("      ⚠️  start() returned None (might be void function)")
        elif isinstance(result, tuple):
            print(f"      ✅ Tuple with {len(result)} elements")
            for i, item in enumerate(result):
                print(f"         [{i}]: {item} (type: {type(item)})")
        elif hasattr(result, '__dict__'):
            print(f"      ✅ Object with attributes: {dir(result)}")
        else:
            print(f"      ⚠️  Unexpected type: {type(result)}")
            
    except Exception as e:
        print(f"   ❌ Error calling start(): {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"   ❌ Error creating TradeContext: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Debug complete!")
print("=" * 60)

