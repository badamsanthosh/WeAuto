"""
Moomoo API Integration Module
Handles connection, market data, and trade execution
"""
try:
    import moomoo as ft
except ImportError:
    try:
        # Alternative package name
        from moomoo_api import moomoo as ft
    except ImportError:
        print("Warning: Moomoo package not found. Install with: pip install moomoo-api")
        ft = None

from typing import Dict, Optional, List
import time
import config

class MoomooIntegration:
    """Integration with Moomoo trading platform"""
    
    def __init__(self):
        self.quote_ctx = None
        self.trade_ctx = None
        self.connected = False
        
    def connect(self) -> bool:
        """
        Connect to Moomoo API via OpenD
        
        Returns:
            True if connection successful
        """
        if ft is None:
            print("Error: Moomoo package not installed. Please install with: pip install moomoo-api")
            return False
        
        try:
            # Initialize quote context
            self.quote_ctx = ft.OpenQuoteContext(host=config.MOOMOO_HOST, port=config.MOOMOO_PORT)
            ret, err = self.quote_ctx.start()
            
            if ret != ft.RET_OK:
                print(f"Failed to start quote context: {err}")
                return False
            
            # Initialize trade context
            self.trade_ctx = ft.OpenTradeContext(host=config.MOOMOO_HOST, port=config.MOOMOO_PORT)
            ret, err = self.trade_ctx.start()
            
            if ret != ft.RET_OK:
                print(f"Failed to start trade context: {err}")
                return False
            
            # Unlock trade (if required)
            if config.MOOMOO_USERNAME and config.MOOMOO_PASSWORD:
                ret, err = self.trade_ctx.unlock_trade(config.MOOMOO_PASSWORD)
                if ret != ft.RET_OK:
                    print(f"Warning: Failed to unlock trade: {err}")
            
            self.connected = True
            print("Successfully connected to Moomoo API")
            return True
            
        except Exception as e:
            print(f"Error connecting to Moomoo: {e}")
            print("Make sure OpenD is running on your machine")
            return False
    
    def disconnect(self):
        """Disconnect from Moomoo API"""
        try:
            if self.quote_ctx:
                self.quote_ctx.stop()
            if self.trade_ctx:
                self.trade_ctx.stop()
            self.connected = False
            print("Disconnected from Moomoo API")
        except Exception as e:
            print(f"Error disconnecting: {e}")
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current market price for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Current price or None if error
        """
        if not self.connected:
            return None
        
        try:
            # Convert ticker to Moomoo format (US stocks use format like "US.AAPL")
            moomoo_ticker = f"US.{ticker}" if not ticker.startswith("US.") else ticker
            
            ret, data = self.quote_ctx.get_market_snapshot([moomoo_ticker])
            
            if ret == ft.RET_OK and len(data) > 0:
                return float(data.iloc[0]['last_price'])
            else:
                return None
        except Exception as e:
            print(f"Error getting price for {ticker}: {e}")
            return None
    
    def get_account_info(self) -> Optional[Dict]:
        """
        Get account information
        
        Returns:
            Dictionary with account info
        """
        if not self.connected:
            return None
        
        try:
            # Determine trading environment
            trd_env = ft.TrdEnv.REAL if config.TRADING_ENV == 'REAL' else ft.TrdEnv.SIMULATE
            
            ret, data = self.trade_ctx.accinfo_query(trd_env=trd_env)
            
            if ret == ft.RET_OK:
                return data.to_dict() if hasattr(data, 'to_dict') else dict(data)
            else:
                return None
        except Exception as e:
            print(f"Error getting account info: {e}")
            return None
    
    def place_buy_order(self, ticker: str, quantity: int, price: float = 0, 
                       order_type: str = 'MARKET') -> Dict:
        """
        Place a buy order
        
        Args:
            ticker: Stock ticker symbol
            quantity: Number of shares
            price: Limit price (0 for market order)
            order_type: 'MARKET' or 'LIMIT'
            
        Returns:
            Dictionary with order result
        """
        if not self.connected:
            return {'success': False, 'error': 'Not connected to Moomoo'}
        
        try:
            moomoo_ticker = f"US.{ticker}" if not ticker.startswith("US.") else ticker
            trd_env = ft.TrdEnv.REAL if config.TRADING_ENV == 'REAL' else ft.TrdEnv.SIMULATE
            
            # Determine order type
            if order_type == 'MARKET':
                order_type_enum = ft.OrderType.MARKET
                price = 0  # Market orders don't use price
            else:
                order_type_enum = ft.OrderType.NORMAL
                if price <= 0:
                    return {'success': False, 'error': 'Price required for limit order'}
            
            ret, data = self.trade_ctx.place_order(
                price=price,
                qty=quantity,
                code=moomoo_ticker,
                order_type=order_type_enum,
                trd_side=ft.TrdSide.BUY,
                trd_env=trd_env
            )
            
            if ret == ft.RET_OK:
                return {'success': True, 'data': data}
            else:
                return {'success': False, 'error': str(data)}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def place_sell_order(self, ticker: str, quantity: int, price: float = 0,
                        order_type: str = 'MARKET') -> Dict:
        """
        Place a sell order
        
        Args:
            ticker: Stock ticker symbol
            quantity: Number of shares
            price: Limit price (0 for market order)
            order_type: 'MARKET' or 'LIMIT'
            
        Returns:
            Dictionary with order result
        """
        if not self.connected:
            return {'success': False, 'error': 'Not connected to Moomoo'}
        
        try:
            moomoo_ticker = f"US.{ticker}" if not ticker.startswith("US.") else ticker
            trd_env = ft.TrdEnv.REAL if config.TRADING_ENV == 'REAL' else ft.TrdEnv.SIMULATE
            
            if order_type == 'MARKET':
                order_type_enum = ft.OrderType.MARKET
                price = 0
            else:
                order_type_enum = ft.OrderType.NORMAL
                if price <= 0:
                    return {'success': False, 'error': 'Price required for limit order'}
            
            ret, data = self.trade_ctx.place_order(
                price=price,
                qty=quantity,
                code=moomoo_ticker,
                order_type=order_type_enum,
                trd_side=ft.TrdSide.SELL,
                trd_env=trd_env
            )
            
            if ret == ft.RET_OK:
                return {'success': True, 'data': data}
            else:
                return {'success': False, 'error': str(data)}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_positions(self) -> List[Dict]:
        """
        Get current positions
        
        Returns:
            List of position dictionaries
        """
        if not self.connected:
            return []
        
        try:
            trd_env = ft.TrdEnv.REAL if config.TRADING_ENV == 'REAL' else ft.TrdEnv.SIMULATE
            ret, data = self.trade_ctx.position_list_query(trd_env=trd_env)
            
            if ret == ft.RET_OK:
                if hasattr(data, 'to_dict'):
                    return data.to_dict('records')
                else:
                    return [dict(row) for row in data]
            else:
                return []
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    def cancel_order(self, order_id: str) -> Dict:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dictionary with cancellation result
        """
        if not self.connected:
            return {'success': False, 'error': 'Not connected to Moomoo'}
        
        try:
            trd_env = ft.TrdEnv.REAL if config.TRADING_ENV == 'REAL' else ft.TrdEnv.SIMULATE
            ret, data = self.trade_ctx.modify_order(
                modify_order_op=ft.ModifyOrderOp.CANCEL,
                order_id=order_id,
                trd_env=trd_env
            )
            
            if ret == ft.RET_OK:
                return {'success': True, 'data': data}
            else:
                return {'success': False, 'error': str(data)}
        except Exception as e:
            return {'success': False, 'error': str(e)}

