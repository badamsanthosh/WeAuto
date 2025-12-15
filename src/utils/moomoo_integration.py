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

from typing import Dict, Optional, List, Tuple, Any
import time
import core.config as config

class MoomooIntegration:
    """Integration with Moomoo trading platform"""
    
    def __init__(self):
        self.quote_ctx = None
        self.trade_ctx = None
        self.connected = False
    
    def _unpack_api_result(self, result: Any) -> Tuple[Any, Any]:
        """
        Safely unpack Moomoo API result which may be in different formats
        
        Args:
            result: API call result (may be tuple, object, or None)
            
        Returns:
            Tuple of (ret_code, data) or (None, None) if invalid
        """
        try:
            if result is None:
                return None, None
            
            # Standard tuple format: (ret, data)
            if isinstance(result, tuple):
                if len(result) >= 2:
                    return result[0], result[1]
                elif len(result) == 1:
                    return result[0], None
                else:
                    return None, None
            
            # Object format with ret and data attributes
            if hasattr(result, 'ret') and hasattr(result, 'data'):
                return result.ret, result.data
            if hasattr(result, 'ret') and hasattr(result, 'err'):
                return result.ret, result.err
            
            # Single value (assume it's a return code)
            return result, None
        except Exception as e:
            # If unpacking fails, return None values
            print(f"Warning: Error unpacking API result: {e}, type: {type(result)}")
            return None, None
        
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
            # Check if RET_OK constant exists
            RET_OK = getattr(ft, 'RET_OK', 0)  # Default to 0 if not found
            
            # Initialize quote context
            try:
                self.quote_ctx = ft.OpenQuoteContext(host=config.MOOMOO_HOST, port=config.MOOMOO_PORT)
            except Exception as e:
                print(f"Error creating quote context: {e}")
                return False
            
            # Start quote context - handle different API return formats
            try:
                start_result = self.quote_ctx.start()
                
                # If start() returns None, it might be a void function - check for exceptions instead
                if start_result is None:
                    # Assume success if no exception was raised
                    print("Quote context started (no return value)")
                else:
                    ret, err = self._unpack_api_result(start_result)
                    
                    if ret is not None and ret != RET_OK:
                        print(f"Failed to start quote context: {err if err else 'Unknown error'}")
                        return False
            except Exception as start_err:
                # If start() raises an exception, connection failed
                print(f"Error starting quote context: {start_err}")
                return False
            
            # Initialize trade context
            try:
                self.trade_ctx = ft.OpenTradeContext(host=config.MOOMOO_HOST, port=config.MOOMOO_PORT)
            except Exception as e:
                print(f"Error creating trade context: {e}")
                return False
            
            # Start trade context - handle different API return formats
            try:
                start_result = self.trade_ctx.start()
                
                # If start() returns None, it might be a void function - check for exceptions instead
                if start_result is None:
                    # Assume success if no exception was raised
                    print("Trade context started (no return value)")
                else:
                    ret, err = self._unpack_api_result(start_result)
                    
                    if ret is not None and ret != RET_OK:
                        print(f"Failed to start trade context: {err if err else 'Unknown error'}")
                        return False
            except Exception as start_err:
                # If start() raises an exception, connection failed
                print(f"Error starting trade context: {start_err}")
                return False
            
            # Unlock trade (if required)
            if config.MOOMOO_USERNAME and config.MOOMOO_PASSWORD:
                try:
                    unlock_result = self.trade_ctx.unlock_trade(config.MOOMOO_PASSWORD)
                    if unlock_result is not None:
                        ret, err = self._unpack_api_result(unlock_result)
                        if ret is not None and ret != RET_OK:
                            print(f"Warning: Failed to unlock trade: {err if err else 'Unknown error'}")
                except Exception as unlock_err:
                    print(f"Warning: Could not unlock trade (may not be required): {unlock_err}")
            
            self.connected = True
            print("Successfully connected to Moomoo API")
            return True
            
        except Exception as e:
            print(f"Error connecting to Moomoo: {e}")
            print("Make sure OpenD is running on your machine")
            import traceback
            traceback.print_exc()
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
        if not self.connected or self.quote_ctx is None:
            return None
        
        try:
            # Check if RET_OK constant exists
            RET_OK = getattr(ft, 'RET_OK', 0) if ft else 0
            
            # Convert ticker to Moomoo format (US stocks use format like "US.AAPL")
            moomoo_ticker = f"US.{ticker}" if not ticker.startswith("US.") else ticker
            
            result = self.quote_ctx.get_market_snapshot([moomoo_ticker])
            
            if result is None:
                return None
            
            ret, data = self._unpack_api_result(result)
            
            if ret == RET_OK and data is not None and len(data) > 0:
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
        if not self.connected or self.trade_ctx is None:
            return None
        
        try:
            # Check if RET_OK constant exists
            RET_OK = getattr(ft, 'RET_OK', 0) if ft else 0
            
            # Determine trading environment
            trd_env = ft.TrdEnv.REAL if config.TRADING_ENV == 'REAL' else ft.TrdEnv.SIMULATE
            
            result = self.trade_ctx.accinfo_query(trd_env=trd_env)
            
            if result is None:
                return None
            
            ret, data = self._unpack_api_result(result)
            
            if ret == RET_OK and data is not None:
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
            # Check if RET_OK constant exists
            RET_OK = getattr(ft, 'RET_OK', 0) if ft else 0
            
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
            
            result = self.trade_ctx.place_order(
                price=price,
                qty=quantity,
                code=moomoo_ticker,
                order_type=order_type_enum,
                trd_side=ft.TrdSide.BUY,
                trd_env=trd_env
            )
            
            if result is None:
                return {'success': False, 'error': 'Order placement returned None'}
            
            ret, data = self._unpack_api_result(result)
            
            if ret == RET_OK:
                return {'success': True, 'data': data}
            else:
                return {'success': False, 'error': str(data) if data else 'Unknown error'}
                
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
            # Check if RET_OK constant exists
            RET_OK = getattr(ft, 'RET_OK', 0) if ft else 0
            
            moomoo_ticker = f"US.{ticker}" if not ticker.startswith("US.") else ticker
            trd_env = ft.TrdEnv.REAL if config.TRADING_ENV == 'REAL' else ft.TrdEnv.SIMULATE
            
            if order_type == 'MARKET':
                order_type_enum = ft.OrderType.MARKET
                price = 0
            else:
                order_type_enum = ft.OrderType.NORMAL
                if price <= 0:
                    return {'success': False, 'error': 'Price required for limit order'}
            
            result = self.trade_ctx.place_order(
                price=price,
                qty=quantity,
                code=moomoo_ticker,
                order_type=order_type_enum,
                trd_side=ft.TrdSide.SELL,
                trd_env=trd_env
            )
            
            if result is None:
                return {'success': False, 'error': 'Order placement returned None'}
            
            ret, data = self._unpack_api_result(result)
            
            if ret == RET_OK:
                return {'success': True, 'data': data}
            else:
                return {'success': False, 'error': str(data) if data else 'Unknown error'}
                
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
            # Check if RET_OK constant exists
            RET_OK = getattr(ft, 'RET_OK', 0) if ft else 0
            
            trd_env = ft.TrdEnv.REAL if config.TRADING_ENV == 'REAL' else ft.TrdEnv.SIMULATE
            result = self.trade_ctx.position_list_query(trd_env=trd_env)
            
            if result is None:
                return []
            
            ret, data = self._unpack_api_result(result)
            
            if ret == RET_OK and data is not None:
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
            # Check if RET_OK constant exists
            RET_OK = getattr(ft, 'RET_OK', 0) if ft else 0
            
            trd_env = ft.TrdEnv.REAL if config.TRADING_ENV == 'REAL' else ft.TrdEnv.SIMULATE
            result = self.trade_ctx.modify_order(
                modify_order_op=ft.ModifyOrderOp.CANCEL,
                order_id=order_id,
                trd_env=trd_env
            )
            
            if result is None:
                return {'success': False, 'error': 'Cancel order returned None'}
            
            ret, data = self._unpack_api_result(result)
            
            if ret == RET_OK:
                return {'success': True, 'data': data}
            else:
                return {'success': False, 'error': str(data) if data else 'Unknown error'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

