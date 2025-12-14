"""
Risk Management Module
Handles position sizing, stop-loss, take-profit, and risk controls
"""
from typing import Dict, Optional
import core.config as config

class RiskManager:
    """Manages trading risk and position sizing"""
    
    def __init__(self):
        self.max_position_size = config.MAX_POSITION_SIZE
        self.stop_loss_percent = config.STOP_LOSS_PERCENT
        self.take_profit_percent = config.TAKE_PROFIT_PERCENT
        self.max_positions = config.MAX_POSITIONS
        
    def calculate_position_size(self, account_balance: float, stock_price: float, 
                               confidence: float) -> int:
        """
        Calculate appropriate position size based on risk management rules
        
        Args:
            account_balance: Total account balance
            stock_price: Current stock price
            confidence: Confidence score (0-1)
            
        Returns:
            Number of shares to buy
        """
        # Base position size as percentage of account
        base_allocation = 0.1  # 10% of account per position
        
        # Adjust based on confidence
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5x to 1.0x
        
        # Calculate dollar amount
        dollar_amount = account_balance * base_allocation * confidence_multiplier
        
        # Cap at max position size
        dollar_amount = min(dollar_amount, self.max_position_size)
        
        # Calculate shares
        shares = int(dollar_amount / stock_price)
        
        # Minimum 1 share
        return max(1, shares)
    
    def calculate_stop_loss_price(self, entry_price: float) -> float:
        """
        Calculate stop-loss price
        
        Args:
            entry_price: Entry price of the position
            
        Returns:
            Stop-loss price
        """
        return entry_price * (1 - self.stop_loss_percent / 100)
    
    def calculate_take_profit_price(self, entry_price: float, target_gain: float = None) -> float:
        """
        Calculate take-profit price
        
        Args:
            entry_price: Entry price of the position
            target_gain: Target gain percentage (defaults to config value)
            
        Returns:
            Take-profit price
        """
        if target_gain is None:
            target_gain = self.take_profit_percent
        return entry_price * (1 + target_gain / 100)
    
    def validate_trade(self, ticker: str, price: float, quantity: int, 
                      account_balance: float, current_positions: int) -> Dict:
        """
        Validate if a trade meets risk management criteria
        
        Args:
            ticker: Stock ticker
            price: Stock price
            quantity: Number of shares
            account_balance: Account balance
            current_positions: Current number of open positions
            
        Returns:
            Dictionary with validation result
        """
        # Check max positions
        if current_positions >= self.max_positions:
            return {
                'valid': False,
                'reason': f'Maximum positions ({self.max_positions}) already reached'
            }
        
        # Check position size
        position_value = price * quantity
        if position_value > self.max_position_size:
            return {
                'valid': False,
                'reason': f'Position size (${position_value:.2f}) exceeds maximum (${self.max_position_size:.2f})'
            }
        
        # Check if we have enough buying power
        if position_value > account_balance * 0.2:  # Don't use more than 20% of balance
            return {
                'valid': False,
                'reason': 'Insufficient buying power'
            }
        
        # Check price range
        if price < config.MIN_PRICE:
            return {
                'valid': False,
                'reason': f'Price (${price:.2f}) below minimum (${config.MIN_PRICE:.2f})'
            }
        
        if price > config.MAX_PRICE:
            return {
                'valid': False,
                'reason': f'Price (${price:.2f}) above maximum (${config.MAX_PRICE:.2f})'
            }
        
        return {'valid': True}
    
    def should_exit_position(self, entry_price: float, current_price: float) -> Dict:
        """
        Determine if a position should be exited based on stop-loss or take-profit
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            
        Returns:
            Dictionary with exit recommendation
        """
        stop_loss_price = self.calculate_stop_loss_price(entry_price)
        take_profit_price = self.calculate_take_profit_price(entry_price)
        
        # Check stop-loss
        if current_price <= stop_loss_price:
            return {
                'should_exit': True,
                'reason': 'STOP_LOSS',
                'entry_price': entry_price,
                'current_price': current_price,
                'loss_percent': ((current_price - entry_price) / entry_price) * 100
            }
        
        # Check take-profit
        if current_price >= take_profit_price:
            return {
                'should_exit': True,
                'reason': 'TAKE_PROFIT',
                'entry_price': entry_price,
                'current_price': current_price,
                'gain_percent': ((current_price - entry_price) / entry_price) * 100
            }
        
        return {'should_exit': False}


