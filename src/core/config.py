"""
Configuration file for automated trading system
WEEKLY TRADING STRATEGY - 2 trades per week (1 buy, 1 sell)
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Trading Configuration - WEEKLY STRATEGY
TRADING_TIMEFRAME = 'WEEKLY'  # WEEKLY trading strategy
TARGET_GAIN_PERCENT_MIN = 5.0  # Minimum target weekly gain
TARGET_GAIN_PERCENT_MAX = 10.0  # Maximum target weekly gain
TARGET_GAIN_PERCENT = 7.5  # Default target weekly gain (midpoint)
MAX_POSITIONS = 2  # Maximum number of positions to hold (1 buy + 1 sell per week)
TRADES_PER_WEEK = 2  # Number of trades per week (1 buy, 1 sell)
MIN_VOLUME = 100000  # Minimum daily volume (in dollars)
MIN_PRICE = 0.5  # Minimum stock price
MAX_PRICE = 50000.0  # Maximum stock price

# Risk Management - WEEKLY TIMEFRAME
STOP_LOSS_PERCENT = 3.0  # Stop loss percentage (wider for weekly trades)
TAKE_PROFIT_PERCENT = 7.5  # Take profit percentage (5-10% range)
TAKE_PROFIT_MIN = 5.0  # Minimum take profit (exit at 5% minimum)
TAKE_PROFIT_MAX = 10.0  # Maximum take profit (target 10% max)
MAX_POSITION_SIZE = 50000  # Maximum position size in dollars
HOLDING_PERIOD_DAYS = 5  # Target holding period (1 week = 5 trading days)

# Moomoo API Configuration
MOOMOO_HOST = os.getenv('MOOMOO_HOST', '127.0.0.1')
MOOMOO_PORT = int(os.getenv('MOOMOO_PORT', 11111))
MOOMOO_USERNAME = os.getenv('MOOMOO_USERNAME', '')
MOOMOO_PASSWORD = os.getenv('MOOMOO_PASSWORD', '')

# Trading Environment (REAL or SIMULATE)
TRADING_ENV = os.getenv('TRADING_ENV', 'SIMULATE')  # Use SIMULATE for testing

# Data Configuration
HISTORICAL_YEARS = 40  # Years of historical data to analyze
DATA_CACHE_DIR = 'data_cache'

# Stock Universe
# Focus on liquid, high-volume stocks
POPULAR_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVTS', 'AMTM', 'TALK'  # ETFs for market analysis
]

# Moving Average Strategy Configuration (PRIMARY STRATEGY)
MA_FAST = 50  # 50-day moving average
MA_SLOW = 250  # 250-day moving average
MA_STRATEGY_ENABLED = True  # Enable MA-based filtering
REQUIRE_GOLDEN_CROSS = True  # Require 50MA > 250MA (golden cross)
REQUIRE_PRICE_ABOVE_MA50 = True  # Require price above 50MA
REQUIRE_PRICE_ABOVE_MA250 = False  # Require price above 250MA (optional, less strict)
MA_CROSSOVER_LOOKBACK = 5  # Days to look back for recent crossover
MIN_MA_SEPARATION_PERCENT = 1.0  # Minimum separation between MAs (1% = bullish strength)

# Technical Indicators
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2

# ML Model Configuration
MODEL_TYPE = 'xgboost'  # 'xgboost', 'random_forest', 'gradient_boosting'
MIN_CONFIDENCE_SCORE = 0.7  # Minimum confidence score to consider a trade

# Approval Settings
REQUIRE_APPROVAL = True  # Require manual approval before executing trades
APPROVAL_TIMEOUT = 300  # Timeout in seconds for approval

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'trading_log.log'

# iMessage Notification Configuration
IMESSAGE_ENABLED = os.getenv('IMESSAGE_ENABLED', 'False').lower() == 'true'
IMESSAGE_APPLE_ID = os.getenv('IMESSAGE_APPLE_ID', '')  # Apple ID (email) for iMessage
IMESSAGE_PHONE = os.getenv('IMESSAGE_PHONE', '')  # Phone number(s) for iMessage - comma-separated list (e.g., +1234567890,+19876543210)
# Note: IMESSAGE_PHONE can accept multiple phone numbers separated by commas.
# Phone numbers take precedence over Apple ID if both are set.


