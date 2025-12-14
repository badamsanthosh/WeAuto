"""
Data Analysis Module for Historical Market Data Analysis
Analyzes 40 years of market data to identify patterns and trends
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import pickle
from typing import List, Dict, Optional
import warnings
import core.config as config
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Analyzes historical market data to identify trading patterns"""
    
    def __init__(self, cache_dir: str = 'data_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_historical_data(self, ticker: str, years: int = 40) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a ticker
        
        Args:
            ticker: Stock ticker symbol
            years: Number of years of historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_file = os.path.join(self.cache_dir, f"{ticker}_historical.pkl")
        
        # Check cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    # Check if data is recent (within 1 day)
                    if len(data) > 0 and (datetime.now() - data.index[-1]).days < 1:
                        return data
            except:
                pass
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, interval='1d')
            
            if data.empty:
                return None
                
            # Cache the data
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = data.copy()
        
        # Moving Averages - PRIMARY STRATEGY: 50-day and 250-day
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()  # Fast MA
        df['SMA_250'] = df['Close'].rolling(window=250).mean()  # Slow MA (primary)
        df['SMA_200'] = df['Close'].rolling(window=200).mean()  # Keep for compatibility
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # Moving Average Strategy Indicators
        # Golden Cross: 50MA > 250MA (bullish signal)
        df['MA_Golden_Cross'] = (df['SMA_50'] > df['SMA_250']).astype(int)
        
        # Price position relative to MAs
        df['Price_Above_MA50'] = (df['Close'] > df['SMA_50']).astype(int)
        df['Price_Above_MA250'] = (df['Close'] > df['SMA_250']).astype(int)
        
        # Distance from MAs (percentage) - handle division by zero
        df['Distance_From_MA50'] = np.where(
            df['SMA_50'] != 0,
            ((df['Close'] - df['SMA_50']) / df['SMA_50']) * 100,
            np.nan
        )
        df['Distance_From_MA250'] = np.where(
            df['SMA_250'] != 0,
            ((df['Close'] - df['SMA_250']) / df['SMA_250']) * 100,
            np.nan
        )
        
        # MA Separation (strength of trend) - handle division by zero
        df['MA_Separation'] = np.where(
            df['SMA_250'] != 0,
            ((df['SMA_50'] - df['SMA_250']) / df['SMA_250']) * 100,
            np.nan
        )
        
        # MA Slopes (momentum indicators) - replace inf with NaN
        df['MA50_Slope'] = df['SMA_50'].pct_change(5) * 100  # 5-day slope
        df['MA50_Slope'] = df['MA50_Slope'].replace([np.inf, -np.inf], np.nan)
        
        df['MA250_Slope'] = df['SMA_250'].pct_change(20) * 100  # 20-day slope
        df['MA250_Slope'] = df['MA250_Slope'].replace([np.inf, -np.inf], np.nan)
        
        # Recent crossover detection
        df['MA_Crossover_Recent'] = 0
        for i in range(1, min(10, len(df))):
            if (df['SMA_50'].iloc[i] > df['SMA_250'].iloc[i] and 
                df['SMA_50'].iloc[i-1] <= df['SMA_250'].iloc[i-1]):
                df.loc[df.index[i], 'MA_Crossover_Recent'] = 1
        
        # MA Trend Strength (combination of all factors)
        df['MA_Trend_Strength'] = (
            df['MA_Golden_Cross'] * 0.3 +
            df['Price_Above_MA50'] * 0.2 +
            df['Price_Above_MA250'] * 0.2 +
            (df['MA_Separation'] > 1.0).astype(int) * 0.15 +
            (df['MA50_Slope'] > 0).astype(int) * 0.15
        )
        
        # RSI - handle division by zero
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = np.where(loss != 0, gain / loss, np.nan)
        df['RSI'] = np.where(
            ~np.isnan(rs) & (rs != -1),
            100 - (100 / (1 + rs)),
            np.nan
        )
        
        # RSI Overbought/Oversold signals
        df['RSI_Overbought'] = (df['RSI'] >= config.RSI_OVERBOUGHT).astype(int)
        df['RSI_Oversold'] = (df['RSI'] <= config.RSI_OVERSOLD).astype(int)
        df['RSI_Neutral'] = ((df['RSI'] > config.RSI_OVERSOLD) & 
                            (df['RSI'] < config.RSI_OVERBOUGHT)).astype(int)
        
        # RSI Signal (for intraday trading)
        # Oversold = potential buy, Overbought = potential sell
        df['RSI_Signal'] = np.where(
            df['RSI'] <= config.RSI_OVERSOLD, 'OVERSOLD',
            np.where(df['RSI'] >= config.RSI_OVERBOUGHT, 'OVERBOUGHT', 'NEUTRAL')
        )
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        
        # BB_Position - handle division by zero
        bb_range = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = np.where(
            bb_range != 0,
            (df['Close'] - df['BB_Lower']) / bb_range,
            np.nan
        )
        df['BB_Position'] = df['BB_Position'].replace([np.inf, -np.inf], np.nan)
        
        # Volume indicators - handle division by zero
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = np.where(
            df['Volume_SMA'] != 0,
            df['Volume'] / df['Volume_SMA'],
            np.nan
        )
        df['Volume_Ratio'] = df['Volume_Ratio'].replace([np.inf, -np.inf], np.nan)
        
        # Price change indicators - replace inf with NaN
        df['Daily_Return'] = df['Close'].pct_change()
        df['Daily_Return'] = df['Daily_Return'].replace([np.inf, -np.inf], np.nan)
        
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        df['Volatility'] = df['Volatility'].replace([np.inf, -np.inf], np.nan)
        
        # Intraday range - handle division by zero (kept for reference)
        df['Intraday_Range'] = np.where(
            df['Close'] != 0,
            (df['High'] - df['Low']) / df['Close'],
            np.nan
        )
        
        df['Intraday_Gain'] = np.where(
            df['Open'] != 0,
            (df['Close'] - df['Open']) / df['Open'] * 100,
            np.nan
        )
        df['Intraday_Gain'] = df['Intraday_Gain'].replace([np.inf, -np.inf], np.nan)
        
        # === WEEKLY TRADING METRICS ===
        # Weekly gain (5-day forward return for swing trading)
        df['Weekly_Gain'] = df['Close'].pct_change(5).shift(-5) * 100
        df['Weekly_Gain'] = df['Weekly_Gain'].replace([np.inf, -np.inf], np.nan)
        
        # Weekly high/low range
        df['Weekly_High'] = df['High'].rolling(window=5).max()
        df['Weekly_Low'] = df['Low'].rolling(window=5).min()
        df['Weekly_Range'] = np.where(
            df['Close'] != 0,
            (df['Weekly_High'] - df['Weekly_Low']) / df['Close'] * 100,
            np.nan
        )
        
        # Weekly volatility (5-day rolling std)
        df['Weekly_Volatility'] = df['Daily_Return'].rolling(window=5).std() * np.sqrt(5) * 100
        df['Weekly_Volatility'] = df['Weekly_Volatility'].replace([np.inf, -np.inf], np.nan)
        
        # Weekly momentum indicators
        df['Weekly_Momentum'] = df['Close'].pct_change(5) * 100
        df['Weekly_Momentum'] = df['Weekly_Momentum'].replace([np.inf, -np.inf], np.nan)
        
        # Weekly RSI (calculated over 5 days for swing trading)
        weekly_delta = df['Close'].diff()
        weekly_gain = (weekly_delta.where(weekly_delta > 0, 0)).rolling(window=5).mean()
        weekly_loss = (-weekly_delta.where(weekly_delta < 0, 0)).rolling(window=5).mean()
        weekly_rs = np.where(weekly_loss != 0, weekly_gain / weekly_loss, np.nan)
        df['Weekly_RSI'] = np.where(
            ~np.isnan(weekly_rs) & (weekly_rs != -1),
            100 - (100 / (1 + weekly_rs)),
            np.nan
        )
        
        # Weekly volume trend
        df['Weekly_Volume_Avg'] = df['Volume'].rolling(window=5).mean()
        df['Weekly_Volume_Trend'] = np.where(
            df['Weekly_Volume_Avg'].shift(1) != 0,
            (df['Weekly_Volume_Avg'] - df['Weekly_Volume_Avg'].shift(1)) / df['Weekly_Volume_Avg'].shift(1) * 100,
            np.nan
        )
        
        # Swing trading strength (combination of weekly indicators)
        df['Swing_Strength'] = (
            (df['Weekly_Momentum'] > 0).astype(int) * 0.3 +
            ((df['Weekly_RSI'] > 30) & (df['Weekly_RSI'] < 70)).astype(int) * 0.3 +
            (df['Weekly_Volume_Trend'] > 0).astype(int) * 0.2 +
            (df['MA_Golden_Cross'] == 1).astype(int) * 0.2
        )
        
        # Target: 5-10% weekly gain for swing trading
        df['Target_Hit'] = ((df['Weekly_Gain'] >= config.TARGET_GAIN_PERCENT_MIN) & 
                           (df['Weekly_Gain'] <= config.TARGET_GAIN_PERCENT_MAX * 1.5)).astype(int)
        df['Target_Hit_Min'] = (df['Weekly_Gain'] >= config.TARGET_GAIN_PERCENT_MIN).astype(int)
        df['Target_Hit_Max'] = (df['Weekly_Gain'] >= config.TARGET_GAIN_PERCENT_MAX).astype(int)
        
        return df
    
    def analyze_market_trends(self, tickers: List[str], years: int = 40) -> Dict:
        """
        Analyze overall market trends from major indices
        
        Args:
            tickers: List of index tickers (e.g., SPY, QQQ, DIA)
            years: Number of years to analyze
            
        Returns:
            Dictionary with market trend analysis
        """
        market_data = {}
        
        for ticker in tickers:
            data = self.get_historical_data(ticker, years)
            if data is not None and not data.empty:
                data = self.calculate_technical_indicators(data)
                market_data[ticker] = {
                    'current_price': data['Close'].iloc[-1],
                    'sma_50': data['SMA_50'].iloc[-1] if not pd.isna(data['SMA_50'].iloc[-1]) else None,
                    'sma_250': data['SMA_250'].iloc[-1] if not pd.isna(data['SMA_250'].iloc[-1]) else None,
                    'sma_200': data['SMA_200'].iloc[-1] if not pd.isna(data['SMA_200'].iloc[-1]) else None,
                    'golden_cross': bool(data['MA_Golden_Cross'].iloc[-1]) if 'MA_Golden_Cross' in data.columns and not pd.isna(data['MA_Golden_Cross'].iloc[-1]) else False,
                    'trend': 'BULL' if (not pd.isna(data['SMA_250'].iloc[-1]) and 
                                       data['Close'].iloc[-1] > data['SMA_250'].iloc[-1] and 
                                       not pd.isna(data['SMA_50'].iloc[-1]) and
                                       data['SMA_50'].iloc[-1] > data['SMA_250'].iloc[-1]) else 'BEAR',
                    'ma_separation': data['MA_Separation'].iloc[-1] if 'MA_Separation' in data.columns and not pd.isna(data['MA_Separation'].iloc[-1]) else None,
                    'volatility': data['Volatility'].iloc[-1] if not pd.isna(data['Volatility'].iloc[-1]) else None,
                    'rsi': data['RSI'].iloc[-1] if not pd.isna(data['RSI'].iloc[-1]) else None,
                }
        
        return market_data
    
    def get_intraday_patterns(self, data: pd.DataFrame) -> Dict:
        """
        Analyze intraday patterns from historical data (legacy method, kept for compatibility)
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            
        Returns:
            Dictionary with pattern statistics
        """
        if data.empty:
            return {}
        
        # Filter for days with 5%+ intraday gains
        big_gain_days = data[data['Intraday_Gain'] >= 5.0]
        
        if len(big_gain_days) == 0:
            return {}
        
        patterns = {
            'total_big_gain_days': len(big_gain_days),
            'frequency': len(big_gain_days) / len(data),
            'avg_rsi_before': big_gain_days['RSI'].shift(1).mean(),
            'avg_volume_ratio': big_gain_days['Volume_Ratio'].mean(),
            'avg_volatility': big_gain_days['Volatility'].mean(),
            'avg_bb_position': big_gain_days['BB_Position'].shift(1).mean(),
            'macd_bullish_ratio': (big_gain_days['MACD'].shift(1) > big_gain_days['MACD_Signal'].shift(1)).mean(),
        }
        
        return patterns
    
    def get_weekly_patterns(self, data: pd.DataFrame) -> Dict:
        """
        Analyze weekly gain patterns from historical data (for swing trading)
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            
        Returns:
            Dictionary with weekly pattern statistics
        """
        if data.empty:
            return {}
        
        # Filter for periods with 5%+ weekly gains
        big_gain_weeks = data[data['Weekly_Gain'] >= config.TARGET_GAIN_PERCENT_MIN]
        
        # Filter for periods with 10%+ weekly gains
        excellent_gain_weeks = data[data['Weekly_Gain'] >= config.TARGET_GAIN_PERCENT_MAX]
        
        if len(big_gain_weeks) == 0:
            return {}
        
        patterns = {
            'total_big_gain_weeks': len(big_gain_weeks),
            'total_excellent_gain_weeks': len(excellent_gain_weeks),
            'frequency': len(big_gain_weeks) / len(data),
            'excellent_frequency': len(excellent_gain_weeks) / len(data) if len(data) > 0 else 0,
            'avg_weekly_gain': big_gain_weeks['Weekly_Gain'].mean(),
            'max_weekly_gain': big_gain_weeks['Weekly_Gain'].max(),
            'avg_weekly_volatility': big_gain_weeks['Weekly_Volatility'].mean(),
            'avg_rsi_before': big_gain_weeks['RSI'].shift(5).mean(),
            'avg_weekly_rsi': big_gain_weeks['Weekly_RSI'].mean(),
            'avg_volume_ratio': big_gain_weeks['Volume_Ratio'].mean(),
            'avg_weekly_momentum': big_gain_weeks['Weekly_Momentum'].shift(5).mean(),
            'avg_swing_strength': big_gain_weeks['Swing_Strength'].mean(),
            'golden_cross_ratio': (big_gain_weeks['MA_Golden_Cross'] == 1).mean(),
            'avg_bb_position': big_gain_weeks['BB_Position'].shift(5).mean(),
            'macd_bullish_ratio': (big_gain_weeks['MACD'].shift(5) > big_gain_weeks['MACD_Signal'].shift(5)).mean(),
        }
        
        return patterns


