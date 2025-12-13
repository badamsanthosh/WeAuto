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
        
        # Distance from MAs (percentage)
        df['Distance_From_MA50'] = ((df['Close'] - df['SMA_50']) / df['SMA_50']) * 100
        df['Distance_From_MA250'] = ((df['Close'] - df['SMA_250']) / df['SMA_250']) * 100
        
        # MA Separation (strength of trend)
        df['MA_Separation'] = ((df['SMA_50'] - df['SMA_250']) / df['SMA_250']) * 100
        
        # MA Slopes (momentum indicators)
        df['MA50_Slope'] = df['SMA_50'].pct_change(5) * 100  # 5-day slope
        df['MA250_Slope'] = df['SMA_250'].pct_change(20) * 100  # 20-day slope
        
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
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
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
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price change indicators
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        
        # Intraday range
        df['Intraday_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Intraday_Gain'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # Target: 5% intraday gain
        df['Target_Hit'] = (df['Intraday_Gain'] >= 5.0).astype(int)
        
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
        Analyze intraday patterns from historical data
        
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


