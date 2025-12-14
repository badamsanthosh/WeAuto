"""
Volatility and Volume Analysis Module
Analyzes weekly volatility and volume for swing trading
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

from core import config

class VolatilityAnalyzer:
    """Analyzes volatility and volume for weekly swing trading"""
    
    def __init__(self):
        pass
    
    def calculate_intraday_volatility(self, ticker: str, days: int = 30) -> Optional[Dict]:
        """
        Calculate intraday volatility metrics (legacy method for compatibility)
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to analyze
            
        Returns:
            Dictionary with volatility metrics
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f'{days}d', interval='1d')
            
            if hist.empty or len(hist) < 5:
                return None
            
            # Calculate intraday range (High - Low) / Close
            hist['Intraday_Range'] = (hist['High'] - hist['Low']) / hist['Close']
            hist['Intraday_Range_Pct'] = hist['Intraday_Range'] * 100
            
            # Average intraday volatility
            avg_intraday_vol = hist['Intraday_Range_Pct'].mean()
            max_intraday_vol = hist['Intraday_Range_Pct'].max()
            min_intraday_vol = hist['Intraday_Range_Pct'].min()
            
            # Volatility consistency (lower std = more consistent)
            vol_std = hist['Intraday_Range_Pct'].std()
            
            # Recent volatility (last 5 days)
            recent_vol = hist['Intraday_Range_Pct'].tail(5).mean()
            
            # Price volatility (daily returns)
            hist['Daily_Return'] = hist['Close'].pct_change()
            price_volatility = hist['Daily_Return'].std() * np.sqrt(252) * 100  # Annualized
            
            return {
                'ticker': ticker,
                'avg_intraday_volatility': avg_intraday_vol,
                'max_intraday_volatility': max_intraday_vol,
                'min_intraday_volatility': min_intraday_vol,
                'volatility_consistency': vol_std,
                'recent_volatility': recent_vol,
                'price_volatility': price_volatility,
                'volatility_score': self._calculate_volatility_score(
                    avg_intraday_vol, recent_vol, vol_std
                )
            }
        except Exception as e:
            print(f"Error calculating volatility for {ticker}: {e}")
            return None
    
    def calculate_weekly_volatility(self, ticker: str, weeks: int = 12) -> Optional[Dict]:
        """
        Calculate weekly volatility metrics for swing trading
        
        Args:
            ticker: Stock ticker symbol
            weeks: Number of weeks to analyze (default 12 weeks = ~3 months)
            
        Returns:
            Dictionary with weekly volatility metrics
        """
        try:
            days = weeks * 7
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f'{days}d', interval='1d')
            
            if hist.empty or len(hist) < 10:
                return None
            
            # Calculate weekly ranges (5-day rolling)
            hist['Weekly_High'] = hist['High'].rolling(window=5).max()
            hist['Weekly_Low'] = hist['Low'].rolling(window=5).min()
            hist['Weekly_Range'] = (hist['Weekly_High'] - hist['Weekly_Low']) / hist['Close']
            hist['Weekly_Range_Pct'] = hist['Weekly_Range'] * 100
            
            # Weekly returns (5-day)
            hist['Weekly_Return'] = hist['Close'].pct_change(5) * 100
            
            # Average weekly volatility
            avg_weekly_vol = hist['Weekly_Range_Pct'].mean()
            max_weekly_vol = hist['Weekly_Range_Pct'].max()
            min_weekly_vol = hist['Weekly_Range_Pct'].min()
            
            # Weekly return volatility
            weekly_return_vol = hist['Weekly_Return'].std()
            
            # Volatility consistency
            vol_std = hist['Weekly_Range_Pct'].std()
            
            # Recent weekly volatility (last 2 weeks)
            recent_weekly_vol = hist['Weekly_Range_Pct'].tail(10).mean()
            
            # Price volatility (weekly basis)
            hist['Daily_Return'] = hist['Close'].pct_change()
            price_volatility = hist['Daily_Return'].rolling(window=5).std().mean() * np.sqrt(52) * 100  # Weekly annualized
            
            # Swing trading suitability score
            swing_score = self._calculate_swing_volatility_score(
                avg_weekly_vol, recent_weekly_vol, weekly_return_vol, vol_std
            )
            
            return {
                'ticker': ticker,
                'avg_weekly_volatility': avg_weekly_vol,
                'max_weekly_volatility': max_weekly_vol,
                'min_weekly_volatility': min_weekly_vol,
                'weekly_return_volatility': weekly_return_vol,
                'volatility_consistency': vol_std,
                'recent_weekly_volatility': recent_weekly_vol,
                'price_volatility': price_volatility,
                'volatility_score': swing_score,
                'swing_trading_score': swing_score
            }
        except Exception as e:
            print(f"Error calculating weekly volatility for {ticker}: {e}")
            return None
    
    def _calculate_volatility_score(self, avg_vol: float, recent_vol: float, consistency: float) -> float:
        """
        Calculate volatility score (0-100) for intraday trading (legacy)
        Higher score = better for intraday trading
        
        Args:
            avg_vol: Average intraday volatility
            recent_vol: Recent volatility
            consistency: Volatility consistency (std)
            
        Returns:
            Score from 0-100
        """
        # Ideal intraday volatility: 2-8% (not too low, not too high)
        ideal_range = (2.0, 8.0)
        
        # Score based on average volatility
        if ideal_range[0] <= avg_vol <= ideal_range[1]:
            vol_score = 50.0
        elif avg_vol < ideal_range[0]:
            vol_score = 30.0  # Too low volatility
        elif avg_vol <= 12.0:
            vol_score = 40.0  # Acceptable
        else:
            vol_score = 20.0  # Too high, too risky
        
        # Bonus for recent volatility (momentum)
        if recent_vol > avg_vol * 1.2:
            vol_score += 20.0  # Increasing volatility
        elif recent_vol > avg_vol:
            vol_score += 10.0
        
        # Penalty for inconsistent volatility
        if consistency > avg_vol * 0.5:
            vol_score -= 10.0  # Too inconsistent
        
        return min(100.0, max(0.0, vol_score))
    
    def _calculate_swing_volatility_score(self, avg_vol: float, recent_vol: float, 
                                         return_vol: float, consistency: float) -> float:
        """
        Calculate volatility score for swing trading (weekly timeframe)
        Higher score = better for weekly swing trading
        
        Args:
            avg_vol: Average weekly volatility
            recent_vol: Recent weekly volatility
            return_vol: Weekly return volatility
            consistency: Volatility consistency (std)
            
        Returns:
            Score from 0-100
        """
        # Ideal weekly volatility: 5-15% (good range for 5-10% profit targets)
        ideal_range = (5.0, 15.0)
        
        # Score based on average weekly volatility
        if ideal_range[0] <= avg_vol <= ideal_range[1]:
            vol_score = 50.0
        elif avg_vol < ideal_range[0]:
            vol_score = 30.0  # Too low for weekly gains
        elif avg_vol <= 20.0:
            vol_score = 45.0  # Still acceptable
        else:
            vol_score = 25.0  # Too volatile, risky for swing trading
        
        # Bonus for optimal recent volatility
        if ideal_range[0] <= recent_vol <= ideal_range[1]:
            vol_score += 20.0  # Perfect for current swing trades
        elif recent_vol > avg_vol * 1.1:
            vol_score += 15.0  # Increasing volatility (good for entries)
        
        # Return volatility score (consistent weekly returns)
        if 3.0 <= return_vol <= 10.0:
            vol_score += 15.0  # Good weekly movement
        elif return_vol < 3.0:
            vol_score += 5.0  # Low movement
        
        # Consistency bonus (predictable patterns)
        if consistency < avg_vol * 0.6:
            vol_score += 15.0  # Consistent weekly patterns
        elif consistency > avg_vol * 0.8:
            vol_score -= 10.0  # Too unpredictable
        
        return min(100.0, max(0.0, vol_score))
    
    def calculate_volume_metrics(self, ticker: str, days: int = 30) -> Optional[Dict]:
        """
        Calculate volume metrics
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to analyze
            
        Returns:
            Dictionary with volume metrics
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f'{days}d', interval='1d')
            
            if hist.empty or len(hist) < 5:
                return None
            
            # Average volume
            avg_volume = hist['Volume'].mean()
            
            # Recent volume (last 5 days)
            recent_volume = hist['Volume'].tail(5).mean()
            
            # Volume ratio (recent vs average)
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
            
            # Volume trend
            volume_trend = 'increasing' if recent_volume > avg_volume else 'decreasing'
            
            # Dollar volume
            avg_dollar_volume = (hist['Close'] * hist['Volume']).mean()
            recent_dollar_volume = (hist['Close'].tail(5) * hist['Volume'].tail(5)).mean()
            
            # Volume score
            volume_score = self._calculate_volume_score(
                avg_dollar_volume, recent_dollar_volume, volume_ratio
            )
            
            return {
                'ticker': ticker,
                'avg_volume': avg_volume,
                'recent_volume': recent_volume,
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'avg_dollar_volume': avg_dollar_volume,
                'recent_dollar_volume': recent_dollar_volume,
                'volume_score': volume_score
            }
        except Exception as e:
            print(f"Error calculating volume for {ticker}: {e}")
            return None
    
    def _calculate_volume_score(self, avg_dollar_vol: float, recent_dollar_vol: float, volume_ratio: float) -> float:
        """
        Calculate volume score (0-100)
        Higher score = better liquidity
        
        Args:
            avg_dollar_vol: Average dollar volume
            recent_dollar_vol: Recent dollar volume
            volume_ratio: Recent volume / average volume
            
        Returns:
            Score from 0-100
        """
        score = 0.0
        
        # Minimum dollar volume requirement
        if avg_dollar_vol >= config.MIN_VOLUME:
            score += 30.0
        elif avg_dollar_vol >= config.MIN_VOLUME * 0.5:
            score += 15.0
        
        # Recent volume bonus
        if recent_dollar_vol >= config.MIN_VOLUME:
            score += 30.0
        elif recent_dollar_vol >= config.MIN_VOLUME * 0.5:
            score += 15.0
        
        # Volume momentum (increasing volume)
        if volume_ratio >= 1.5:
            score += 25.0  # Strong volume increase
        elif volume_ratio >= 1.2:
            score += 15.0  # Moderate increase
        elif volume_ratio >= 1.0:
            score += 10.0  # At least maintaining
        
        # High volume bonus
        if avg_dollar_vol >= config.MIN_VOLUME * 5:
            score += 15.0  # Very liquid
        
        return min(100.0, max(0.0, score))
    
    def rank_stocks_by_volatility(self, tickers: List[str], weekly: bool = True) -> List[Dict]:
        """
        Rank stocks by volatility and volume
        
        Args:
            tickers: List of ticker symbols
            weekly: Use weekly volatility (True) or intraday (False)
            
        Returns:
            List of ranked stocks with scores
        """
        ranked_stocks = []
        
        timeframe = "weekly" if weekly else "intraday"
        print(f"Analyzing {timeframe} volatility and volume for {len(tickers)} stocks...")
        
        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i}/{len(tickers)}] Analyzing {ticker}...", end='\r')
            
            if weekly:
                vol_data = self.calculate_weekly_volatility(ticker)
            else:
                vol_data = self.calculate_intraday_volatility(ticker)
            
            vol_metrics = self.calculate_volume_metrics(ticker)
            
            if vol_data and vol_metrics:
                # Combined score (60% volatility, 40% volume)
                combined_score = (
                    vol_data['volatility_score'] * 0.6 +
                    vol_metrics['volume_score'] * 0.4
                )
                
                stock_info = {
                    'ticker': ticker,
                    'volatility_score': vol_data['volatility_score'],
                    'volume_score': vol_metrics['volume_score'],
                    'combined_score': combined_score,
                    'avg_dollar_volume': vol_metrics['avg_dollar_volume'],
                    'volume_ratio': vol_metrics['volume_ratio'],
                    **vol_data,
                    **vol_metrics
                }
                
                # Add weekly-specific fields
                if weekly:
                    stock_info['avg_weekly_volatility'] = vol_data.get('avg_weekly_volatility', 0)
                    stock_info['recent_weekly_volatility'] = vol_data.get('recent_weekly_volatility', 0)
                    stock_info['swing_trading_score'] = vol_data.get('swing_trading_score', 0)
                else:
                    stock_info['avg_intraday_volatility'] = vol_data['avg_intraday_volatility']
                    stock_info['recent_volatility'] = vol_data['recent_volatility']
                
                ranked_stocks.append(stock_info)
            
            # Rate limiting
            time.sleep(0.1)
        
        print()  # New line after progress
        
        # Sort by combined score
        ranked_stocks.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return ranked_stocks

