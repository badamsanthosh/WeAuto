"""
Volatility and Volume Analysis Module
Analyzes intraday volatility and volume for stock ranking
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import config

class VolatilityAnalyzer:
    """Analyzes volatility and volume for intraday trading"""
    
    def __init__(self):
        pass
    
    def calculate_intraday_volatility(self, ticker: str, days: int = 30) -> Optional[Dict]:
        """
        Calculate intraday volatility metrics
        
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
    
    def _calculate_volatility_score(self, avg_vol: float, recent_vol: float, consistency: float) -> float:
        """
        Calculate volatility score (0-100)
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
    
    def rank_stocks_by_volatility(self, tickers: List[str]) -> List[Dict]:
        """
        Rank stocks by volatility and volume
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            List of ranked stocks with scores
        """
        ranked_stocks = []
        
        print(f"Analyzing volatility and volume for {len(tickers)} stocks...")
        
        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i}/{len(tickers)}] Analyzing {ticker}...", end='\r')
            
            vol_data = self.calculate_intraday_volatility(ticker)
            vol_metrics = self.calculate_volume_metrics(ticker)
            
            if vol_data and vol_metrics:
                # Combined score (60% volatility, 40% volume)
                combined_score = (
                    vol_data['volatility_score'] * 0.6 +
                    vol_metrics['volume_score'] * 0.4
                )
                
                ranked_stocks.append({
                    'ticker': ticker,
                    'volatility_score': vol_data['volatility_score'],
                    'volume_score': vol_metrics['volume_score'],
                    'combined_score': combined_score,
                    'avg_intraday_volatility': vol_data['avg_intraday_volatility'],
                    'recent_volatility': vol_data['recent_volatility'],
                    'avg_dollar_volume': vol_metrics['avg_dollar_volume'],
                    'volume_ratio': vol_metrics['volume_ratio'],
                    **vol_data,
                    **vol_metrics
                })
            
            # Rate limiting
            time.sleep(0.1)
        
        print()  # New line after progress
        
        # Sort by combined score
        ranked_stocks.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return ranked_stocks

