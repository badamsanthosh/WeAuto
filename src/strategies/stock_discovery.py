"""
Multi-Source Stock Discovery Module
Discovers trending stocks from multiple sources: Yahoo Finance, Moomoo, Twitter, Options, News
"""
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

import config

class StockDiscovery:
    """Discovers trending stocks from multiple sources"""
    
    def __init__(self):
        self.discovered_stocks = {}
        
    def get_yahoo_trending(self, limit: int = 10) -> List[str]:
        """
        Get trending stocks from Yahoo Finance
        
        Args:
            limit: Maximum number of stocks to return
            
        Returns:
            List of ticker symbols
        """
        try:
            # Yahoo Finance trending (most active)
            tickers = []
            
            # Get most active stocks
            try:
                # Use yfinance to get most active
                # Note: Yahoo Finance API may have limitations
                popular = yf.Tickers(' '.join(config.POPULAR_TICKERS[:20]))
                for ticker in config.POPULAR_TICKERS[:20]:
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        if info and 'volume' in info and info.get('volume', 0) > 1000000:
                            tickers.append(ticker)
                    except:
                        continue
            except:
                pass
            
            # Also try to get from market movers
            # Yahoo Finance doesn't have a direct API, so we use alternative methods
            return tickers[:limit]
        except Exception as e:
            print(f"Error getting Yahoo Finance trending: {e}")
            return []
    
    def get_moomoo_trending(self, limit: int = 10) -> List[str]:
        """
        Get trending stocks from Moomoo
        
        Args:
            limit: Maximum number of stocks to return
            
        Returns:
            List of ticker symbols
        """
        try:
            from moomoo_integration import MoomooIntegration
            moomoo = MoomooIntegration()
            
            if not moomoo.connect():
                return []
            
            # Get most active stocks from Moomoo
            # Note: This depends on Moomoo API capabilities
            # For now, return popular tickers if Moomoo is connected
            trending = config.POPULAR_TICKERS[:limit]
            
            moomoo.disconnect()
            return trending
        except Exception as e:
            print(f"Error getting Moomoo trending: {e}")
            return []
    
    def get_twitter_trending(self, limit: int = 10) -> List[str]:
        """
        Get trending stocks from Twitter/X
        
        Args:
            limit: Maximum number of stocks to return
            
        Returns:
            List of ticker symbols
        """
        try:
            # Note: Twitter API requires authentication
            # For now, we'll use a simple approach with common stock mentions
            # In production, you'd use Twitter API v2
            
            # Common trending patterns (this is a placeholder)
            # Real implementation would require Twitter API access
            trending = []
            
            # Placeholder: Return popular tech stocks that are often mentioned
            trending = ['AAPL', 'TSLA', 'NVDA', 'META', 'GOOGL', 'MSFT', 'AMZN', 'NFLX', 'AMD', 'INTC']
            
            return trending[:limit]
        except Exception as e:
            print(f"Error getting Twitter trending: {e}")
            return []
    
    def get_unusual_options(self, limit: int = 10) -> List[str]:
        """
        Get stocks with unusual options activity
        
        Args:
            limit: Maximum number of stocks to return
            
        Returns:
            List of ticker symbols
        """
        try:
            # Unusual options activity detection
            # This would typically require options data API
            # For now, we'll analyze volume spikes which often correlate with options activity
            
            unusual_stocks = []
            
            # Check for high volume stocks (often indicates options activity)
            for ticker in config.POPULAR_TICKERS:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period='5d')
                    if not hist.empty:
                        recent_volume = hist['Volume'].iloc[-1]
                        avg_volume = hist['Volume'].mean()
                        if recent_volume > avg_volume * 2:  # 2x average volume
                            unusual_stocks.append(ticker)
                except:
                    continue
                
                if len(unusual_stocks) >= limit:
                    break
            
            return unusual_stocks[:limit]
        except Exception as e:
            print(f"Error getting unusual options: {e}")
            return []
    
    def get_news_trending(self, limit: int = 10) -> List[str]:
        """
        Get stocks trending in financial news
        
        Args:
            limit: Maximum number of stocks to return
            
        Returns:
            List of ticker symbols
        """
        try:
            # Financial news analysis
            # This would typically use news APIs like NewsAPI, Alpha Vantage, etc.
            # For now, we'll use a simple approach
            
            news_stocks = []
            
            # Placeholder: In production, integrate with news API
            # Common stocks that appear in news frequently
            news_stocks = ['AAPL', 'TSLA', 'NVDA', 'META', 'GOOGL', 'MSFT', 'AMZN']
            
            return news_stocks[:limit]
        except Exception as e:
            print(f"Error getting news trending: {e}")
            return []
    
    def discover_all_sources(self, limit_per_source: int = 10) -> Dict[str, List[str]]:
        """
        Discover stocks from all sources
        
        Args:
            limit_per_source: Maximum stocks per source
            
        Returns:
            Dictionary with stocks from each source
        """
        print("Discovering stocks from multiple sources...")
        
        sources = {
            'yahoo_finance': self.get_yahoo_trending(limit_per_source),
            'moomoo': self.get_moomoo_trending(limit_per_source),
            'twitter': self.get_twitter_trending(limit_per_source),
            'unusual_options': self.get_unusual_options(limit_per_source),
            'news': self.get_news_trending(limit_per_source),
        }
        
        # Count occurrences
        ticker_counts = {}
        for source, tickers in sources.items():
            for ticker in tickers:
                if ticker not in ticker_counts:
                    ticker_counts[ticker] = {'count': 0, 'sources': []}
                ticker_counts[ticker]['count'] += 1
                ticker_counts[ticker]['sources'].append(source)
        
        self.discovered_stocks = ticker_counts
        
        return sources
    
    def get_combined_trending(self, min_sources: int = 2) -> List[str]:
        """
        Get stocks that appear in multiple sources
        
        Args:
            min_sources: Minimum number of sources a stock must appear in
            
        Returns:
            List of ticker symbols sorted by frequency
        """
        if not self.discovered_stocks:
            self.discover_all_sources()
        
        # Filter by minimum sources and sort by count
        filtered = {
            ticker: info 
            for ticker, info in self.discovered_stocks.items()
            if info['count'] >= min_sources
        }
        
        # Sort by count (descending)
        sorted_tickers = sorted(
            filtered.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        return [ticker for ticker, _ in sorted_tickers]

