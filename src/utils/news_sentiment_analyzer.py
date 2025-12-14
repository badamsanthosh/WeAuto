"""
News Sentiment Analysis Module
Analyzes news sentiment for stocks and ranks them by sentiment score (0-100)
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    try:
        from textblob import TextBlob
        TEXTBLOB_AVAILABLE = True
        VADER_AVAILABLE = False
    except ImportError:
        VADER_AVAILABLE = False
        TEXTBLOB_AVAILABLE = False

from core import config


class NewsSentimentAnalyzer:
    """Analyzes news sentiment for stocks"""
    
    def __init__(self):
        if VADER_AVAILABLE:
            self.analyzer = SentimentIntensityAnalyzer()
            self.use_vader = True
        elif TEXTBLOB_AVAILABLE:
            self.use_vader = False
        else:
            self.use_vader = None
            print("Warning: No sentiment analysis library available. Install vaderSentiment or textblob.")
    
    def _analyze_sentiment_text(self, text: str) -> float:
        """
        Analyze sentiment of a text string
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score from -1 (negative) to 1 (positive)
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        if self.use_vader is True:
            # VADER returns compound score from -1 to 1
            scores = self.analyzer.polarity_scores(text)
            return scores['compound']
        elif self.use_vader is False:
            # TextBlob returns polarity from -1 to 1
            blob = TextBlob(text)
            return blob.sentiment.polarity
        else:
            # Fallback: simple keyword-based approach
            return self._simple_sentiment(text)
    
    def _simple_sentiment(self, text: str) -> float:
        """
        Simple keyword-based sentiment analysis (fallback)
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score from -1 to 1
        """
        positive_words = ['bullish', 'surge', 'rally', 'gain', 'profit', 'growth', 'upgrade', 
                         'beat', 'exceed', 'strong', 'positive', 'buy', 'outperform', 'rise']
        negative_words = ['bearish', 'plunge', 'crash', 'loss', 'decline', 'downgrade', 
                         'miss', 'weak', 'negative', 'sell', 'underperform', 'fall', 'drop']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        # Normalize to -1 to 1
        score = (positive_count - negative_count) / (positive_count + negative_count + 1)
        return max(-1.0, min(1.0, score))
    
    def get_stock_news(self, ticker: str, days: int = 7) -> List[Dict]:
        """
        Get recent news for a stock
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            List of news articles with title and summary
        """
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return []
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_news = []
            
            for article in news:
                try:
                    # Convert timestamp to datetime
                    pub_date = datetime.fromtimestamp(article.get('providerPublishTime', 0))
                    
                    if pub_date >= cutoff_date:
                        filtered_news.append({
                            'title': article.get('title', ''),
                            'summary': article.get('summary', ''),
                            'publisher': article.get('publisher', ''),
                            'date': pub_date
                        })
                except:
                    # If date parsing fails, include anyway
                    filtered_news.append({
                        'title': article.get('title', ''),
                        'summary': article.get('summary', ''),
                        'publisher': article.get('publisher', ''),
                        'date': datetime.now()
                    })
            
            return filtered_news[:20]  # Limit to 20 most recent
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []
    
    def calculate_sentiment_score(self, ticker: str, days: int = 7) -> Optional[Dict]:
        """
        Calculate sentiment score for a stock based on recent news
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to analyze
            
        Returns:
            Dictionary with sentiment metrics including score (0-100)
        """
        try:
            news_articles = self.get_stock_news(ticker, days)
            
            if not news_articles:
                return {
                    'ticker': ticker,
                    'sentiment_score': 50.0,  # Neutral if no news
                    'sentiment_raw': 0.0,
                    'num_articles': 0,
                    'sentiment_label': 'NEUTRAL',
                    'confidence': 'LOW'
                }
            
            # Analyze sentiment for each article
            sentiment_scores = []
            for article in news_articles:
                title_score = self._analyze_sentiment_text(article['title'])
                summary_score = self._analyze_sentiment_text(article.get('summary', ''))
                
                # Weight title more heavily (60% title, 40% summary)
                combined_score = title_score * 0.6 + summary_score * 0.4
                sentiment_scores.append(combined_score)
            
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            
            # Convert from -1 to 1 scale to 0-100 scale
            # Formula: (sentiment + 1) / 2 * 100
            sentiment_score = ((avg_sentiment + 1) / 2) * 100
            
            # Determine label
            if sentiment_score >= 70:
                label = 'VERY_POSITIVE'
            elif sentiment_score >= 60:
                label = 'POSITIVE'
            elif sentiment_score >= 50:
                label = 'NEUTRAL'
            elif sentiment_score >= 40:
                label = 'NEGATIVE'
            else:
                label = 'VERY_NEGATIVE'
            
            # Confidence based on number of articles
            if len(news_articles) >= 10:
                confidence = 'HIGH'
            elif len(news_articles) >= 5:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            return {
                'ticker': ticker,
                'sentiment_score': round(sentiment_score, 2),
                'sentiment_raw': round(avg_sentiment, 4),
                'num_articles': len(news_articles),
                'sentiment_label': label,
                'confidence': confidence,
                'recent_news': news_articles[:5]  # Store top 5 for reference
            }
        except Exception as e:
            print(f"Error calculating sentiment for {ticker}: {e}")
            return None
    
    def rank_stocks_by_sentiment(self, tickers: List[str], days: int = 7) -> List[Dict]:
        """
        Rank stocks by sentiment score
        
        Args:
            tickers: List of ticker symbols
            days: Number of days to analyze
            
        Returns:
            List of stocks ranked by sentiment score (highest first)
        """
        ranked_stocks = []
        
        print(f"Analyzing news sentiment for {len(tickers)} stocks...")
        
        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i}/{len(tickers)}] Analyzing {ticker}...", end='\r')
            
            sentiment_data = self.calculate_sentiment_score(ticker, days)
            if sentiment_data:
                ranked_stocks.append(sentiment_data)
            
            # Rate limiting to avoid API throttling
            time.sleep(0.2)
        
        print()  # New line after progress
        
        # Sort by sentiment score (highest first)
        ranked_stocks.sort(key=lambda x: x['sentiment_score'], reverse=True)
        
        return ranked_stocks
    
    def get_sentiment_summary(self, ticker: str) -> str:
        """
        Get a human-readable sentiment summary for a stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Summary string
        """
        sentiment_data = self.calculate_sentiment_score(ticker)
        if not sentiment_data:
            return f"{ticker}: No sentiment data available"
        
        return (f"{ticker}: {sentiment_data['sentiment_label']} "
                f"(Score: {sentiment_data['sentiment_score']:.1f}/100, "
                f"Articles: {sentiment_data['num_articles']}, "
                f"Confidence: {sentiment_data['confidence']})")

