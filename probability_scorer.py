"""
Probability and Conviction Level Scoring System
Calculates probability of profit and conviction level for each stock
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime

from data_analyzer import DataAnalyzer
from stock_predictor import StockPredictor
from ma_strategy import MAStrategy
from volatility_analyzer import VolatilityAnalyzer
import config

class ProbabilityScorer:
    """Calculates probability of profit and conviction level"""
    
    def __init__(self):
        self.data_analyzer = DataAnalyzer()
        self.predictor = StockPredictor()
        self.ma_strategy = MAStrategy()
        self.vol_analyzer = VolatilityAnalyzer()
    
    def calculate_probability(self, ticker: str) -> Optional[Dict]:
        """
        Calculate probability of profit for a stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with probability metrics
        """
        try:
            # Get historical data
            data = self.data_analyzer.get_historical_data(ticker, years=1)
            if data is None or data.empty:
                return None
            
            # Calculate technical indicators
            data = self.data_analyzer.calculate_technical_indicators(data)
            
            # ML prediction
            ml_prediction = self.predictor.predict_stock(ticker)
            ml_prob = ml_prediction.get('probability', 0) if ml_prediction else 0
            
            # MA strategy evaluation
            ma_eval = self.ma_strategy.evaluate_stock(data)
            ma_score = ma_eval.get('score', 0) if ma_eval.get('valid') else 0
            ma_prob = ma_score / 100.0
            
            # Technical indicator score
            tech_score = self._calculate_technical_score(data)
            tech_prob = tech_score / 100.0
            
            # Volatility score
            vol_data = self.vol_analyzer.calculate_intraday_volatility(ticker)
            vol_score = vol_data.get('volatility_score', 0) if vol_data else 0
            vol_prob = vol_score / 100.0
            
            # Weighted combination
            # ML: 40%, MA: 30%, Technical: 20%, Volatility: 10%
            combined_probability = (
                ml_prob * 0.4 +
                ma_prob * 0.3 +
                tech_prob * 0.2 +
                vol_prob * 0.1
            )
            
            return {
                'ticker': ticker,
                'ml_probability': ml_prob,
                'ma_probability': ma_prob,
                'technical_probability': tech_prob,
                'volatility_probability': vol_prob,
                'combined_probability': combined_probability,
                'ml_score': ml_prediction.get('confidence', 'LOW') if ml_prediction else 'LOW',
                'ma_signal': ma_eval.get('signal', 'NEUTRAL') if ma_eval else 'NEUTRAL',
            }
        except Exception as e:
            print(f"Error calculating probability for {ticker}: {e}")
            return None
    
    def _calculate_technical_score(self, data: pd.DataFrame) -> float:
        """
        Calculate technical indicator score
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            Score from 0-100
        """
        if data.empty:
            return 0.0
        
        latest = data.iloc[-1]
        score = 0.0
        
        # RSI score (oversold/overbought)
        rsi = latest.get('RSI', np.nan)
        if not pd.isna(rsi):
            if 30 <= rsi <= 40:  # Oversold but recovering
                score += 25.0
            elif 40 < rsi < 60:  # Neutral, good for entry
                score += 20.0
            elif 60 <= rsi <= 70:  # Strong but not overbought
                score += 15.0
            elif rsi < 30:  # Very oversold
                score += 10.0
            else:  # Overbought
                score += 5.0
        
        # MACD score
        macd = latest.get('MACD', np.nan)
        macd_signal = latest.get('MACD_Signal', np.nan)
        if not pd.isna(macd) and not pd.isna(macd_signal):
            if macd > macd_signal:  # Bullish
                score += 20.0
            else:  # Bearish
                score += 5.0
        
        # Bollinger Bands position
        bb_pos = latest.get('BB_Position', np.nan)
        if not pd.isna(bb_pos):
            if 0.2 <= bb_pos <= 0.8:  # Not at extremes
                score += 15.0
            elif bb_pos < 0.2:  # Near lower band (oversold)
                score += 10.0
            else:  # Near upper band (overbought)
                score += 5.0
        
        # Volume ratio
        vol_ratio = latest.get('Volume_Ratio', np.nan)
        if not pd.isna(vol_ratio):
            if vol_ratio >= 1.5:  # High volume
                score += 20.0
            elif vol_ratio >= 1.2:
                score += 15.0
            elif vol_ratio >= 1.0:
                score += 10.0
        
        # Price momentum
        momentum_5 = latest.get('Price_Momentum_5', np.nan)
        if not pd.isna(momentum_5):
            if 0 < momentum_5 < 5:  # Positive but not extreme
                score += 10.0
            elif momentum_5 >= 5:  # Strong momentum
                score += 5.0
        
        return min(100.0, score)
    
    def calculate_conviction_level(self, probability: float, ticker: str) -> str:
        """
        Calculate conviction level based on probability
        
        Args:
            probability: Combined probability (0-1)
            ticker: Stock ticker (for additional checks)
            
        Returns:
            Conviction level: VERY_HIGH, HIGH, MEDIUM, LOW, VERY_LOW
        """
        if probability >= 0.85:
            return 'VERY_HIGH'
        elif probability >= 0.75:
            return 'HIGH'
        elif probability >= 0.65:
            return 'MEDIUM'
        elif probability >= 0.50:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def get_comprehensive_score(self, ticker: str) -> Optional[Dict]:
        """
        Get comprehensive score with probability and conviction
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with all scoring metrics
        """
        prob_data = self.calculate_probability(ticker)
        if not prob_data:
            return None
        
        conviction = self.calculate_conviction_level(
            prob_data['combined_probability'],
            ticker
        )
        
        return {
            **prob_data,
            'conviction_level': conviction,
            'probability_percent': prob_data['combined_probability'] * 100,
        }

