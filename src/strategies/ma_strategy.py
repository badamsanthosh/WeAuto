"""
Moving Average Strategy Module
Implements 50-day and 250-day MA crossover strategy for optimal trade selection
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from core import config

class MAStrategy:
    """50/250 Moving Average Crossover Strategy"""
    
    def __init__(self):
        self.ma_fast = config.MA_FAST
        self.ma_slow = config.MA_SLOW
        self.require_golden_cross = config.REQUIRE_GOLDEN_CROSS
        self.require_price_above_ma50 = config.REQUIRE_PRICE_ABOVE_MA50
        self.require_price_above_ma250 = config.REQUIRE_PRICE_ABOVE_MA250
        self.min_ma_separation = config.MIN_MA_SEPARATION_PERCENT
    
    def evaluate_stock(self, data: pd.DataFrame) -> Dict:
        """
        Evaluate a stock based on 50/250 MA strategy
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            
        Returns:
            Dictionary with MA strategy evaluation
        """
        if data.empty or len(data) < 250:
            return {
                'valid': False,
                'reason': 'Insufficient data for MA calculation'
            }
        
        latest = data.iloc[-1]
        prev_day = data.iloc[-2] if len(data) > 1 else latest
        
        # Get MA values
        sma_50 = latest.get('SMA_50', np.nan)
        sma_250 = latest.get('SMA_250', np.nan)
        current_price = latest['Close']
        
        if pd.isna(sma_50) or pd.isna(sma_250):
            return {
                'valid': False,
                'reason': 'MA values not available'
            }
        
        # Calculate key metrics
        golden_cross = sma_50 > sma_250
        price_above_ma50 = current_price > sma_50
        price_above_ma250 = current_price > sma_250
        ma_separation = ((sma_50 - sma_250) / sma_250) * 100
        
        # Distance from MAs
        distance_from_ma50 = ((current_price - sma_50) / sma_50) * 100
        distance_from_ma250 = ((current_price - sma_250) / sma_250) * 100
        
        # MA slopes (momentum)
        ma50_slope = latest.get('MA50_Slope', 0)
        ma250_slope = latest.get('MA250_Slope', 0)
        
        # Recent crossover
        recent_crossover = latest.get('MA_Crossover_Recent', 0) == 1
        
        # Trend strength score
        trend_strength = latest.get('MA_Trend_Strength', 0)
        
        # Strategy validation
        valid = True
        reasons = []
        score = 0.0
        
        # Check golden cross requirement
        if self.require_golden_cross and not golden_cross:
            valid = False
            reasons.append('50MA not above 250MA (no golden cross)')
        elif golden_cross:
            score += 30.0  # Base score for golden cross
        
        # Check price above MA50
        if self.require_price_above_ma50 and not price_above_ma50:
            valid = False
            reasons.append('Price below 50MA')
        elif price_above_ma50:
            score += 25.0
        
        # Check price above MA250 (optional)
        if self.require_price_above_ma250 and not price_above_ma250:
            valid = False
            reasons.append('Price below 250MA')
        elif price_above_ma250:
            score += 20.0
        
        # MA separation (trend strength)
        if ma_separation >= self.min_ma_separation:
            score += 15.0
        elif ma_separation < 0:
            # Death cross - very bearish
            if self.require_golden_cross:
                valid = False
                reasons.append('Death cross detected (50MA < 250MA)')
        
        # Positive MA slopes (momentum)
        if ma50_slope > 0:
            score += 5.0
        if ma250_slope > 0:
            score += 5.0
        
        # Recent crossover bonus
        if recent_crossover:
            score += 10.0
        
        # Distance from MAs (not too extended)
        if 0 < distance_from_ma50 < 10:  # Between 0-10% above MA50
            score += 5.0
        elif distance_from_ma50 > 15:  # Too extended, might pull back
            score -= 5.0
        
        # Normalize score to 0-100
        score = min(100.0, max(0.0, score))
        
        return {
            'valid': valid,
            'score': score,
            'reasons': reasons,
            'golden_cross': golden_cross,
            'price_above_ma50': price_above_ma50,
            'price_above_ma250': price_above_ma250,
            'ma_separation': ma_separation,
            'distance_from_ma50': distance_from_ma50,
            'distance_from_ma250': distance_from_ma250,
            'ma50_slope': ma50_slope,
            'ma250_slope': ma250_slope,
            'recent_crossover': recent_crossover,
            'trend_strength': trend_strength,
            'sma_50': sma_50,
            'sma_250': sma_250,
            'current_price': current_price,
            'signal': 'STRONG_BUY' if score >= 80 and valid else 
                      'BUY' if score >= 60 and valid else 
                      'NEUTRAL' if valid else 'AVOID'
        }
    
    def filter_stocks_by_ma(self, predictions: list) -> list:
        """
        Filter and rank stocks based on MA strategy
        
        Args:
            predictions: List of stock prediction dictionaries
            
        Returns:
            Filtered and ranked list of stocks
        """
        if not config.MA_STRATEGY_ENABLED:
            return predictions
        
        filtered = []
        
        for pred in predictions:
            ticker = pred.get('ticker')
            if not ticker:
                continue
            
            # Get data for MA evaluation
            from data_analyzer import DataAnalyzer
            analyzer = DataAnalyzer()
            data = analyzer.get_historical_data(ticker, years=1)
            
            if data is None or data.empty:
                continue
            
            data = analyzer.calculate_technical_indicators(data)
            
            # Evaluate MA strategy
            ma_eval = self.evaluate_stock(data)
            
            if ma_eval['valid']:
                # Add MA strategy info to prediction
                pred['ma_score'] = ma_eval['score']
                pred['ma_signal'] = ma_eval['signal']
                pred['ma_golden_cross'] = ma_eval['golden_cross']
                pred['ma_separation'] = ma_eval['ma_separation']
                pred['ma_trend_strength'] = ma_eval['trend_strength']
                pred['distance_from_ma50'] = ma_eval['distance_from_ma50']
                pred['sma_50'] = ma_eval['sma_50']
                pred['sma_250'] = ma_eval['sma_250']
                
                # Combine ML probability with MA score
                ml_prob = pred.get('probability', 0)
                ma_score_norm = ma_eval['score'] / 100.0
                
                # Weighted combination: 60% ML, 40% MA strategy
                combined_score = (ml_prob * 0.6) + (ma_score_norm * 0.4)
                pred['combined_score'] = combined_score
                
                filtered.append(pred)
        
        # Sort by combined score (ML + MA strategy)
        filtered.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return filtered

