"""
Stock Prediction Engine using Machine Learning
Predicts stocks with high probability of 5% intraday gains
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from data_analyzer import DataAnalyzer
from ma_strategy import MAStrategy
import config

class StockPredictor:
    """ML-based stock predictor for intraday gains"""
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.data_analyzer = DataAnalyzer()
        self.ma_strategy = MAStrategy()
        self.feature_importance = None
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for ML model
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            Tuple of (features, target)
        """
        # Select features - PRIORITIZE 50/250 MA STRATEGY FEATURES
        feature_cols = [
            # PRIMARY: Moving Average Strategy Features
            'SMA_50', 'SMA_250',  # Core MAs
            'MA_Golden_Cross',  # Golden cross signal
            'Price_Above_MA50', 'Price_Above_MA250',  # Price position
            'Distance_From_MA50', 'Distance_From_MA250',  # Distance metrics
            'MA_Separation',  # Trend strength
            'MA50_Slope', 'MA250_Slope',  # Momentum
            'MA_Crossover_Recent',  # Recent crossover
            'MA_Trend_Strength',  # Combined strength score
            # Secondary: Other Technical Indicators
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Position', 'BB_Width',
            'Volume_Ratio', 'Volatility',
            'SMA_20', 'SMA_200',  # Additional MAs
            'Intraday_Range',
            'Daily_Return',
        ]
        
        # Create lagged features (previous day's indicators)
        for col in feature_cols:
            if col in data.columns:
                data[f'{col}_lag1'] = data[col].shift(1)
                data[f'{col}_lag2'] = data[col].shift(2)
        
        # Add momentum features
        data['Price_Momentum_5'] = data['Close'].pct_change(5)
        data['Price_Momentum_10'] = data['Close'].pct_change(10)
        data['Volume_Momentum'] = data['Volume'].pct_change(5)
        
        # Add all lagged features to feature list
        all_features = feature_cols + [f'{col}_lag1' for col in feature_cols] + \
                      [f'{col}_lag2' for col in feature_cols] + \
                      ['Price_Momentum_5', 'Price_Momentum_10', 'Volume_Momentum']
        
        # Filter available features
        available_features = [f for f in all_features if f in data.columns]
        
        # Prepare data
        data = data.dropna()
        
        if len(data) == 0:
            return pd.DataFrame(), pd.Series()
        
        X = data[available_features]
        y = data['Target_Hit']  # Binary: 1 if 5%+ gain, 0 otherwise
        
        return X, y
    
    def train_model(self, tickers: List[str]) -> bool:
        """
        Train ML model on historical data from multiple tickers
        
        Args:
            tickers: List of tickers to train on
            
        Returns:
            True if training successful
        """
        all_X = []
        all_y = []
        
        print(f"Training model on {len(tickers)} tickers...")
        
        for ticker in tickers:
            data = self.data_analyzer.get_historical_data(ticker, config.HISTORICAL_YEARS)
            if data is None or data.empty:
                continue
                
            data = self.data_analyzer.calculate_technical_indicators(data)
            X, y = self.prepare_features(data)
            
            if len(X) > 0 and len(y) > 0:
                all_X.append(X)
                all_y.append(y)
        
        if len(all_X) == 0:
            print("No data available for training")
            return False
        
        # Combine all data
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        
        # Remove any remaining NaN values
        mask = ~(X_combined.isna().any(axis=1) | y_combined.isna())
        X_combined = X_combined[mask]
        y_combined = y_combined[mask]
        
        if len(X_combined) == 0:
            print("No valid data after cleaning")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Model trained successfully!")
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return True
    
    def predict_stock(self, ticker: str) -> Optional[Dict]:
        """
        Predict probability of 5% intraday gain for a stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return None
        
        # Get recent data
        data = self.data_analyzer.get_historical_data(ticker, years=1)
        if data is None or data.empty:
            return None
        
        data = self.data_analyzer.calculate_technical_indicators(data)
        X, _ = self.prepare_features(data)
        
        if len(X) == 0:
            return None
        
        # Use most recent data point
        X_latest = X.iloc[-1:].copy()
        X_latest_scaled = self.scaler.transform(X_latest)
        
        # Predict
        probability = self.model.predict_proba(X_latest_scaled)[0][1]
        prediction = self.model.predict(X_latest_scaled)[0]
        
        current_price = data['Close'].iloc[-1]
        target_price = current_price * (1 + config.TARGET_GAIN_PERCENT / 100)
        
        # Get MA strategy evaluation
        ma_eval = self.ma_strategy.evaluate_stock(data)
        
        result = {
            'ticker': ticker,
            'current_price': current_price,
            'target_price': target_price,
            'probability': probability,
            'prediction': bool(prediction),
            'confidence': 'HIGH' if probability >= 0.8 else 'MEDIUM' if probability >= 0.6 else 'LOW',
            'rsi': data['RSI'].iloc[-1] if not pd.isna(data['RSI'].iloc[-1]) else None,
            'volume_ratio': data['Volume_Ratio'].iloc[-1] if not pd.isna(data['Volume_Ratio'].iloc[-1]) else None,
        }
        
        # Add MA strategy information
        if ma_eval.get('valid'):
            result['ma_score'] = ma_eval['score']
            result['ma_signal'] = ma_eval['signal']
            result['ma_golden_cross'] = ma_eval['golden_cross']
            result['ma_separation'] = ma_eval['ma_separation']
            result['sma_50'] = ma_eval['sma_50']
            result['sma_250'] = ma_eval['sma_250']
            result['distance_from_ma50'] = ma_eval['distance_from_ma50']
            result['ma_trend_strength'] = ma_eval['trend_strength']
        
        return result
    
    def get_top_picks(self, tickers: List[str], top_n: int = 2) -> List[Dict]:
        """
        Get top N stock picks with highest probability of 5% intraday gain
        Uses 50/250 MA strategy as primary filter
        
        Args:
            tickers: List of tickers to evaluate
            top_n: Number of top picks to return
            
        Returns:
            List of top picks with predictions (filtered by MA strategy)
        """
        predictions = []
        
        for ticker in tickers:
            pred = self.predict_stock(ticker)
            if pred and pred['probability'] >= config.MIN_CONFIDENCE_SCORE:
                predictions.append(pred)
        
        # Apply MA strategy filter and ranking
        if config.MA_STRATEGY_ENABLED:
            predictions = self.ma_strategy.filter_stocks_by_ma(predictions)
        else:
            # Sort by probability if MA strategy disabled
            predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return predictions[:top_n]


