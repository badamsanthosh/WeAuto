"""
Advanced ML Predictor with State-of-the-Art Techniques
Target: >95% Win Rate for Weekly Swing Trading
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, StackingClassifier, AdaBoostClassifier)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from data_analyzer import DataAnalyzer
from ma_strategy import MAStrategy
import config

class AdvancedMLPredictor:
    """
    Advanced ML predictor with ensemble methods and feature engineering
    Targets >95% accuracy for weekly swing trading
    """
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = None
        self.data_analyzer = DataAnalyzer()
        self.ma_strategy = MAStrategy()
        self.feature_importance = None
        self.selected_features = None
        
    def prepare_advanced_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare advanced features with extensive engineering for >95% accuracy
        """
        # Base features from original system
        base_features = [
            # Moving Averages
            'SMA_50', 'SMA_250', 'SMA_20', 'SMA_200',
            'MA_Golden_Cross', 'Price_Above_MA50', 'Price_Above_MA250',
            'Distance_From_MA50', 'Distance_From_MA250',
            'MA_Separation', 'MA50_Slope', 'MA250_Slope',
            'MA_Trend_Strength',
            # Weekly indicators
            'Weekly_Gain', 'Weekly_Momentum', 'Weekly_RSI',
            'Weekly_Volatility', 'Weekly_Range', 'Weekly_Volume_Trend',
            'Swing_Strength',
            # Technical indicators
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Position', 'BB_Width',
            'Volume_Ratio', 'Volatility',
            'Daily_Return', 'Intraday_Range',
        ]
        
        # ADVANCED FEATURE ENGINEERING
        
        # 1. Momentum Features (multiple timeframes)
        data['Momentum_3d'] = data['Close'].pct_change(3) * 100
        data['Momentum_5d'] = data['Close'].pct_change(5) * 100
        data['Momentum_10d'] = data['Close'].pct_change(10) * 100
        data['Momentum_20d'] = data['Close'].pct_change(20) * 100
        
        # 2. Rate of Change indicators
        data['ROC_5'] = ((data['Close'] - data['Close'].shift(5)) / data['Close'].shift(5)) * 100
        data['ROC_10'] = ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)) * 100
        
        # 3. Volume indicators (advanced)
        data['Volume_MA_5'] = data['Volume'].rolling(window=5).mean()
        data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio_5_20'] = data['Volume_MA_5'] / data['Volume_MA_20']
        data['Volume_Trend_5d'] = (data['Volume'] - data['Volume'].shift(5)) / data['Volume'].shift(5)
        
        # 4. Volatility ratios
        data['Volatility_5d'] = data['Daily_Return'].rolling(window=5).std()
        data['Volatility_20d'] = data['Daily_Return'].rolling(window=20).std()
        data['Volatility_Ratio'] = data['Volatility_5d'] / data['Volatility_20d']
        
        # 5. Price patterns
        data['Higher_High'] = ((data['High'] > data['High'].shift(1)) & 
                               (data['High'].shift(1) > data['High'].shift(2))).astype(int)
        data['Higher_Low'] = ((data['Low'] > data['Low'].shift(1)) & 
                              (data['Low'].shift(1) > data['Low'].shift(2))).astype(int)
        data['Lower_Low'] = ((data['Low'] < data['Low'].shift(1)) & 
                             (data['Low'].shift(1) < data['Low'].shift(2))).astype(int)
        
        # 6. Candlestick patterns
        data['Body_Size'] = abs(data['Close'] - data['Open']) / data['Open']
        data['Upper_Shadow'] = (data['High'] - data[['Close', 'Open']].max(axis=1)) / data['Open']
        data['Lower_Shadow'] = (data[['Close', 'Open']].min(axis=1) - data['Low']) / data['Open']
        data['Bullish_Candle'] = (data['Close'] > data['Open']).astype(int)
        
        # 7. Support/Resistance (simple version using rolling max/min)
        data['Resistance_20'] = data['High'].rolling(window=20).max()
        data['Support_20'] = data['Low'].rolling(window=20).min()
        data['Distance_To_Resistance'] = (data['Resistance_20'] - data['Close']) / data['Close']
        data['Distance_To_Support'] = (data['Close'] - data['Support_20']) / data['Close']
        
        # 8. Trend strength indicators
        data['ADX_Period'] = 14  # Simplified ADX calculation
        data['Price_Range_14'] = data['High'].rolling(14).max() - data['Low'].rolling(14).min()
        data['Trend_Consistency'] = (data['Close'].rolling(14).apply(
            lambda x: 1 if (x.iloc[-1] > x.iloc[0] and all(x.diff().dropna() > -x.std())) else 0
        ))
        
        # 9. Multi-timeframe confirmation
        data['MA_Alignment'] = ((data['SMA_20'] > data['SMA_50']) & 
                               (data['SMA_50'] > data['SMA_250'])).astype(int)
        data['Price_Position'] = (data['Close'] - data['SMA_250']) / data['SMA_250'] * 100
        
        # 10. Volume-Price relationship
        data['Price_Volume_Trend'] = (data['Close'].pct_change() * data['Volume_Ratio']).rolling(5).mean()
        
        # 11. Consecutive days features
        data['Consecutive_Up_Days'] = (data['Close'] > data['Close'].shift(1)).astype(int).rolling(5).sum()
        data['Consecutive_Down_Days'] = (data['Close'] < data['Close'].shift(1)).astype(int).rolling(5).sum()
        
        # 12. Gap analysis
        data['Gap_Up'] = ((data['Open'] > data['Close'].shift(1)) & 
                         ((data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1) > 0.01)).astype(int)
        data['Gap_Down'] = ((data['Open'] < data['Close'].shift(1)) & 
                           ((data['Close'].shift(1) - data['Open']) / data['Close'].shift(1) > 0.01)).astype(int)
        
        # 13. Weekly pattern features
        data['Week_High_Low_Range'] = (data['Weekly_High'] - data['Weekly_Low']) / data['Close']
        data['Weekly_Close_Position'] = (data['Close'] - data['Weekly_Low']) / (data['Weekly_High'] - data['Weekly_Low'])
        
        # 14. Composite scoring features
        data['Momentum_Score'] = (
            (data['Momentum_5d'] > 0).astype(int) * 0.3 +
            (data['Weekly_Momentum'] > 0).astype(int) * 0.4 +
            (data['MA_Golden_Cross'] == 1).astype(int) * 0.3
        ) * 100
        
        data['Volatility_Score'] = np.where(
            (data['Weekly_Volatility'] >= 5) & (data['Weekly_Volatility'] <= 15),
            100,
            np.where(
                (data['Weekly_Volatility'] < 5) | (data['Weekly_Volatility'] > 15),
                50,
                0
            )
        )
        
        # Compile all features
        advanced_features = [
            'Momentum_3d', 'Momentum_5d', 'Momentum_10d', 'Momentum_20d',
            'ROC_5', 'ROC_10',
            'Volume_MA_5', 'Volume_MA_20', 'Volume_Ratio_5_20', 'Volume_Trend_5d',
            'Volatility_5d', 'Volatility_20d', 'Volatility_Ratio',
            'Higher_High', 'Higher_Low', 'Lower_Low',
            'Body_Size', 'Upper_Shadow', 'Lower_Shadow', 'Bullish_Candle',
            'Distance_To_Resistance', 'Distance_To_Support',
            'Price_Range_14', 'Trend_Consistency',
            'MA_Alignment', 'Price_Position',
            'Price_Volume_Trend',
            'Consecutive_Up_Days', 'Consecutive_Down_Days',
            'Gap_Up', 'Gap_Down',
            'Week_High_Low_Range', 'Weekly_Close_Position',
            'Momentum_Score', 'Volatility_Score'
        ]
        
        all_features = base_features + advanced_features
        
        # Create lagged features for all important indicators
        lag_features = []
        key_features = ['RSI', 'MACD', 'Weekly_RSI', 'Weekly_Momentum', 
                       'Momentum_5d', 'Volume_Ratio', 'MA_Separation']
        
        for col in key_features:
            if col in data.columns:
                for lag in [1, 2, 3, 5]:
                    lag_col = f'{col}_lag{lag}'
                    data[lag_col] = data[col].shift(lag)
                    lag_features.append(lag_col)
        
        all_features = all_features + lag_features
        
        # Filter available features
        available_features = [f for f in all_features if f in data.columns]
        
        # Clean data
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        
        if len(data) == 0:
            return pd.DataFrame(), pd.Series()
        
        X = data[available_features].copy()
        
        # Target: 5%+ weekly gain
        y = data['Target_Hit_Min'] if 'Target_Hit_Min' in data.columns else data['Target_Hit']
        
        # Remove rows with NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            return pd.DataFrame(), pd.Series()
        
        return X, y
    
    def train_ensemble_model(self, tickers: List[str]) -> bool:
        """
        Train advanced ensemble model for >95% accuracy
        """
        all_X = []
        all_y = []
        
        print(f"Training ADVANCED ENSEMBLE MODEL for >95% win rate...")
        print(f"Using {len(tickers)} tickers for training...")
        
        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i}/{len(tickers)}] Processing {ticker}...", end='\r')
            
            data = self.data_analyzer.get_historical_data(ticker, config.HISTORICAL_YEARS)
            if data is None or data.empty:
                continue
            
            data = self.data_analyzer.calculate_technical_indicators(data)
            X, y = self.prepare_advanced_features(data)
            
            if len(X) > 0 and len(y) > 0:
                all_X.append(X)
                all_y.append(y)
        
        print()  # New line after progress
        
        if len(all_X) == 0:
            print("❌ No data available for training")
            return False
        
        # Combine all data
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        
        print(f"\nTraining data shape: {X_combined.shape}")
        print(f"Positive samples: {y_combined.sum()} ({y_combined.mean()*100:.2f}%)")
        print(f"Negative samples: {len(y_combined) - y_combined.sum()} ({(1-y_combined.mean())*100:.2f}%)")
        
        # Feature selection to reduce overfitting
        print("\n🔍 Selecting best features...")
        selector = SelectKBest(f_classif, k=min(50, X_combined.shape[1]))
        X_selected = selector.fit_transform(X_combined, y_combined)
        self.feature_selector = selector
        self.selected_features = X_combined.columns[selector.get_support()].tolist()
        
        print(f"Selected {len(self.selected_features)} best features out of {X_combined.shape[1]}")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_combined, test_size=0.2, random_state=42, 
            stratify=y_combined
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build ensemble model
        print("\n🤖 Building ADVANCED ENSEMBLE MODEL...")
        
        # Individual models with optimized parameters
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        ada_model = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.5,
            random_state=42
        )
        
        # Voting ensemble (soft voting for probability)
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('gb', gb_model),
                ('ada', ada_model)
            ],
            voting='soft',
            weights=[0.35, 0.30, 0.25, 0.10]  # Weight by expected performance
        )
        
        print("Training ensemble model...")
        self.model = ensemble
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Cross-validation
        print("\n📊 Cross-validation (5-fold)...")
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        print(f"\n✅ MODEL TRAINING COMPLETE!")
        print(f"=" * 60)
        print(f"Train Accuracy: {train_score*100:.2f}%")
        print(f"Test Accuracy: {test_score*100:.2f}%")
        print(f"CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
        print(f"=" * 60)
        
        # Feature importance (from XGB model)
        if hasattr(self.model.estimators_[0], 'feature_importances_'):
            importances = self.model.estimators_[0].feature_importances_
            self.feature_importance = pd.DataFrame({
                'feature': [self.selected_features[i] for i in range(len(importances))],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            for idx, row in self.feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return True
    
    def predict_stock(self, ticker: str) -> Optional[Dict]:
        """
        Predict with advanced ensemble model
        """
        if self.model is None:
            return None
        
        # Get data
        data = self.data_analyzer.get_historical_data(ticker, years=2)
        if data is None or data.empty:
            return None
        
        data = self.data_analyzer.calculate_technical_indicators(data)
        X, _ = self.prepare_advanced_features(data)
        
        if len(X) == 0:
            return None
        
        # Use latest data
        X_latest = X.iloc[-1:].copy()
        
        # Apply feature selection
        if self.feature_selector is not None:
            X_latest_selected = self.feature_selector.transform(X_latest)
        else:
            X_latest_selected = X_latest[self.selected_features] if self.selected_features else X_latest
        
        X_latest_scaled = self.scaler.transform(X_latest_selected)
        
        # Predict
        probability = self.model.predict_proba(X_latest_scaled)[0][1]
        prediction = self.model.predict(X_latest_scaled)[0]
        
        current_price = data['Close'].iloc[-1]
        target_price_min = current_price * (1 + config.TARGET_GAIN_PERCENT_MIN / 100)
        target_price = current_price * (1 + config.TARGET_GAIN_PERCENT / 100)
        target_price_max = current_price * (1 + config.TARGET_GAIN_PERCENT_MAX / 100)
        
        # Enhanced confidence based on probability
        if probability >= 0.95:
            confidence = 'VERY_HIGH'
        elif probability >= 0.90:
            confidence = 'HIGH'
        elif probability >= 0.80:
            confidence = 'MEDIUM'
        elif probability >= 0.70:
            confidence = 'LOW'
        else:
            confidence = 'VERY_LOW'
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'target_price': target_price,
            'target_price_min': target_price_min,
            'target_price_max': target_price_max,
            'probability': probability,
            'prediction': bool(prediction),
            'confidence': confidence,
            'trading_style': 'WEEKLY_SWING',
            'model_type': 'ADVANCED_ENSEMBLE'
        }
