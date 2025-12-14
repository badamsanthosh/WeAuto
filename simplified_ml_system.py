"""
Simplified World-Class ML System (without CatBoost/LightGBM)
Uses XGBoost, Random Forest, Gradient Boosting, and other reliable models
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier, BaggingClassifier)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from typing import List, Dict, Tuple, Optional
import warnings
import joblib
warnings.filterwarnings('ignore')

from data_analyzer import DataAnalyzer
import config

class SimplifiedMLSystem:
    """
    Simplified ML system with reliable models only
    """
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.data_analyzer = DataAnalyzer()
        self.selected_features = None
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer essential features"""
        df = data.copy()
        
        # Multi-timeframe momentum
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['Close'].pct_change(period) * 100
        
        # Volume features
        for period in [5, 10, 20]:
            df[f'volume_ma_{period}'] = df['Volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['Volume'] / df[f'volume_ma_{period}']
        
        # Volatility
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['Close'].pct_change().rolling(window=period).std() * 100
        
        # RSI variations
        for period in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Price patterns
        df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['higher_low'] = (df['Low'] > df['Low'].shift(1)).astype(int)
        
        return df
    
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        data = self.engineer_features(data)
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        
        if len(data) == 0:
            return pd.DataFrame(), pd.Series()
        
        feature_cols = [col for col in data.columns if col not in ['Target_Hit', 'Target_Hit_Min', 'Date']]
        feature_cols = [col for col in feature_cols if data[col].dtype in ['float64', 'int64']]
        
        X = data[feature_cols].copy()
        
        if 'Target_Hit_Min' in data.columns:
            y = data['Target_Hit_Min']
        elif 'Target_Hit' in data.columns:
            y = data['Target_Hit']
        else:
            future_returns = data['Close'].pct_change(5).shift(-5) * 100
            y = (future_returns >= config.TARGET_GAIN_PERCENT_MIN).astype(int)
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def build_ensemble(self) -> VotingClassifier:
        """Build ensemble with 6 reliable models"""
        print("🚀 Building Ensemble with 6 models...")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        et_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        ada_model = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.5,
            random_state=42
        )
        
        bag_model = BaggingClassifier(
            n_estimators=100,
            max_samples=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('gb', gb_model),
                ('et', et_model),
                ('ada', ada_model),
                ('bag', bag_model)
            ],
            voting='soft',
            weights=[0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
        )
        
        return ensemble
    
    def train(self, tickers: List[str]) -> bool:
        """Train ML system"""
        print("\n" + "="*80)
        print("🌟 TRAINING ML SYSTEM")
        print("="*80)
        
        all_X = []
        all_y = []
        
        for i, ticker in enumerate(tickers, 1):
            if i % 10 == 0:
                print(f"  [{i}/{len(tickers)}] Processed...")
            
            try:
                data = self.data_analyzer.get_historical_data(ticker, years=10)
                if data is None or data.empty or len(data) < 300:
                    continue
                
                data = self.data_analyzer.calculate_technical_indicators(data)
                X, y = self.prepare_training_data(data)
                
                if len(X) > 0 and len(y) > 0:
                    all_X.append(X)
                    all_y.append(y)
            except Exception as e:
                continue
        
        if len(all_X) == 0:
            return False
        
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        
        print(f"\nTraining data: {len(X_combined):,} samples, {X_combined.shape[1]} features")
        
        # Feature selection
        n_features = min(50, X_combined.shape[1])
        selector = SelectKBest(f_classif, k=n_features)
        X_selected = selector.fit_transform(X_combined, y_combined)
        self.feature_selector = selector
        self.selected_features = X_combined.columns[selector.get_support()].tolist()
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_combined, test_size=0.2, random_state=42, stratify=y_combined
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble
        ensemble = self.build_ensemble()
        ensemble.fit(X_train_scaled, y_train)
        
        # Calibrate
        self.model = CalibratedClassifierCV(ensemble, cv=5, method='isotonic')
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"\n✅ Training Complete!")
        print(f"   Train Accuracy: {train_score*100:.2f}%")
        print(f"   Test Accuracy: {test_score*100:.2f}%")
        
        return True
    
    def save_model(self, filename: str = 'simplified_ml_model.pkl'):
        """Save model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features
        }, filename)
        print(f"💾 Model saved to {filename}")
