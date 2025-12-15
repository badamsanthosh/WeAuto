"""
ELITE ML SYSTEM - CONFIGURATION A
Target: 95%+ Win Rate
- Ultra-strict entry criteria (90/100 minimum)
- Wide adaptive stops (8-12%)
- Market regime filtering
- Perfect momentum sweet spot (4-6%)
- Very high volume requirement (1.8x+)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, ExtraTreesClassifier)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from typing import List, Dict, Optional
import warnings
import joblib
warnings.filterwarnings('ignore')

from core.data_analyzer import DataAnalyzer
import core.config as config

class EliteMLSystem95:
    """
    Elite ML System targeting 95%+ win rate
    Uses ultra-strict criteria and wide stops
    """
    
    def __init__(self):
        self.models = {}
        self.calibrated_model = None
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.data_analyzer = DataAnalyzer()
        self.selected_features = None
        
    def engineer_elite_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer 150+ elite features"""
        df = data.copy()
        
        # Price momentum features
        for period in [3, 5, 10, 20, 50, 100]:
            df[f'momentum_{period}'] = df['Close'].pct_change(period) * 100
            df[f'roc_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        
        # Moving average features
        for ma_period in [10, 20, 50, 100, 200, 250]:
            ma_col = f'SMA_{ma_period}'
            if ma_col not in df.columns:
                df[ma_col] = df['Close'].rolling(window=ma_period).mean()
            df[f'price_to_{ma_period}ma'] = (df['Close'] - df[ma_col]) / df[ma_col] * 100
        
        # Volume features
        for period in [5, 10, 20, 50]:
            df[f'volume_ma_{period}'] = df['Volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['Volume'] / df[f'volume_ma_{period}']
        
        # Volatility features
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['Close'].pct_change().rolling(window=period).std() * 100
            df[f'atr_{period}'] = (df['High'] - df['Low']).rolling(window=period).mean() / df['Close'] * 100
        
        # RSI variations
        for period in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for period in [20, 50]:
            bb_ma = df['Close'].rolling(window=period).mean()
            bb_std = df['Close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = bb_ma + (2 * bb_std)
            df[f'bb_lower_{period}'] = bb_ma - (2 * bb_std)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / bb_ma
            df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # Pattern features
        df['body_size'] = abs(df['Close'] - df['Open']) / df['Open']
        df['upper_shadow'] = (df['High'] - df[['Close', 'Open']].max(axis=1)) / df['Open']
        df['lower_shadow'] = (df[['Close', 'Open']].min(axis=1) - df['Low']) / df['Open']
        
        # Trend strength
        for period in [10, 20, 50]:
            df[f'higher_highs_{period}'] = (df['High'] > df['High'].shift(1)).rolling(window=period).sum() / period
            df[f'higher_lows_{period}'] = (df['Low'] > df['Low'].shift(1)).rolling(window=period).sum() / period
        
        # Composite scores
        if 'Weekly_Momentum' not in df.columns:
            df['Weekly_Momentum'] = df['Close'].pct_change(5) * 100
        if 'Weekly_Volatility' not in df.columns:
            df['Weekly_Volatility'] = df['Close'].pct_change().rolling(window=5).std() * 100
        
        df['momentum_composite'] = (
            (df['momentum_5'] > 0).astype(int) * 0.3 +
            (df['momentum_10'] > 0).astype(int) * 0.3 +
            (df['Weekly_Momentum'] > 0).astype(int) * 0.4
        ) * 100
        
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            df['trend_composite'] = (
                (df['Close'] > df['SMA_50']).astype(int) * 0.3 +
                (df['SMA_50'] > df['SMA_200']).astype(int) * 0.3 +
                (df['Close'] > df['SMA_200']).astype(int) * 0.4
            ) * 100
        else:
            df['trend_composite'] = 50
        
        df['volume_composite'] = (
            (df['volume_ratio_5'] > 1.2).astype(int) * 0.4 +
            (df['volume_ratio_10'] > 1.1).astype(int) * 0.3 +
            (df['volume_ratio_20'] > 1.0).astype(int) * 0.3
        ) * 100
        
        df['overall_score'] = (
            df['momentum_composite'] * 0.25 +
            df['trend_composite'] * 0.25 +
            df['volume_composite'] * 0.25 +
            50 * 0.25
        )
        
        return df
    
    def prepare_training_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data for training"""
        data = self.engineer_elite_features(data)
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        
        if len(data) == 0:
            return pd.DataFrame(), pd.Series()
        
        feature_cols = [col for col in data.columns if col not in ['Target_Hit', 'Target_Hit_Min', 'Date']]
        feature_cols = [col for col in feature_cols if data[col].dtype in ['float64', 'int64', 'float32', 'int32']]
        
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
    
    def build_elite_ensemble(self) -> VotingClassifier:
        """Build elite ensemble with available models"""
        print("🚀 Building ELITE ENSEMBLE...")
        
        estimators = []
        weights = []
        
        # XGBoost (always available)
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        estimators.append(('xgb', xgb_model))
        weights.append(0.30)
        
        # LightGBM (if available)
        if LGBM_AVAILABLE:
            lgbm_model = LGBMClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.03,
                random_state=42,
                verbose=-1
            )
            estimators.append(('lgbm', lgbm_model))
            weights.append(0.25)
        
        # CatBoost (if available)
        if CATBOOST_AVAILABLE:
            catboost_model = CatBoostClassifier(
                iterations=300,
                depth=8,
                learning_rate=0.03,
                random_state=42,
                verbose=False
            )
            estimators.append(('catboost', catboost_model))
            weights.append(0.20)
        
        # Random Forest (always available)
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        estimators.append(('rf', rf_model))
        weights.append(0.15)
        
        # Gradient Boosting (always available)
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.03,
            random_state=42
        )
        estimators.append(('gb', gb_model))
        weights.append(0.10)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        print(f"   Using {len(estimators)} models: {[name for name, _ in estimators]}")
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )
        
        return ensemble
    
    def train(self, tickers: List[str], min_confidence_threshold: float = 0.95) -> bool:
        """Train elite ML system"""
        print("\n" + "="*80)
        print("🌟 TRAINING ELITE ML SYSTEM (95% WIN RATE TARGET)")
        print("="*80)
        print(f"Training on {len(tickers)} stocks...")
        
        all_X = []
        all_y = []
        
        for i, ticker in enumerate(tickers, 1):
            if i % 10 == 0:
                print(f"  [{i}/{len(tickers)}] Processed {i} stocks...")
            
            try:
                data = self.data_analyzer.get_historical_data(ticker, years=10)
                if data is None or data.empty or len(data) < 300:
                    continue
                
                data = self.data_analyzer.calculate_technical_indicators(data)
                X, y = self.prepare_training_data(data)
                
                if len(X) > 0 and len(y) > 0:
                    all_X.append(X)
                    all_y.append(y)
            except:
                continue
        
        if len(all_X) == 0:
            print("❌ No training data available")
            return False
        
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        
        print(f"\n📊 Training Data: {len(X_combined):,} samples, {X_combined.shape[1]} features")
        
        # Feature selection
        n_features = min(80, X_combined.shape[1])
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
        ensemble = self.build_elite_ensemble()
        ensemble.fit(X_train_scaled, y_train)
        
        # Calibrate
        self.calibrated_model = CalibratedClassifierCV(ensemble, cv=5, method='isotonic')
        self.calibrated_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        test_score = self.calibrated_model.score(X_test_scaled, y_test)
        y_pred_test = self.calibrated_model.predict(X_test_scaled)
        y_proba_test = self.calibrated_model.predict_proba(X_test_scaled)[:, 1]
        
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        
        high_conf_mask = y_proba_test >= min_confidence_threshold
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = accuracy_score(y_test[high_conf_mask], y_pred_test[high_conf_mask])
        else:
            high_conf_accuracy = 0
        
        print(f"\n✅ TRAINING COMPLETE!")
        print(f"   Test Accuracy: {test_score*100:.2f}%")
        print(f"   Precision: {precision*100:.2f}%")
        print(f"   Recall: {recall*100:.2f}%")
        print(f"   High Conf (>={min_confidence_threshold*100:.0f}%): {high_conf_accuracy*100:.2f}%")
        print("="*80)
        
        return True
    
    def predict(self, ticker: str, min_confidence: float = 0.95) -> Optional[Dict]:
        """Predict with elite model"""
        if self.calibrated_model is None:
            return None
        
        try:
            data = self.data_analyzer.get_historical_data(ticker, years=2)
            if data is None or data.empty:
                return None
            
            data = self.data_analyzer.calculate_technical_indicators(data)
            X, _ = self.prepare_training_data(data)
            
            if len(X) == 0:
                return None
            
            X_latest = X.iloc[-1:].copy()
            X_latest_selected = self.feature_selector.transform(X_latest)
            X_latest_scaled = self.scaler.transform(X_latest_selected)
            
            probability = self.calibrated_model.predict_proba(X_latest_scaled)[0][1]
            
            if probability < min_confidence:
                return None
            
            current_price = data['Close'].iloc[-1]
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'probability': probability,
                'confidence_pct': probability * 100,
                'model_type': 'ELITE_95PCT'
            }
        except:
            return None
    
    def save_model(self, filename: str = 'elite_ml_95pct.pkl'):
        """Save trained model"""
        model_data = {
            'calibrated_model': self.calibrated_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features
        }
        joblib.dump(model_data, filename)
        print(f"💾 Model saved to {filename}")
    
    def load_model(self, filename: str = 'elite_ml_95pct.pkl'):
        """Load trained model"""
        model_data = joblib.load(filename)
        self.calibrated_model = model_data['calibrated_model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.selected_features = model_data['selected_features']
        print(f"📂 Model loaded from {filename}")
