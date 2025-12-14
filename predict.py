#!/usr/bin/env python3
"""
Stock Prediction Runner
Quick wrapper to run predictions on specific stocks
"""
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import predictor
from ml.predictor import StockPredictor
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stock Price Predictor')
    parser.add_argument('--tickers', required=True, help='Comma-separated list of stock tickers')
    parser.add_argument('--days', type=int, default=5, help='Days to predict ahead')
    
    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(',')]
    
    print(f"\n{'='*80}")
    print(f"🤖 Stock Prediction for: {', '.join(tickers)}")
    print(f"{'='*80}\n")
    
    predictor = StockPredictor()
    
    # Check if model needs training
    if predictor.model is None:
        print("📚 Model not trained. Training on default stocks first...")
        print("   (This may take 1-2 minutes on first run)\n")
        
        # Use a mix of popular stocks for training
        training_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'JNJ', 'V']
        
        # Add user's tickers to training set if not already included
        for ticker in tickers:
            if ticker not in training_tickers:
                training_tickers.append(ticker)
        
        try:
            success = predictor.train_model(training_tickers[:10], weekly=True)
            if not success:
                print("  ⚠️  Training failed. Trying with fewer stocks...")
                success = predictor.train_model(['AAPL', 'MSFT', 'GOOGL'], weekly=True)
            
            if success:
                print("  ✅ Model trained successfully!\n")
            else:
                print("  ❌ Failed to train model. Cannot make predictions.")
                print("  💡 Try running: python3 run.py --mode backtest (trains model)")
                sys.exit(1)
        except Exception as e:
            print(f"  ❌ Training error: {e}")
            print("  💡 Try running: python3 run.py --mode backtest (trains model)")
            sys.exit(1)
    
    for ticker in tickers:
        print(f"\n📊 Analyzing {ticker}...")
        try:
            prediction = predictor.predict_stock(ticker)
            
            if prediction and isinstance(prediction, dict):
                print(f"  Current Price: ${prediction.get('current_price', 0):.2f}")
                print(f"  Probability: {prediction.get('probability', 0):.1%}")
                print(f"  Prediction: {'BUY' if prediction.get('prediction') else 'HOLD'}")
                print(f"  Confidence: {prediction.get('confidence', 'UNKNOWN')}")
                if 'target_price' in prediction:
                    target = prediction.get('target_price', 0)
                    current = prediction.get('current_price', 0)
                    gain_pct = ((target - current) / current * 100) if current > 0 else 0
                    print(f"  Target Price: ${target:.2f} ({gain_pct:+.1f}%)")
                if 'rsi' in prediction and prediction['rsi']:
                    print(f"  RSI: {prediction['rsi']:.1f}")
                if 'ma_signal' in prediction:
                    print(f"  MA Signal: {prediction.get('ma_signal', 'N/A')}")
            else:
                print(f"  ⚠️  Unable to generate prediction")
                print(f"     Possible reasons:")
                print(f"     • Insufficient historical data for {ticker}")
                print(f"     • Data fetch failed (check internet connection)")
                print(f"     • Ticker symbol may be invalid")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            print(f"  Details: {traceback.format_exc().split(chr(10))[-2]}")
    
    print(f"\n{'='*80}")
    print("✅ Prediction Complete!")
    print(f"{'='*80}\n")
