"""
Main Trading Bot
Orchestrates stock selection, prediction, and trade execution with approval
"""
import time
from datetime import datetime
from typing import List, Dict, Optional
import sys

from core.data_analyzer import DataAnalyzer
from ml.predictor import StockPredictor
from utils.moomoo_integration import MoomooIntegration
from core.risk_manager import RiskManager
import core.config as config

class TradingBot:
    """Main automated trading bot"""
    
    def __init__(self):
        self.data_analyzer = DataAnalyzer()
        self.predictor = StockPredictor(model_type=config.MODEL_TYPE)
        self.moomoo = MoomooIntegration()
        self.risk_manager = RiskManager()
        self.trained = False
        
    def initialize(self) -> bool:
        """
        Initialize the trading bot
        
        Returns:
            True if initialization successful
        """
        print("=" * 60)
        print("Initializing Automated Trading Bot")
        print("=" * 60)
        
        # Connect to Moomoo
        print("\n1. Connecting to Moomoo API...")
        if not self.moomoo.connect():
            print("Warning: Could not connect to Moomoo. Running in analysis mode only.")
        else:
            # Get account info
            account_info = self.moomoo.get_account_info()
            if account_info:
                print(f"Account connected: {account_info}")
        
        # Analyze market trends
        print("\n2. Analyzing market trends with 50/250 MA strategy...")
        market_trends = self.data_analyzer.analyze_market_trends(
            ['SPY', 'QQQ', 'DIA'], 
            config.HISTORICAL_YEARS
        )
        for ticker, trend in market_trends.items():
            golden_cross = "✅ Golden Cross" if trend.get('golden_cross') else "❌ No Golden Cross"
            ma_sep = f", MA Separation: {trend.get('ma_separation', 0):.2f}%" if trend.get('ma_separation') is not None else ""
            print(f"  {ticker}: {trend.get('trend', 'N/A')} market, {golden_cross}{ma_sep}, "
                  f"RSI: {trend.get('rsi', 'N/A'):.2f}" if trend.get('rsi') else f"RSI: N/A")
        
        # Train ML model
        print("\n3. Training ML model on historical data...")
        training_tickers = config.POPULAR_TICKERS[:10]  # Use subset for faster training
        if self.predictor.train_model(training_tickers):
            self.trained = True
            print("Model training completed!")
        else:
            print("Warning: Model training failed. Predictions may not be accurate.")
        
        print("\n" + "=" * 60)
        print("Initialization Complete!")
        print("=" * 60)
        
        return True
    
    def get_trade_recommendations(self) -> List[Dict]:
        """
        Get trade recommendations based on ML predictions
        
        Returns:
            List of trade recommendations
        """
        if not self.trained:
            print("Warning: Model not trained. Training now...")
            training_tickers = config.POPULAR_TICKERS[:10]
            self.predictor.train_model(training_tickers)
            self.trained = True
        
        print("\nAnalyzing stocks for intraday opportunities...")
        
        # Get top picks
        top_picks = self.predictor.get_top_picks(
            config.POPULAR_TICKERS,
            top_n=config.MAX_POSITIONS
        )
        
        recommendations = []
        
        for pick in top_picks:
            ticker = pick['ticker']
            current_price = pick['current_price']
            
            # Get real-time price from Moomoo if connected
            if self.moomoo.connected:
                realtime_price = self.moomoo.get_current_price(ticker)
                if realtime_price:
                    current_price = realtime_price
                    pick['current_price'] = realtime_price
                    pick['target_price'] = realtime_price * (1 + config.TARGET_GAIN_PERCENT / 100)
            
            # Calculate position size (if we have account info)
            account_info = self.moomoo.get_account_info() if self.moomoo.connected else None
            account_balance = 100000  # Default if not available
            if account_info:
                # Extract balance from account info (format may vary)
                try:
                    if isinstance(account_info, dict):
                        account_balance = float(account_info.get('total_assets', account_balance))
                except:
                    pass
            
            quantity = self.risk_manager.calculate_position_size(
                account_balance,
                current_price,
                pick['probability']
            )
            
            stop_loss = self.risk_manager.calculate_stop_loss_price(current_price)
            take_profit = self.risk_manager.calculate_take_profit_price(current_price)
            
            recommendation = {
                'ticker': ticker,
                'action': 'BUY',
                'current_price': current_price,
                'target_price': pick['target_price'],
                'stop_loss_price': stop_loss,
                'take_profit_price': take_profit,
                'quantity': quantity,
                'probability': pick['probability'],
                'confidence': pick['confidence'],
                'rsi': pick.get('rsi'),
                'volume_ratio': pick.get('volume_ratio'),
                'expected_gain_percent': config.TARGET_GAIN_PERCENT,
                'timestamp': datetime.now().isoformat(),
                # MA Strategy Information
                'ma_score': pick.get('ma_score'),
                'ma_signal': pick.get('ma_signal'),
                'ma_golden_cross': pick.get('ma_golden_cross'),
                'ma_separation': pick.get('ma_separation'),
                'sma_50': pick.get('sma_50'),
                'sma_250': pick.get('sma_250'),
                'distance_from_ma50': pick.get('distance_from_ma50'),
                'ma_trend_strength': pick.get('ma_trend_strength'),
                'combined_score': pick.get('combined_score', pick['probability']),
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def display_recommendations(self, recommendations: List[Dict]):
        """
        Display trade recommendations in a formatted way
        
        Args:
            recommendations: List of trade recommendations
        """
        if not recommendations:
            print("\nNo trade recommendations at this time.")
            return
        
        print("\n" + "=" * 80)
        print("TRADE RECOMMENDATIONS")
        print("=" * 80)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\nRecommendation #{i}:")
            print(f"  Ticker: {rec['ticker']}")
            print(f"  Action: {rec['action']}")
            print(f"  Current Price: ${rec['current_price']:.2f}")
            print(f"  Target Price: ${rec['target_price']:.2f}")
            print(f"  Stop Loss: ${rec['stop_loss_price']:.2f}")
            print(f"  Take Profit: ${rec['take_profit_price']:.2f}")
            print(f"  Quantity: {rec['quantity']} shares")
            print(f"  Position Value: ${rec['current_price'] * rec['quantity']:.2f}")
            print(f"  ML Probability: {rec['probability']:.2%}")
            print(f"  Confidence: {rec['confidence']}")
            
            # MA Strategy Information (PRIMARY STRATEGY)
            if rec.get('ma_score') is not None:
                print(f"\n  📊 MOVING AVERAGE STRATEGY (50/250 MA):")
                print(f"     MA Strategy Score: {rec['ma_score']:.1f}/100")
                print(f"     MA Signal: {rec.get('ma_signal', 'N/A')}")
                print(f"     Golden Cross (50MA > 250MA): {'✅ YES' if rec.get('ma_golden_cross') else '❌ NO'}")
                if rec.get('sma_50') and rec.get('sma_250'):
                    print(f"     50-Day MA: ${rec['sma_50']:.2f}")
                    print(f"     250-Day MA: ${rec['sma_250']:.2f}")
                    print(f"     MA Separation: {rec.get('ma_separation', 0):.2f}%")
                if rec.get('distance_from_ma50') is not None:
                    print(f"     Distance from 50MA: {rec['distance_from_ma50']:.2f}%")
                if rec.get('ma_trend_strength') is not None:
                    print(f"     Trend Strength: {rec['ma_trend_strength']:.2f}")
                if rec.get('combined_score'):
                    print(f"     Combined Score (ML + MA): {rec['combined_score']:.2%}")
            
            # Technical Indicators
            print(f"\n  📈 Technical Indicators:")
            if rec.get('rsi'):
                print(f"     RSI: {rec['rsi']:.2f}")
            if rec.get('volume_ratio'):
                print(f"     Volume Ratio: {rec['volume_ratio']:.2f}")
            print(f"     Expected Gain: {rec['expected_gain_percent']:.2f}%")
        
        print("\n" + "=" * 80)
    
    def request_approval(self, recommendation: Dict) -> bool:
        """
        Request user approval for a trade
        
        Args:
            recommendation: Trade recommendation dictionary
            
        Returns:
            True if approved, False otherwise
        """
        if not config.REQUIRE_APPROVAL:
            return True
        
        print(f"\n{'='*60}")
        print(f"APPROVAL REQUEST")
        print(f"{'='*60}")
        print(f"Ticker: {recommendation['ticker']}")
        print(f"Action: {recommendation['action']}")
        print(f"Price: ${recommendation['current_price']:.2f}")
        print(f"Quantity: {recommendation['quantity']} shares")
        print(f"Total Value: ${recommendation['current_price'] * recommendation['quantity']:.2f}")
        print(f"Target: ${recommendation['target_price']:.2f} ({recommendation['expected_gain_percent']:.2f}% gain)")
        print(f"Stop Loss: ${recommendation['stop_loss_price']:.2f}")
        print(f"Confidence: {recommendation['confidence']} ({recommendation['probability']:.2%})")
        print(f"{'='*60}")
        
        # Check risk validation
        account_info = self.moomoo.get_account_info() if self.moomoo.connected else None
        account_balance = 100000  # Default
        if account_info:
            try:
                if isinstance(account_info, dict):
                    account_balance = float(account_info.get('total_assets', account_balance))
            except:
                pass
        
        current_positions = len(self.moomoo.get_positions()) if self.moomoo.connected else 0
        
        validation = self.risk_manager.validate_trade(
            recommendation['ticker'],
            recommendation['current_price'],
            recommendation['quantity'],
            account_balance,
            current_positions
        )
        
        if not validation['valid']:
            print(f"\n⚠️  RISK VALIDATION FAILED: {validation['reason']}")
            return False
        
        # Request approval
        print(f"\nDo you approve this trade? (yes/no): ", end='')
        
        # For automated testing, you can set timeout
        try:
            response = input().strip().lower()
            return response in ['yes', 'y', 'approve']
        except (EOFError, KeyboardInterrupt):
            return False
    
    def execute_trade(self, recommendation: Dict) -> Dict:
        """
        Execute a trade through Moomoo
        
        Args:
            recommendation: Trade recommendation dictionary
            
        Returns:
            Dictionary with execution result
        """
        if not self.moomoo.connected:
            return {
                'success': False,
                'error': 'Not connected to Moomoo API'
            }
        
        ticker = recommendation['ticker']
        quantity = recommendation['quantity']
        price = recommendation['current_price']
        
        print(f"\nExecuting {recommendation['action']} order for {ticker}...")
        
        # Place buy order
        if recommendation['action'] == 'BUY':
            result = self.moomoo.place_buy_order(
                ticker=ticker,
                quantity=quantity,
                price=price,
                order_type='MARKET'  # Using market orders for speed
            )
        else:
            result = self.moomoo.place_sell_order(
                ticker=ticker,
                quantity=quantity,
                price=price,
                order_type='MARKET'
            )
        
        if result['success']:
            print(f"✅ Order executed successfully!")
            print(f"   Order details: {result.get('data', 'N/A')}")
        else:
            print(f"❌ Order failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    def monitor_positions(self):
        """
        Monitor open positions and check for exit signals
        """
        if not self.moomoo.connected:
            return
        
        positions = self.moomoo.get_positions()
        
        if not positions:
            return
        
        print("\n" + "=" * 60)
        print("MONITORING POSITIONS")
        print("=" * 60)
        
        for position in positions:
            ticker = position.get('code', '').replace('US.', '')
            entry_price = float(position.get('cost_price', 0))
            quantity = int(position.get('qty', 0))
            
            # Get current price
            current_price = self.moomoo.get_current_price(ticker)
            if not current_price:
                continue
            
            # Check exit conditions
            exit_signal = self.risk_manager.should_exit_position(entry_price, current_price)
            
            print(f"\nPosition: {ticker}")
            print(f"  Entry: ${entry_price:.2f}")
            print(f"  Current: ${current_price:.2f}")
            print(f"  Quantity: {quantity}")
            print(f"  P&L: ${(current_price - entry_price) * quantity:.2f} "
                  f"({((current_price - entry_price) / entry_price) * 100:.2f}%)")
            
            if exit_signal['should_exit']:
                print(f"  ⚠️  EXIT SIGNAL: {exit_signal['reason']}")
                
                if config.REQUIRE_APPROVAL:
                    print(f"  Requesting approval to close position...")
                    approval = self.request_approval({
                        'ticker': ticker,
                        'action': 'SELL',
                        'current_price': current_price,
                        'quantity': quantity,
                        'expected_gain_percent': exit_signal.get('gain_percent', 0)
                    })
                    
                    if approval:
                        self.execute_trade({
                            'ticker': ticker,
                            'action': 'SELL',
                            'current_price': current_price,
                            'quantity': quantity
                        })
    
    def run_daily_analysis(self):
        """
        Run daily analysis and generate recommendations
        """
        print(f"\n{'='*80}")
        print(f"DAILY TRADING ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Get recommendations
        recommendations = self.get_trade_recommendations()
        
        # Display recommendations
        self.display_recommendations(recommendations)
        
        # Process each recommendation
        for rec in recommendations:
            if self.request_approval(rec):
                self.execute_trade(rec)
            else:
                print(f"Trade for {rec['ticker']} was not approved.")
        
        # Monitor existing positions
        self.monitor_positions()
    
    def shutdown(self):
        """Shutdown the trading bot"""
        print("\nShutting down trading bot...")
        self.moomoo.disconnect()
        print("Trading bot shutdown complete.")


