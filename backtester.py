"""
Backtesting and Testing Framework
Supports multiple testing modes: backtesting, forward testing, stress testing, etc.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_analyzer import DataAnalyzer
from stock_predictor import StockPredictor
from ma_strategy import MAStrategy
from probability_scorer import ProbabilityScorer
import config

class Backtester:
    """Comprehensive testing framework for trading strategies"""
    
    def __init__(self):
        self.data_analyzer = DataAnalyzer()
        self.predictor = StockPredictor()
        self.ma_strategy = MAStrategy()
        self.probability_scorer = ProbabilityScorer()
        self.results = []
    
    def backtest_strategy(self, ticker: str, start_date: str, end_date: str,
                         initial_capital: float = 10000) -> Dict:
        """
        Backtest trading strategy on historical data
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Get historical data
            stock = self.data_analyzer.get_historical_data(ticker, years=1)
            if stock is None or stock.empty:
                return {'error': 'No data available'}
            
            # Filter by date range
            stock = stock[(stock.index >= start_date) & (stock.index <= end_date)]
            if stock.empty:
                return {'error': 'No data in date range'}
            
            # Calculate indicators
            stock = self.data_analyzer.calculate_technical_indicators(stock)
            
            # Simulate trades
            capital = initial_capital
            positions = []
            trades = []
            
            for i in range(1, len(stock)):
                current = stock.iloc[i]
                prev = stock.iloc[i-1]
                
                # Entry signal (simplified)
                entry_signal = self._get_entry_signal(current, prev)
                
                # Exit signal
                exit_signal = self._get_exit_signal(current, prev, positions)
                
                # Execute trades
                if entry_signal and capital > 0:
                    # Buy
                    shares = int(capital / current['Close'])
                    if shares > 0:
                        cost = shares * current['Close']
                        capital -= cost
                        positions.append({
                            'entry_date': current.name,
                            'entry_price': current['Close'],
                            'shares': shares,
                            'stop_loss': current['Close'] * (1 - config.STOP_LOSS_PERCENT / 100),
                            'take_profit': current['Close'] * (1 + config.TAKE_PROFIT_PERCENT / 100)
                        })
                
                # Check exits
                for pos in positions[:]:
                    if exit_signal or current['Close'] <= pos['stop_loss'] or current['Close'] >= pos['take_profit']:
                        # Sell
                        proceeds = pos['shares'] * current['Close']
                        capital += proceeds
                        profit = proceeds - (pos['shares'] * pos['entry_price'])
                        profit_pct = (profit / (pos['shares'] * pos['entry_price'])) * 100
                        
                        trades.append({
                            'ticker': ticker,
                            'entry_date': pos['entry_date'],
                            'exit_date': current.name,
                            'entry_price': pos['entry_price'],
                            'exit_price': current['Close'],
                            'shares': pos['shares'],
                            'profit': profit,
                            'profit_pct': profit_pct
                        })
                        positions.remove(pos)
            
            # Close remaining positions
            if positions and len(stock) > 0:
                final_price = stock.iloc[-1]['Close']
                for pos in positions:
                    proceeds = pos['shares'] * final_price
                    capital += proceeds
                    profit = proceeds - (pos['shares'] * pos['entry_price'])
                    profit_pct = (profit / (pos['shares'] * pos['entry_price'])) * 100
                    
                    trades.append({
                        'ticker': ticker,
                        'entry_date': pos['entry_date'],
                        'exit_date': stock.index[-1],
                        'entry_price': pos['entry_price'],
                        'exit_price': final_price,
                        'shares': pos['shares'],
                        'profit': profit,
                        'profit_pct': profit_pct
                    })
            
            # Calculate metrics
            total_return = capital - initial_capital
            total_return_pct = (total_return / initial_capital) * 100
            
            winning_trades = [t for t in trades if t['profit'] > 0]
            losing_trades = [t for t in trades if t['profit'] <= 0]
            
            win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
            avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0
            
            return {
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'final_capital': capital,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'num_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                'trades': trades
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_entry_signal(self, current: pd.Series, prev: pd.Series) -> bool:
        """Determine entry signal"""
        # Simple strategy: RSI oversold + price above MA50
        rsi_oversold = current.get('RSI', 50) <= config.RSI_OVERSOLD
        price_above_ma50 = current.get('Close', 0) > current.get('SMA_50', 0)
        golden_cross = current.get('MA_Golden_Cross', 0) == 1
        
        return (rsi_oversold or price_above_ma50) and golden_cross
    
    def _get_exit_signal(self, current: pd.Series, prev: pd.Series, positions: List) -> bool:
        """Determine exit signal"""
        if not positions:
            return False
        
        # Exit if RSI overbought
        rsi_overbought = current.get('RSI', 50) >= config.RSI_OVERBOUGHT
        
        return rsi_overbought
    
    def forward_test(self, ticker: str, days: int = 30) -> Dict:
        """
        Forward test on recent data (out-of-sample)
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to test
            
        Returns:
            Dictionary with forward test results
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        return self.backtest_strategy(ticker, start_date, end_date)
    
    def stress_test(self, ticker: str, scenarios: List[str] = None) -> Dict:
        """
        Stress test under different market conditions
        
        Args:
            ticker: Stock ticker symbol
            scenarios: List of scenarios to test
            
        Returns:
            Dictionary with stress test results
        """
        if scenarios is None:
            scenarios = ['bull_market', 'bear_market', 'high_volatility', 'low_volatility']
        
        results = {}
        
        for scenario in scenarios:
            # Define date ranges for each scenario
            scenario_dates = {
                'bull_market': ('2020-04-01', '2021-12-31'),  # Post-COVID recovery
                'bear_market': ('2022-01-01', '2022-12-31'),  # 2022 bear market
                'high_volatility': ('2020-03-01', '2020-04-30'),  # COVID crash
                'low_volatility': ('2017-01-01', '2017-12-31'),  # Calm year
            }
            
            if scenario in scenario_dates:
                start, end = scenario_dates[scenario]
                result = self.backtest_strategy(ticker, start, end)
                results[scenario] = result
        
        return results
    
    def run_comprehensive_test(self, tickers: List[str]) -> Dict:
        """
        Run comprehensive testing on multiple stocks
        
        Args:
            tickers: List of ticker symbols to test
            
        Returns:
            Dictionary with all test results
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TESTING")
        print("=" * 80)
        
        all_results = {
            'backtest': {},
            'forward_test': {},
            'stress_test': {},
            'summary': {}
        }
        
        for ticker in tickers:
            print(f"\nTesting {ticker}...")
            
            # Backtest (last 6 months)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            backtest_result = self.backtest_strategy(ticker, start_date, end_date)
            all_results['backtest'][ticker] = backtest_result
            
            # Forward test (last 30 days)
            forward_result = self.forward_test(ticker, days=30)
            all_results['forward_test'][ticker] = forward_result
            
            # Stress test
            stress_result = self.stress_test(ticker)
            all_results['stress_test'][ticker] = stress_result
        
        # Summary
        all_results['summary'] = self._calculate_summary(all_results)
        
        return all_results
    
    def _calculate_summary(self, results: Dict) -> Dict:
        """Calculate summary statistics"""
        backtests = results['backtest']
        forward_tests = results['forward_test']
        
        avg_return = np.mean([r.get('total_return_pct', 0) for r in backtests.values() if 'error' not in r])
        avg_win_rate = np.mean([r.get('win_rate', 0) for r in backtests.values() if 'error' not in r])
        
        return {
            'avg_return_pct': avg_return,
            'avg_win_rate': avg_win_rate,
            'num_stocks_tested': len(backtests)
        }

