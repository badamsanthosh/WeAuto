"""
Advanced Backtesting Framework with Failure Analysis
Comprehensive 10-year backtest with detailed trade analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')

from data_analyzer import DataAnalyzer
from stock_predictor import StockPredictor
from ma_strategy import MAStrategy
from probability_scorer import ProbabilityScorer
from volatility_analyzer import VolatilityAnalyzer
import config

class AdvancedBacktester:
    """Advanced backtesting with comprehensive failure analysis"""
    
    def __init__(self):
        self.data_analyzer = DataAnalyzer()
        self.predictor = StockPredictor()
        self.ma_strategy = MAStrategy()
        self.probability_scorer = ProbabilityScorer()
        self.vol_analyzer = VolatilityAnalyzer()
        self.all_trades = []
        self.failed_trades = []
        self.successful_trades = []
    
    def backtest_weekly_strategy(self, ticker: str, start_date: str, end_date: str,
                                 initial_capital: float = 10000) -> Dict:
        """
        Backtest WEEKLY trading strategy on historical data
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
            
        Returns:
            Dictionary with comprehensive backtest results
        """
        try:
            print(f"\n{'='*80}")
            print(f"Backtesting {ticker}: {start_date} to {end_date}")
            print(f"{'='*80}")
            
            # Get historical data
            stock = self.data_analyzer.get_historical_data(ticker, years=15)
            if stock is None or stock.empty:
                return {'error': 'No data available', 'ticker': ticker}
            
            # Filter by date range
            stock = stock[(stock.index >= start_date) & (stock.index <= end_date)]
            if stock.empty or len(stock) < 100:
                return {'error': 'Insufficient data in date range', 'ticker': ticker}
            
            # Calculate indicators (including weekly metrics)
            stock = self.data_analyzer.calculate_technical_indicators(stock)
            
            # Simulate weekly trades
            capital = initial_capital
            positions = []
            trades = []
            weeks_traded = 0
            max_trades_per_week = config.TRADES_PER_WEEK
            
            i = 0
            while i < len(stock):
                current = stock.iloc[i]
                week_start = i
                
                # Look for entry signals in the week
                trades_this_week = 0
                
                while i < len(stock) and trades_this_week < max_trades_per_week:
                    current = stock.iloc[i]
                    
                    # Check for entry signal (using advanced criteria)
                    if len(positions) < config.MAX_POSITIONS and capital > 0:
                        entry_signal, entry_score = self._get_advanced_entry_signal(
                            stock, i, ticker
                        )
                        
                        if entry_signal:
                            # Calculate position size
                            position_size = min(
                                capital * 0.5,  # Use 50% of capital per trade
                                config.MAX_POSITION_SIZE
                            )
                            shares = int(position_size / current['Close'])
                            
                            if shares > 0:
                                cost = shares * current['Close']
                                capital -= cost
                                
                                # Weekly targets
                                stop_loss = current['Close'] * (1 - config.STOP_LOSS_PERCENT / 100)
                                take_profit_min = current['Close'] * (1 + config.TAKE_PROFIT_MIN / 100)
                                take_profit_target = current['Close'] * (1 + config.TAKE_PROFIT_PERCENT / 100)
                                take_profit_max = current['Close'] * (1 + config.TAKE_PROFIT_MAX / 100)
                                
                                positions.append({
                                    'entry_date': current.name,
                                    'entry_price': current['Close'],
                                    'shares': shares,
                                    'stop_loss': stop_loss,
                                    'take_profit_min': take_profit_min,
                                    'take_profit_target': take_profit_target,
                                    'take_profit_max': take_profit_max,
                                    'entry_score': entry_score,
                                    'days_held': 0,
                                    'max_holding_days': config.HOLDING_PERIOD_DAYS,
                                    'entry_indicators': self._capture_indicators(current)
                                })
                                
                                trades_this_week += 1
                                weeks_traded += 1
                    
                    # Check exit conditions for open positions
                    for pos in positions[:]:
                        pos['days_held'] += 1
                        exit_reason = None
                        
                        # Stop loss
                        if current['Close'] <= pos['stop_loss']:
                            exit_reason = 'stop_loss'
                        # Take profit (5% minimum)
                        elif current['Close'] >= pos['take_profit_max']:
                            exit_reason = 'take_profit_max'
                        elif current['Close'] >= pos['take_profit_target'] and pos['days_held'] >= 3:
                            exit_reason = 'take_profit_target'
                        elif current['Close'] >= pos['take_profit_min'] and pos['days_held'] >= config.HOLDING_PERIOD_DAYS:
                            exit_reason = 'take_profit_min_eow'
                        # End of week exit
                        elif pos['days_held'] >= config.HOLDING_PERIOD_DAYS:
                            exit_reason = 'end_of_week'
                        
                        if exit_reason:
                            # Close position
                            proceeds = pos['shares'] * current['Close']
                            capital += proceeds
                            profit = proceeds - (pos['shares'] * pos['entry_price'])
                            profit_pct = (profit / (pos['shares'] * pos['entry_price'])) * 100
                            
                            trade_record = {
                                'ticker': ticker,
                                'entry_date': pos['entry_date'],
                                'exit_date': current.name,
                                'entry_price': pos['entry_price'],
                                'exit_price': current['Close'],
                                'shares': pos['shares'],
                                'profit': profit,
                                'profit_pct': profit_pct,
                                'days_held': pos['days_held'],
                                'exit_reason': exit_reason,
                                'entry_score': pos['entry_score'],
                                'entry_indicators': pos['entry_indicators'],
                                'exit_indicators': self._capture_indicators(current),
                                'success': profit > 0
                            }
                            
                            trades.append(trade_record)
                            self.all_trades.append(trade_record)
                            
                            if profit > 0:
                                self.successful_trades.append(trade_record)
                            else:
                                self.failed_trades.append(trade_record)
                            
                            positions.remove(pos)
                    
                    i += 1
                    
                    # Move to next week if we've traded or moved forward 5 days
                    if trades_this_week > 0 or i >= week_start + 5:
                        break
            
            # Close remaining positions at end
            if positions and len(stock) > 0:
                final_price = stock.iloc[-1]['Close']
                for pos in positions:
                    proceeds = pos['shares'] * final_price
                    capital += proceeds
                    profit = proceeds - (pos['shares'] * pos['entry_price'])
                    profit_pct = (profit / (pos['shares'] * pos['entry_price'])) * 100
                    
                    trade_record = {
                        'ticker': ticker,
                        'entry_date': pos['entry_date'],
                        'exit_date': stock.index[-1],
                        'entry_price': pos['entry_price'],
                        'exit_price': final_price,
                        'shares': pos['shares'],
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'days_held': pos['days_held'],
                        'exit_reason': 'end_of_backtest',
                        'entry_score': pos['entry_score'],
                        'entry_indicators': pos['entry_indicators'],
                        'exit_indicators': self._capture_indicators(stock.iloc[-1]),
                        'success': profit > 0
                    }
                    
                    trades.append(trade_record)
                    self.all_trades.append(trade_record)
                    
                    if profit > 0:
                        self.successful_trades.append(trade_record)
                    else:
                        self.failed_trades.append(trade_record)
            
            # Calculate comprehensive metrics
            return self._calculate_metrics(ticker, start_date, end_date, 
                                           initial_capital, capital, trades)
        
        except Exception as e:
            print(f"Error backtesting {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'ticker': ticker}
    
    def _get_advanced_entry_signal(self, stock: pd.DataFrame, idx: int, 
                                   ticker: str) -> Tuple[bool, float]:
        """
        Advanced entry signal with multi-factor scoring
        Returns (should_enter, confidence_score)
        """
        if idx < 250:  # Need enough data for indicators
            return False, 0.0
        
        current = stock.iloc[idx]
        score = 0.0
        max_score = 100.0
        
        # 1. Moving Average Strategy (30 points)
        if current.get('MA_Golden_Cross', 0) == 1:
            score += 10
        if current.get('Price_Above_MA50', 0) == 1:
            score += 10
        ma_separation = current.get('MA_Separation', 0)
        if ma_separation > config.MIN_MA_SEPARATION_PERCENT:
            score += 10
        
        # 2. Weekly Momentum (20 points)
        weekly_momentum = current.get('Weekly_Momentum', 0)
        if 2 < weekly_momentum < 8:
            score += 15
        elif 0 < weekly_momentum < 2:
            score += 8
        elif weekly_momentum >= 8:
            score += 5
        
        # 3. RSI Conditions (15 points)
        rsi = current.get('RSI', 50)
        weekly_rsi = current.get('Weekly_RSI', 50)
        if 35 <= rsi <= 65:
            score += 8
        if 35 <= weekly_rsi <= 65:
            score += 7
        
        # 4. Volume Confirmation (10 points)
        vol_ratio = current.get('Volume_Ratio', 1.0)
        if vol_ratio >= 1.2:
            score += 10
        elif vol_ratio >= 1.0:
            score += 5
        
        # 5. MACD (10 points)
        macd = current.get('MACD', 0)
        macd_signal = current.get('MACD_Signal', 0)
        if macd > macd_signal:
            score += 10
        
        # 6. Weekly Volatility (10 points)
        weekly_vol = current.get('Weekly_Volatility', 0)
        if 5 <= weekly_vol <= 15:
            score += 10
        elif 3 <= weekly_vol < 5 or 15 < weekly_vol <= 20:
            score += 5
        
        # 7. Swing Strength (5 points)
        swing_strength = current.get('Swing_Strength', 0)
        score += swing_strength * 5
        
        # Require minimum score of 60/100 for entry
        should_enter = score >= 60.0
        
        return should_enter, score
    
    def _capture_indicators(self, row: pd.Series) -> Dict:
        """Capture key indicators at trade time"""
        return {
            'rsi': row.get('RSI', None),
            'weekly_rsi': row.get('Weekly_RSI', None),
            'macd': row.get('MACD', None),
            'weekly_momentum': row.get('Weekly_Momentum', None),
            'ma_separation': row.get('MA_Separation', None),
            'volume_ratio': row.get('Volume_Ratio', None),
            'weekly_volatility': row.get('Weekly_Volatility', None),
            'swing_strength': row.get('Swing_Strength', None),
            'price': row.get('Close', None)
        }
    
    def _calculate_metrics(self, ticker: str, start_date: str, end_date: str,
                          initial_capital: float, final_capital: float, 
                          trades: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if not trades:
            return {
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'error': 'No trades executed',
                'win_rate': 0,
                'num_trades': 0
            }
        
        # Basic metrics
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] <= 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0
        
        avg_win_pct = np.mean([t['profit_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss_pct = np.mean([t['profit_pct'] for t in losing_trades]) if losing_trades else 0
        
        # Advanced metrics
        max_win = max([t['profit'] for t in trades]) if trades else 0
        max_loss = min([t['profit'] for t in trades]) if trades else 0
        
        max_win_pct = max([t['profit_pct'] for t in trades]) if trades else 0
        max_loss_pct = min([t['profit_pct'] for t in trades]) if trades else 0
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            reason = t.get('exit_reason', 'unknown')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        # Average holding period
        avg_holding_days = np.mean([t['days_held'] for t in trades]) if trades else 0
        
        profit_factor = abs(sum([t['profit'] for t in winning_trades]) / 
                           sum([t['profit'] for t in losing_trades])) if losing_trades and sum([t['profit'] for t in losing_trades]) != 0 else 0
        
        result = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'num_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'max_win': max_win,
            'max_loss': max_loss,
            'max_win_pct': max_win_pct,
            'max_loss_pct': max_loss_pct,
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding_days,
            'exit_reasons': exit_reasons,
            'trades': trades
        }
        
        # Print summary
        print(f"\n{ticker} Results:")
        print(f"  Total Return: ${total_return:.2f} ({total_return_pct:.2f}%)")
        print(f"  Trades: {len(trades)} (W:{len(winning_trades)}, L:{len(losing_trades)})")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Avg Win: {avg_win_pct:.2f}% | Avg Loss: {avg_loss_pct:.2f}%")
        print(f"  Profit Factor: {profit_factor:.2f}")
        
        return result
    
    def run_10_year_backtest(self, tickers: List[str]) -> Dict:
        """
        Run comprehensive 10-year backtest on all tickers
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE 10-YEAR BACKTEST - WEEKLY SWING TRADING")
        print("="*80)
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        
        results = {}
        
        for ticker in tickers:
            result = self.backtest_weekly_strategy(ticker, start_date, end_date)
            results[ticker] = result
        
        # Overall summary
        summary = self._calculate_overall_summary(results)
        
        print("\n" + "="*80)
        print("OVERALL 10-YEAR BACKTEST SUMMARY")
        print("="*80)
        print(f"Tickers Tested: {summary['tickers_tested']}")
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Overall Win Rate: {summary['overall_win_rate']:.2f}%")
        print(f"Average Return per Stock: {summary['avg_return_pct']:.2f}%")
        print(f"Total Winning Trades: {summary['total_winning_trades']}")
        print(f"Total Losing Trades: {summary['total_losing_trades']}")
        
        return {
            'results': results,
            'summary': summary,
            'all_trades': self.all_trades,
            'failed_trades': self.failed_trades,
            'successful_trades': self.successful_trades
        }
    
    def _calculate_overall_summary(self, results: Dict) -> Dict:
        """Calculate overall summary across all backtests"""
        valid_results = [r for r in results.values() if 'error' not in r]
        
        if not valid_results:
            return {
                'tickers_tested': 0,
                'total_trades': 0,
                'overall_win_rate': 0,
                'avg_return_pct': 0
            }
        
        total_trades = sum([r['num_trades'] for r in valid_results])
        total_winning = sum([r['winning_trades'] for r in valid_results])
        total_losing = sum([r['losing_trades'] for r in valid_results])
        
        overall_win_rate = (total_winning / total_trades * 100) if total_trades > 0 else 0
        avg_return_pct = np.mean([r['total_return_pct'] for r in valid_results])
        
        return {
            'tickers_tested': len(valid_results),
            'total_trades': total_trades,
            'total_winning_trades': total_winning,
            'total_losing_trades': total_losing,
            'overall_win_rate': overall_win_rate,
            'avg_return_pct': avg_return_pct
        }
    
    def analyze_failed_trades(self) -> Dict:
        """
        Comprehensive analysis of failed trades to identify patterns
        """
        print("\n" + "="*80)
        print("FAILED TRADES ANALYSIS")
        print("="*80)
        
        if not self.failed_trades:
            print("No failed trades to analyze!")
            return {}
        
        print(f"\nTotal Failed Trades: {len(self.failed_trades)}")
        print(f"Total Successful Trades: {len(self.successful_trades)}")
        
        failed_df = pd.DataFrame(self.failed_trades)
        
        # Analyze patterns
        analysis = {
            'total_failed': len(self.failed_trades),
            'avg_loss_pct': failed_df['profit_pct'].mean(),
            'max_loss_pct': failed_df['profit_pct'].min(),
            'exit_reasons': failed_df['exit_reason'].value_counts().to_dict(),
            'avg_holding_days': failed_df['days_held'].mean(),
            'avg_entry_score': failed_df['entry_score'].mean() if 'entry_score' in failed_df.columns else 0
        }
        
        print(f"\nFailed Trade Patterns:")
        print(f"  Average Loss: {analysis['avg_loss_pct']:.2f}%")
        print(f"  Max Loss: {analysis['max_loss_pct']:.2f}%")
        print(f"  Average Holding Days: {analysis['avg_holding_days']:.2f}")
        print(f"  Average Entry Score: {analysis['avg_entry_score']:.2f}/100")
        print(f"\nExit Reasons for Failed Trades:")
        for reason, count in analysis['exit_reasons'].items():
            pct = (count / len(self.failed_trades)) * 100
            print(f"    {reason}: {count} ({pct:.1f}%)")
        
        # Compare entry indicators between failed and successful trades
        if self.successful_trades:
            print(f"\n" + "="*80)
            print("COMPARISON: FAILED VS SUCCESSFUL TRADES")
            print("="*80)
            
            successful_df = pd.DataFrame(self.successful_trades)
            
            # Compare entry scores
            print(f"\nEntry Scores:")
            print(f"  Failed Trades Avg: {analysis['avg_entry_score']:.2f}")
            print(f"  Successful Trades Avg: {successful_df['entry_score'].mean():.2f}")
            print(f"  Difference: {successful_df['entry_score'].mean() - analysis['avg_entry_score']:.2f}")
        
        return analysis
    
    def save_backtest_results(self, results: Dict, filename: str = 'backtest_results.json'):
        """Save backtest results to file"""
        # Convert datetime objects to strings for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.strftime('%Y-%m-%d')
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nBacktest results saved to: {filename}")
