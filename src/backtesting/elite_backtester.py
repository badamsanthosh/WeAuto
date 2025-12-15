"""
ELITE BACKTESTER - CONFIGURATION A (95% WIN RATE TARGET)
Ultra-strict criteria with wide stops for maximum win rate
- Entry score: 90/100 minimum (vs 85/100)
- Stop loss: 8-12% (vs 3-5%)
- Volume: 1.8x minimum (vs 1.1x)
- Momentum: 4-6% sweet spot only
- RSI: 40-60 range only
- Market regime filter: SPY > 200MA, VIX < 20
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

from core.data_analyzer import DataAnalyzer
import core.config as config

class EliteBacktester95:
    """
    Elite backtester targeting 95%+ win rate
    Configuration A: Wide stops + ultra-strict criteria
    """
    
    def __init__(self, ml_system=None):
        self.data_analyzer = DataAnalyzer()
        self.ml_system = ml_system
        self.all_trades = []
        self.failed_trades = []
        self.successful_trades = []
        self.market_regime_cache = {}  # Cache market regime checks by date
        
    def check_market_regime(self, date: pd.Timestamp) -> bool:
        """
        Check if market regime is favorable for trading
        Only trade when SPY > 200MA and VIX < 20
        Cached by week to avoid excessive API calls
        """
        # Use week as cache key to reduce API calls
        week_key = date.strftime('%Y-W%U')
        
        if week_key in self.market_regime_cache:
            return self.market_regime_cache[week_key]
        
        try:
            # Get SPY data
            spy_end = date
            spy_start = spy_end - timedelta(days=365)
            spy = yf.download('SPY', start=spy_start, end=spy_end, progress=False, show_errors=False)
            
            if spy.empty or len(spy) < 200:
                self.market_regime_cache[week_key] = True
                return True  # If no data, don't filter
            
            # Check if SPY > 200MA
            spy_ma200 = spy['Close'].rolling(window=200).mean()
            spy_above_200ma = spy['Close'].iloc[-1] > spy_ma200.iloc[-1]
            
            # Get VIX data
            vix = yf.download('^VIX', start=spy_start, end=spy_end, progress=False, show_errors=False)
            if vix.empty:
                vix_low = True  # If no VIX data, don't filter on it
            else:
                vix_low = vix['Close'].iloc[-1] < 20
            
            result = spy_above_200ma and vix_low
            self.market_regime_cache[week_key] = result
            return result
        except:
            self.market_regime_cache[week_key] = True
            return True  # If error, don't filter
    
    def calculate_elite_entry_score(self, data: pd.DataFrame, idx: int, 
                                    check_regime: bool = True) -> Tuple[bool, float, Dict]:
        """
        ELITE entry criteria for 95% win rate
        Score threshold: 90/100 (only the absolute best setups)
        """
        if idx < 250:
            return False, 0.0, {}
        
        current = data.iloc[idx]
        score = 0.0
        details = {}
        
        # 1. MOVING AVERAGE FOUNDATION (25 points) - MANDATORY
        ma_score = 0
        if current.get('MA_Golden_Cross', 0) == 1:
            ma_score += 10
        if current.get('Price_Above_MA50', 0) == 1:
            ma_score += 8
        if current.get('Price_Above_MA250', 0) == 1:
            ma_score += 7
        
        score += ma_score
        details['ma_score'] = ma_score
        
        if ma_score < 20:  # Need 80% of MA points
            return False, score, details
        
        # 2. MOMENTUM SWEET SPOT (20 points) - CONFIGURATION A: 4-6% ONLY
        weekly_momentum = current.get('Weekly_Momentum', 0)
        momentum_score = 0
        
        if 4.0 <= weekly_momentum <= 6.0:
            momentum_score = 20  # PERFECT sweet spot
        elif 3.5 <= weekly_momentum < 4.0 or 6.0 < weekly_momentum <= 6.5:
            momentum_score = 10  # Borderline
        else:
            momentum_score = 0  # Reject
        
        score += momentum_score
        details['momentum_score'] = momentum_score
        
        if momentum_score < 15:  # Need strong momentum
            return False, score, details
        
        # 3. RSI OPTIMAL ZONE (15 points) - TIGHTENED: 40-60 only
        rsi = current.get('RSI', 50)
        weekly_rsi = current.get('Weekly_RSI', 50)
        rsi_score = 0
        
        if 45 <= rsi <= 58:
            rsi_score += 8  # Perfect
        elif 40 <= rsi < 45 or 58 < rsi <= 60:
            rsi_score += 5  # Acceptable
        
        if 45 <= weekly_rsi <= 58:
            rsi_score += 7  # Perfect
        elif 40 <= weekly_rsi < 45 or 58 < weekly_rsi <= 60:
            rsi_score += 4  # Acceptable
        
        score += rsi_score
        details['rsi_score'] = rsi_score
        
        # REJECT if RSI too extreme
        if rsi > 65 or weekly_rsi > 65 or rsi < 35 or weekly_rsi < 35:
            return False, score, details
        
        # 4. VOLUME CONFIRMATION (15 points) - CONFIGURATION A: 1.8x minimum
        vol_ratio = current.get('Volume_Ratio', 1.0)
        volume_score = 0
        
        if vol_ratio >= 2.2:
            volume_score = 15  # Exceptional
        elif vol_ratio >= 2.0:
            volume_score = 13
        elif vol_ratio >= 1.8:
            volume_score = 10
        else:
            volume_score = 0
        
        score += volume_score
        details['volume_score'] = volume_score
        
        # CRITICAL: Require 1.8x volume minimum
        if vol_ratio < 1.8:
            return False, score, details
        
        # 5. MACD CONFIRMATION (10 points)
        macd = current.get('MACD', 0)
        macd_signal = current.get('MACD_Signal', 0)
        macd_score = 0
        
        if macd > macd_signal and macd > 0:
            macd_score = 10
        elif macd > macd_signal:
            macd_score = 6
        
        score += macd_score
        details['macd_score'] = macd_score
        
        # 6. VOLATILITY ACCEPTANCE (10 points)
        weekly_vol = current.get('Weekly_Volatility', 0)
        volatility_score = 0
        
        if 5 <= weekly_vol <= 10:
            volatility_score = 10  # Ideal
        elif 4 <= weekly_vol < 5 or 10 < weekly_vol <= 12:
            volatility_score = 6
        
        score += volatility_score
        details['volatility_score'] = volatility_score
        
        if weekly_vol > 18 or weekly_vol < 3:
            return False, score, details
        
        # 7. PRICE POSITION (5 points)
        price = current.get('Close', 0)
        weekly_high = current.get('Weekly_High', price)
        weekly_low = current.get('Weekly_Low', price)
        
        price_position_score = 0
        if weekly_high > weekly_low:
            position_pct = (price - weekly_low) / (weekly_high - weekly_low)
            if 0.3 <= position_pct <= 0.6:
                price_position_score = 5
            elif 0.2 <= position_pct < 0.3 or 0.6 < position_pct <= 0.7:
                price_position_score = 3
        
        score += price_position_score
        details['price_position_score'] = price_position_score
        
        # 8. 52-WEEK HIGH PROXIMITY (5 points)
        try:
            high_52w = data['High'].rolling(252).max().iloc[idx]
            if pd.notna(high_52w) and high_52w > 0:
                if price / high_52w > 0.95:
                    score += 5
                    details['near_high_bonus'] = 5
                elif price / high_52w > 0.90:
                    score += 3
                    details['near_high_bonus'] = 3
        except:
            pass
        
        # 9. SWING STRENGTH (5 points)
        swing_strength = current.get('Swing_Strength', 0)
        swing_score = min(swing_strength * 5, 5)
        score += swing_score
        details['swing_score'] = swing_score
        
        # 10. MARKET REGIME FILTER (bonus points)
        if check_regime:
            regime_ok = self.check_market_regime(current.name)
            if regime_ok:
                score += 5  # Bonus for favorable market
                details['market_regime_bonus'] = 5
            else:
                # Don't trade in bad market regime
                return False, score, details
        
        details['total_score'] = score
        
        # CONFIGURATION A: 90/100 minimum required
        should_enter = score >= 90.0
        
        return should_enter, score, details
    
    def wide_adaptive_stop_loss(self, entry_price: float, weekly_volatility: float,
                                entry_score: float) -> float:
        """
        WIDE adaptive stop loss for 95% win rate
        CONFIGURATION A: 8-12% stops (vs 3-5%)
        """
        # Wide stops based on volatility
        if weekly_volatility <= 7:
            base_stop_pct = 8.0  # 8% for low vol
        elif weekly_volatility <= 10:
            base_stop_pct = 10.0  # 10% for medium vol
        elif weekly_volatility <= 15:
            base_stop_pct = 12.0  # 12% for medium-high vol
        else:
            base_stop_pct = 14.0  # 14% for high vol
        
        # Adjust based on entry quality
        if entry_score >= 95:
            stop_multiplier = 0.9  # Can use slightly tighter
        elif entry_score >= 92:
            stop_multiplier = 0.95
        else:
            stop_multiplier = 1.0
        
        final_stop_pct = base_stop_pct * stop_multiplier
        stop_price = entry_price * (1 - final_stop_pct / 100)
        
        return stop_price
    
    def intelligent_exit_logic(self, entry_price: float, current_price: float,
                               days_held: int, entry_score: float,
                               peak_price: float) -> Tuple[bool, str]:
        """Intelligent exit logic with higher targets"""
        profit_pct = ((current_price - entry_price) / entry_price) * 100
        
        # Higher targets for Configuration A
        # 1. Hit 15% - IMMEDIATE EXIT
        if profit_pct >= 15.0:
            return True, 'target_max_15pct'
        
        # 2. Hit 12% after 2+ days
        if profit_pct >= 12.0 and days_held >= 2:
            return True, 'target_mid_12pct'
        
        # 3. Hit 8% after 3+ days
        if profit_pct >= 8.0 and days_held >= 3:
            return True, 'target_min_8pct'
        
        # 4. TRAILING STOP after 5% profit
        if profit_pct >= 5.0 and peak_price > entry_price:
            drop_from_peak = ((peak_price - current_price) / peak_price) * 100
            if drop_from_peak >= 2.5:
                return True, 'trailing_stop_2.5pct'
        
        # 5. End of week exit if profitable
        if days_held >= 5 and profit_pct >= 2.0:
            return True, 'end_of_week_profitable'
        
        # 6. Mandatory exit after 7 days
        if days_held >= 7:
            if profit_pct > 0:
                return True, 'max_holding_profitable'
            else:
                return True, 'max_holding_forced'
        
        return False, ''
    
    def backtest_stock_40year(self, ticker: str, start_date: str, end_date: str,
                              initial_capital: float = 10000,
                              use_market_regime: bool = True) -> Dict:
        """Backtest single stock with elite criteria"""
        try:
            # Get 40 years of data
            data = self.data_analyzer.get_historical_data(ticker, years=40)
            if data is None or data.empty:
                return {'error': 'No data', 'ticker': ticker}
            
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            if data.empty or len(data) < 300:
                return {'error': 'Insufficient data', 'ticker': ticker}
            
            data = self.data_analyzer.calculate_technical_indicators(data)
            
            # Trading simulation
            capital = initial_capital
            positions = []
            trades = []
            
            i = 0
            while i < len(data):
                current = data.iloc[i]
                
                # Check for entry
                if len(positions) < config.MAX_POSITIONS and capital > 0:
                    should_enter, entry_score, details = self.calculate_elite_entry_score(
                        data, i, check_regime=use_market_regime
                    )
                    
                    if should_enter:
                        weekly_vol = current.get('Weekly_Volatility', 10)
                        
                        # Position size
                        position_size = min(capital * 0.5, config.MAX_POSITION_SIZE)
                        shares = int(position_size / current['Close'])
                        
                        if shares > 0:
                            cost = shares * current['Close']
                            capital -= cost
                            
                            # Wide adaptive stop loss
                            stop_loss = self.wide_adaptive_stop_loss(
                                current['Close'], weekly_vol, entry_score
                            )
                            
                            # Higher targets
                            target_min = current['Close'] * 1.08  # 8%
                            target_mid = current['Close'] * 1.12  # 12%
                            target_max = current['Close'] * 1.15  # 15%
                            
                            positions.append({
                                'entry_date': current.name,
                                'entry_price': current['Close'],
                                'shares': shares,
                                'stop_loss': stop_loss,
                                'target_min': target_min,
                                'target_mid': target_mid,
                                'target_max': target_max,
                                'entry_score': entry_score,
                                'days_held': 0,
                                'peak_price': current['Close'],
                                'entry_details': details
                            })
                            
                            i += 5  # Skip to next week
                            continue
                
                # Check exit conditions
                for pos in positions[:]:
                    pos['days_held'] += 1
                    pos['peak_price'] = max(pos['peak_price'], current['Close'])
                    
                    exit_reason = None
                    
                    # Stop loss check
                    if current['Close'] <= pos['stop_loss']:
                        exit_reason = 'stop_loss'
                    else:
                        should_exit, reason = self.intelligent_exit_logic(
                            pos['entry_price'], current['Close'], pos['days_held'],
                            pos['entry_score'], pos['peak_price']
                        )
                        if should_exit:
                            exit_reason = reason
                    
                    if exit_reason:
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
                            'entry_details': pos['entry_details'],
                            'peak_price': pos['peak_price'],
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
            
            # Close remaining positions
            if positions and len(data) > 0:
                final_price = data.iloc[-1]['Close']
                for pos in positions:
                    proceeds = pos['shares'] * final_price
                    capital += proceeds
                    profit = proceeds - (pos['shares'] * pos['entry_price'])
                    profit_pct = (profit / (pos['shares'] * pos['entry_price'])) * 100
                    
                    trade_record = {
                        'ticker': ticker,
                        'entry_date': pos['entry_date'],
                        'exit_date': data.index[-1],
                        'entry_price': pos['entry_price'],
                        'exit_price': final_price,
                        'shares': pos['shares'],
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'days_held': pos['days_held'],
                        'exit_reason': 'end_of_backtest',
                        'entry_score': pos['entry_score'],
                        'entry_details': pos['entry_details'],
                        'peak_price': pos.get('peak_price', final_price),
                        'success': profit > 0
                    }
                    
                    trades.append(trade_record)
                    self.all_trades.append(trade_record)
                    
                    if profit > 0:
                        self.successful_trades.append(trade_record)
                    else:
                        self.failed_trades.append(trade_record)
            
            return self._calculate_metrics(ticker, start_date, end_date,
                                          initial_capital, capital, trades)
        
        except Exception as e:
            return {'error': str(e), 'ticker': ticker}
    
    def _calculate_metrics(self, ticker: str, start_date: str, end_date: str,
                          initial_capital: float, final_capital: float,
                          trades: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        if not trades:
            return {
                'ticker': ticker,
                'error': 'No trades',
                'win_rate': 0,
                'num_trades': 0
            }
        
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] <= 0]
        
        win_rate = len(winning_trades) / len(trades) * 100
        
        avg_win_pct = np.mean([t['profit_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss_pct = np.mean([t['profit_pct'] for t in losing_trades]) if losing_trades else 0
        
        return {
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
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'avg_entry_score': np.mean([t['entry_score'] for t in trades]),
            'avg_holding_days': np.mean([t['days_held'] for t in trades]),
            'trades': trades
        }
    
    def run_elite_backtest(self, tickers: List[str], use_market_regime: bool = True,
                          parallel: bool = True) -> Dict:
        """Run elite backtest on all tickers"""
        print("\n" + "="*80)
        print("🚀 ELITE BACKTESTER - CONFIGURATION A (95% WIN RATE TARGET)")
        print("="*80)
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*40)).strftime('%Y-%m-%d')
        
        print(f"Period: {start_date} to {end_date} (40 years)")
        print(f"Stocks: {len(tickers)}")
        print(f"Entry Score: 90/100 minimum (ultra-elite)")
        print(f"Stop Loss: 8-12% (wide adaptive)")
        print(f"Volume: 1.8x minimum")
        print(f"Momentum: 4-6% sweet spot only")
        print(f"Market Regime Filter: {'ENABLED' if use_market_regime else 'DISABLED'}")
        print("="*80 + "\n")
        
        results = {}
        
        if parallel and len(tickers) > 5:
            print("🔄 Running parallel backtest...")
            with ThreadPoolExecutor(max_workers=min(10, len(tickers))) as executor:
                futures = {
                    executor.submit(self.backtest_stock_40year, ticker, start_date, 
                                  end_date, 10000, use_market_regime): ticker
                    for ticker in tickers
                }
                
                for i, future in enumerate(as_completed(futures), 1):
                    ticker = futures[future]
                    try:
                        result = future.result()
                        results[ticker] = result
                        print(f"  [{i}/{len(tickers)}] Completed {ticker}" + " "*20)
                    except Exception as e:
                        results[ticker] = {'error': str(e), 'ticker': ticker}
        else:
            for i, ticker in enumerate(tickers, 1):
                print(f"  [{i}/{len(tickers)}] Processing {ticker}...")
                result = self.backtest_stock_40year(ticker, start_date, end_date,
                                                   10000, use_market_regime)
                results[ticker] = result
        
        summary = self._calculate_summary(results)
        
        print("\n" + "="*80)
        print("📊 ELITE BACKTEST RESULTS")
        print("="*80)
        print(f"Stocks Tested: {summary['tickers_tested']}")
        print(f"Total Trades: {summary['total_trades']:,}")
        print(f"Win Rate: {summary['overall_win_rate']:.2f}%")
        print(f"Avg Entry Score: {summary['avg_entry_score']:.1f}/100")
        print(f"Profit Factor: {summary['profit_factor']:.2f}")
        print("="*80)
        
        if summary['overall_win_rate'] >= 95.0:
            print("\n🎉 ✅ TARGET ACHIEVED! Win Rate >= 95%!")
        elif summary['overall_win_rate'] >= 90.0:
            print(f"\n🎉 ✅ EXCELLENT! Win Rate >= 90%!")
        else:
            gap = 90.0 - summary['overall_win_rate']
            print(f"\n📈 Gap to 90%: {gap:.2f}%")
        
        return {
            'results': results,
            'summary': summary,
            'all_trades': self.all_trades,
            'failed_trades': self.failed_trades,
            'successful_trades': self.successful_trades
        }
    
    def _calculate_summary(self, results: Dict) -> Dict:
        """Calculate overall summary"""
        valid_results = [r for r in results.values() if 'error' not in r and r.get('num_trades', 0) > 0]
        
        if not valid_results:
            return {
                'tickers_tested': 0,
                'total_trades': 0,
                'overall_win_rate': 0,
                'avg_entry_score': 0,
                'profit_factor': 0
            }
        
        total_trades = sum([r['num_trades'] for r in valid_results])
        total_winning = sum([r['winning_trades'] for r in valid_results])
        
        overall_win_rate = (total_winning / total_trades * 100) if total_trades > 0 else 0
        avg_entry_score = np.mean([r['avg_entry_score'] for r in valid_results])
        
        total_profit = sum([r.get('total_return', 0) for r in valid_results if r.get('total_return', 0) > 0])
        total_loss = abs(sum([r.get('total_return', 0) for r in valid_results if r.get('total_return', 0) < 0]))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        return {
            'tickers_tested': len(valid_results),
            'total_trades': total_trades,
            'total_winning_trades': total_winning,
            'total_losing_trades': total_trades - total_winning,
            'overall_win_rate': overall_win_rate,
            'avg_entry_score': avg_entry_score,
            'profit_factor': profit_factor
        }
