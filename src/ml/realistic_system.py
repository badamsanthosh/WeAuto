"""
REALISTIC ELITE SYSTEM - Configuration B (70-75% Win Rate Target)
Based on actual 40-year backtest findings and analysis

KEY LEARNINGS:
1. 50% win rate with tight stops (3-5%) is normal for swing trading
2. 95% win rate requires either:
   - VERY wide stops (8-12%) + VERY few trades + accepting lower profit factor
   - OR Different trading style (position trading, not swing trading)
3. 70-75% win rate is EXCELLENT for swing trading and more achievable

CONFIGURATION B PARAMETERS:
- Entry Score: 85/100 (elite but not ultra-elite)
- Stop Loss: 5-7% (moderate adaptive)
- Volume: 1.5x minimum
- Momentum: 3.5-6.5% (wider acceptable range)
- RSI: 35-65  
- Targets: 5% / 7.5% / 10% (realistic for weekly swing trades)
- Holding: Up to 10 trading days (2 weeks)
"""

from backtesting.elite_backtester import EliteBacktester95
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import core.config as config
from core.data_analyzer import DataAnalyzer


class RealisticEliteBacktester(EliteBacktester95):
    """
    Realistic elite backtester targeting 70-75% win rate
    Configuration B: Balanced approach with better trade frequency
    """
    
    def calculate_realistic_entry_score(self, data: pd.DataFrame, idx: int) -> Tuple[bool, float, Dict]:
        """
        REALISTIC entry criteria for 70-75% win rate with decent trade frequency
        Score threshold: 85/100 (vs 90/100 ultra-strict)
        """
        if idx < 250:
            return False, 0.0, {}
        
        current = data.iloc[idx]
        score = 0.0
        details = {}
        
        # 1. MOVING AVERAGE FOUNDATION (25 points)
        ma_score = 0
        if current.get('MA_Golden_Cross', 0) == 1:
            ma_score += 10
        if current.get('Price_Above_MA50', 0) == 1:
            ma_score += 8
        if current.get('Price_Above_MA250', 0) == 1:
            ma_score += 7
        
        score += ma_score
        details['ma_score'] = ma_score
        
        # Require minimum 18/25 (vs 20/25)
        if ma_score < 18:
            return False, score, details
        
        # 2. MOMENTUM QUALITY (20 points) - WIDER RANGE for more opportunities
        weekly_momentum = current.get('Weekly_Momentum', 0)
        momentum_score = 0
        
        if 4.0 <= weekly_momentum <= 6.0:
            momentum_score = 20  # Perfect sweet spot
        elif 3.5 <= weekly_momentum < 4.0 or 6.0 < weekly_momentum <= 6.5:
            momentum_score = 15  # Good
        elif 3.0 <= weekly_momentum < 3.5 or 6.5 < weekly_momentum <= 7.0:
            momentum_score = 10  # Acceptable
        else:
            momentum_score = 0
        
        score += momentum_score
        details['momentum_score'] = momentum_score
        
        # Require minimum 10/20 (vs 15/20)
        if momentum_score < 10:
            return False, score, details
        
        # 3. RSI OPTIMAL ZONE (15 points) - WIDER RANGE
        rsi = current.get('RSI', 50)
        weekly_rsi = current.get('Weekly_RSI', 50)
        rsi_score = 0
        
        if 45 <= rsi <= 60:
            rsi_score += 8
        elif 40 <= rsi < 45 or 60 < rsi <= 65:
            rsi_score += 6  # Still acceptable
        elif 35 <= rsi < 40:
            rsi_score += 4
        
        if 45 <= weekly_rsi <= 60:
            rsi_score += 7
        elif 40 <= weekly_rsi < 45 or 60 < weekly_rsi <= 65:
            rsi_score += 5
        elif 35 <= weekly_rsi < 40:
            rsi_score += 3
        
        score += rsi_score
        details['rsi_score'] = rsi_score
        
        # More lenient: reject only if extremely overbought/oversold
        if rsi > 75 or weekly_rsi > 75 or rsi < 25 or weekly_rsi < 25:
            return False, score, details
        
        # 4. VOLUME CONFIRMATION (15 points) - LOWERED REQUIREMENT
        vol_ratio = current.get('Volume_Ratio', 1.0)
        volume_score = 0
        
        if vol_ratio >= 2.0:
            volume_score = 15
        elif vol_ratio >= 1.7:
            volume_score = 12
        elif vol_ratio >= 1.5:
            volume_score = 9
        elif vol_ratio >= 1.3:
            volume_score = 6
        
        score += volume_score
        details['volume_score'] = volume_score
        
        # Require 1.5x volume minimum (vs 1.8x)
        if vol_ratio < 1.5:
            return False, score, details
        
        # 5. MACD CONFIRMATION (10 points)
        macd = current.get('MACD', 0)
        macd_signal = current.get('MACD_Signal', 0)
        macd_score = 0
        
        if macd > macd_signal and macd > 0:
            macd_score = 10
        elif macd > macd_signal:
            macd_score = 7
        elif macd > 0:
            macd_score = 4
        
        score += macd_score
        details['macd_score'] = macd_score
        
        # 6. VOLATILITY ACCEPTANCE (10 points)
        weekly_vol = current.get('Weekly_Volatility', 0)
        volatility_score = 0
        
        if 5 <= weekly_vol <= 12:
            volatility_score = 10
        elif 4 <= weekly_vol < 5 or 12 < weekly_vol <= 15:
            volatility_score = 7
        elif 3 <= weekly_vol < 4 or 15 < weekly_vol <= 18:
            volatility_score = 4
        
        score += volatility_score
        details['volatility_score'] = volatility_score
        
        # More lenient volatility rejection
        if weekly_vol > 25 or weekly_vol < 2:
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
            elif 0.2 <= position_pct <= 0.7:
                price_position_score = 3
        
        score += price_position_score
        details['price_position_score'] = price_position_score
        
        # 8. 52-WEEK HIGH PROXIMITY (5 points)
        try:
            high_52w = data['High'].rolling(252).max().iloc[idx]
            if pd.notna(high_52w) and high_52w > 0:
                if price / high_52w > 0.95:
                    score += 5
                elif price / high_52w > 0.90:
                    score += 3
                elif price / high_52w > 0.80:
                    score += 1
        except:
            pass
        
        # 9. SWING STRENGTH (5 points)
        swing_strength = current.get('Swing_Strength', 0)
        swing_score = min(swing_strength * 5, 5)
        score += swing_score
        details['swing_score'] = swing_score
        
        details['total_score'] = score
        
        # Configuration B: 85/100 minimum (more opportunities)
        should_enter = score >= 85.0
        
        return should_enter, score, details
    
    def moderate_adaptive_stop_loss(self, entry_price: float, weekly_volatility: float,
                                    entry_score: float) -> float:
        """
        MODERATE adaptive stop loss for 70-75% win rate
        Configuration B: 5-7% stops (vs 8-12%)
        """
        # Moderate stops based on volatility
        if weekly_volatility <= 7:
            base_stop_pct = 5.0  # 5% for low vol
        elif weekly_volatility <= 12:
            base_stop_pct = 6.0  # 6% for medium vol
        else:
            base_stop_pct = 7.0  # 7% for high vol
        
        # Adjust based on entry quality
        if entry_score >= 90:
            stop_multiplier = 0.9
        elif entry_score >= 87:
            stop_multiplier = 0.95
        else:
            stop_multiplier = 1.0
        
        final_stop_pct = base_stop_pct * stop_multiplier
        stop_price = entry_price * (1 - final_stop_pct / 100)
        
        return stop_price
    
    def realistic_exit_logic(self, entry_price: float, current_price: float,
                            days_held: int, entry_score: float,
                            peak_price: float) -> Tuple[bool, str]:
        """
        REALISTIC exit logic with achievable targets
        Configuration B: 5% / 7.5% / 10% targets (vs 8% / 12% / 15%)
        """
        profit_pct = ((current_price - entry_price) / entry_price) * 100
        
        # 1. Hit maximum target (10%) - EXIT
        if profit_pct >= 10.0:
            return True, 'target_max_10pct'
        
        # 2. Hit target (7.5%) after 2+ days - EXIT
        if profit_pct >= 7.5 and days_held >= 2:
            return True, 'target_mid_7.5pct'
        
        # 3. Hit minimum (5%) after 3+ days - EXIT
        if profit_pct >= 5.0 and days_held >= 3:
            return True, 'target_min_5pct'
        
        # 4. TRAILING STOP after 3% profit (lower threshold)
        if profit_pct >= 3.0 and peak_price > entry_price:
            drop_from_peak = ((peak_price - current_price) / peak_price) * 100
            if drop_from_peak >= 2.0:
                return True, 'trailing_stop_2pct'
        
        # 5. End of week exit if profitable (5 days)
        if days_held >= 5 and profit_pct >= 1.5:
            return True, 'end_of_week_profitable'
        
        # 6. Extended holding - allow up to 10 days (2 weeks)
        if days_held >= 10:
            if profit_pct > 0:
                return True, 'max_holding_profitable'
            else:
                return True, 'max_holding_forced'
        
        return False, ''
    
    def backtest_stock_realistic(self, ticker: str, start_date: str, end_date: str,
                                initial_capital: float = 10000) -> Dict:
        """Backtest with realistic criteria"""
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
                    should_enter, entry_score, details = self.calculate_realistic_entry_score(data, i)
                    
                    if should_enter:
                        weekly_vol = current.get('Weekly_Volatility', 10)
                        
                        # Position size
                        position_size = min(capital * 0.5, config.MAX_POSITION_SIZE)
                        shares = int(position_size / current['Close'])
                        
                        if shares > 0:
                            cost = shares * current['Close']
                            capital -= cost
                            
                            # Moderate adaptive stop loss
                            stop_loss = self.moderate_adaptive_stop_loss(
                                current['Close'], weekly_vol, entry_score
                            )
                            
                            # Realistic targets
                            target_min = current['Close'] * 1.05  # 5%
                            target_mid = current['Close'] * 1.075  # 7.5%
                            target_max = current['Close'] * 1.10  # 10%
                            
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
                        should_exit, reason = self.realistic_exit_logic(
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
    
    def run_realistic_backtest(self, tickers: List[str], parallel: bool = True) -> Dict:
        """Run realistic backtest on all tickers"""
        print("\n" + "="*80)
        print("🚀 REALISTIC ELITE BACKTESTER - Configuration B (70-75% Win Rate Target)")
        print("="*80)
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*40)).strftime('%Y-%m-%d')
        
        print(f"Period: {start_date} to {end_date} (40 years)")
        print(f"Stocks: {len(tickers)}")
        print(f"Entry Score: 85/100 minimum (elite)")
        print(f"Stop Loss: 5-7% (moderate adaptive)")
        print(f"Volume: 1.5x minimum")
        print(f"Momentum: 3.5-6.5% (wider range)")
        print(f"Targets: 5% / 7.5% / 10% (realistic)")
        print(f"Max Holding: 10 days (2 weeks)")
        print("="*80 + "\n")
        
        results = {}
        
        if parallel and len(tickers) > 5:
            print("🔄 Running parallel backtest...")
            with ThreadPoolExecutor(max_workers=min(10, len(tickers))) as executor:
                futures = {
                    executor.submit(self.backtest_stock_realistic, ticker, start_date, end_date): ticker
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
                result = self.backtest_stock_realistic(ticker, start_date, end_date)
                results[ticker] = result
        
        summary = self._calculate_summary(results)
        
        print("\n" + "="*80)
        print("📊 REALISTIC BACKTEST RESULTS")
        print("="*80)
        print(f"Stocks Tested: {summary['tickers_tested']}")
        print(f"Total Trades: {summary['total_trades']:,}")
        print(f"Win Rate: {summary['overall_win_rate']:.2f}%")
        print(f"Avg Entry Score: {summary['avg_entry_score']:.1f}/100")
        print(f"Profit Factor: {summary['profit_factor']:.2f}")
        print("="*80)
        
        if summary['overall_win_rate'] >= 75.0:
            print("\n🎉 ✅ EXCEPTIONAL! Win Rate >= 75%!")
        elif summary['overall_win_rate'] >= 70.0:
            print(f"\n🎉 ✅ EXCELLENT! Win Rate >= 70%!")
        elif summary['overall_win_rate'] >= 65.0:
            print(f"\n✅ VERY GOOD! Win Rate >= 65%!")
        else:
            gap = 70.0 - summary['overall_win_rate']
            print(f"\n📈 Gap to 70%: {gap:.2f}%")
        
        return {
            'results': results,
            'summary': summary,
            'all_trades': self.all_trades,
            'failed_trades': self.failed_trades,
            'successful_trades': self.successful_trades
        }


if __name__ == '__main__':
    import sys
    from sp500_fetcher import SP500Fetcher
    
    # Parse arguments
    stock_count = 100
    
    if '--stocks' in sys.argv:
        idx = sys.argv.index('--stocks')
        stock_count = int(sys.argv[idx + 1])
    
    print("\n" + "="*100)
    print("🚀 REALISTIC ELITE SYSTEM - Configuration B")
    print("="*100)
    print(f"Target: 70-75% Win Rate with good trade frequency")
    print(f"Stocks: {stock_count}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100 + "\n")
    
    # Get stocks
    print("📊 Fetching stock universe...")
    fetcher = SP500Fetcher()
    all_tickers = fetcher.get_top_liquid_stocks(limit=500)
    test_tickers = all_tickers[:stock_count]
    
    print(f"✅ Testing {len(test_tickers)} stocks\n")
    
    # Run backtest
    backtester = RealisticEliteBacktester()
    results = backtester.run_realistic_backtest(
        tickers=test_tickers,
        parallel=True
    )
    
    summary = results['summary']
    
    print("\n" + "="*100)
    print("📊 FINAL RESULTS - Configuration B")
    print("="*100)
    print(f"Win Rate: {summary['overall_win_rate']:.2f}%")
    print(f"Total Trades: {summary['total_trades']:,}")
    print(f"Winning Trades: {summary['total_winning_trades']:,}")
    print(f"Losing Trades: {summary['total_losing_trades']:,}")
    print(f"Avg Entry Score: {summary['avg_entry_score']:.1f}/100")
    print(f"Profit Factor: {summary['profit_factor']:.2f}")
    
    if results['failed_trades']:
        import pandas as pd
        failed_df = pd.DataFrame(results['failed_trades'])
        print(f"\n📉 Failed Trade Analysis:")
        print(f"   Total Failures: {len(results['failed_trades'])}")
        print(f"   Avg Loss: {failed_df['profit_pct'].mean():.2f}%")
        print(f"   Max Loss: {failed_df['profit_pct'].min():.2f}%")
        
        print(f"\n🚪 Exit Reasons for Failures:")
        exit_reasons = failed_df['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            pct = (count / len(failed_df)) * 100
            print(f"   {reason}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*100)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100 + "\n")
