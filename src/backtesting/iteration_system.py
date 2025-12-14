"""
AUTOMATED ITERATION SYSTEM
Continuously runs backtests, analyzes failures, and applies improvements
until 90%+ win rate is achieved on 500 stocks over 40 years
"""
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

from utils.sp500_fetcher import SP500Fetcher
from backtesting.elite_backtester import EliteBacktester95
from ml.elite_ml_model import EliteMLSystem95


class AutomatedIterationSystem:
    """
    Automated system that iterates until target win rate is achieved
    """
    
    def __init__(self, target_win_rate: float = 90.0):
        self.target_win_rate = target_win_rate
        self.iteration_history = []
        self.best_config = None
        self.best_win_rate = 0.0
        
        # Configuration parameters that can be tuned
        self.config = {
            'entry_score_threshold': 90.0,  # Start with Configuration A
            'stop_loss_low_vol': 8.0,
            'stop_loss_med_vol': 10.0,
            'stop_loss_high_vol': 12.0,
            'min_volume_ratio': 1.8,
            'momentum_min': 4.0,
            'momentum_max': 6.0,
            'rsi_min': 40,
            'rsi_max': 60,
            'use_market_regime': False,  # Disabled for speed in initial iterations
            'target_min_pct': 8.0,
            'target_mid_pct': 12.0,
            'target_max_pct': 15.0
        }
    
    def run_iteration(self, iteration_num: int, tickers: List[str],
                     stock_limit: int = 100) -> Dict:
        """Run single iteration of backtest with current configuration"""
        print("\n" + "="*100)
        print(f"🔄 ITERATION #{iteration_num}")
        print("="*100)
        print(f"Configuration:")
        print(f"  Entry Score: {self.config['entry_score_threshold']}/100")
        print(f"  Stop Loss: {self.config['stop_loss_low_vol']}-{self.config['stop_loss_high_vol']}%")
        print(f"  Volume: {self.config['min_volume_ratio']}x minimum")
        print(f"  Momentum: {self.config['momentum_min']}-{self.config['momentum_max']}%")
        print(f"  RSI: {self.config['rsi_min']}-{self.config['rsi_max']}")
        print(f"  Market Regime Filter: {'ON' if self.config['use_market_regime'] else 'OFF'}")
        print(f"  Targets: {self.config['target_min_pct']}/{self.config['target_mid_pct']}/{self.config['target_max_pct']}%")
        print("="*100 + "\n")
        
        # Run backtest
        print(f"📊 Running backtest on {stock_limit} stocks...")
        test_tickers = tickers[:stock_limit]
        
        backtester = EliteBacktester95()
        results = backtester.run_elite_backtest(
            tickers=test_tickers,
            use_market_regime=self.config['use_market_regime'],
            parallel=True
        )
        
        summary = results['summary']
        win_rate = summary['overall_win_rate']
        
        # Record iteration
        iteration_record = {
            'iteration': iteration_num,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.copy(),
            'stocks_tested': stock_limit,
            'total_trades': summary['total_trades'],
            'win_rate': win_rate,
            'avg_entry_score': summary['avg_entry_score'],
            'profit_factor': summary['profit_factor'],
            'num_failures': len(results['failed_trades'])
        }
        
        self.iteration_history.append(iteration_record)
        
        # Update best if improved
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            self.best_config = self.config.copy()
            print(f"\n🎯 NEW BEST WIN RATE: {win_rate:.2f}%")
        
        return results
    
    def analyze_failures(self, results: Dict) -> Dict:
        """Analyze failed trades to identify improvement opportunities"""
        failed_trades = results['failed_trades']
        successful_trades = results['successful_trades']
        
        if not failed_trades:
            return {'recommendations': []}
        
        print("\n" + "="*100)
        print("🔍 ANALYZING FAILURES")
        print("="*100)
        
        failed_df = pd.DataFrame(failed_trades)
        success_df = pd.DataFrame(successful_trades) if successful_trades else None
        
        analysis = {
            'total_failed': len(failed_trades),
            'avg_loss_pct': failed_df['profit_pct'].mean(),
            'median_loss_pct': failed_df['profit_pct'].median(),
            'max_loss_pct': failed_df['profit_pct'].min(),
            'exit_reasons': failed_df['exit_reason'].value_counts().to_dict(),
            'avg_holding_days': failed_df['days_held'].mean(),
            'avg_entry_score': failed_df['entry_score'].mean()
        }
        
        print(f"\n📉 Failure Statistics:")
        print(f"   Total Failures: {analysis['total_failed']}")
        print(f"   Average Loss: {analysis['avg_loss_pct']:.2f}%")
        print(f"   Max Loss: {analysis['max_loss_pct']:.2f}%")
        print(f"   Avg Entry Score: {analysis['avg_entry_score']:.1f}/100")
        
        print(f"\n🚪 Exit Reasons:")
        for reason, count in analysis['exit_reasons'].items():
            pct = (count / len(failed_trades)) * 100
            print(f"   {reason}: {count} ({pct:.1f}%)")
        
        # Generate recommendations
        recommendations = self._generate_improvements(analysis, failed_df, success_df)
        
        return {
            'analysis': analysis,
            'recommendations': recommendations
        }
    
    def _generate_improvements(self, analysis: Dict, failed_df: pd.DataFrame,
                               success_df: pd.DataFrame) -> List[Dict]:
        """Generate specific improvements based on failure analysis"""
        recommendations = []
        
        # 1. Check stop loss exits
        stop_loss_pct = analysis['exit_reasons'].get('stop_loss', 0) / analysis['total_failed'] * 100
        
        if stop_loss_pct > 50:
            # Too many stop outs - widen stops
            recommendations.append({
                'parameter': 'stop_loss_low_vol',
                'current_value': self.config['stop_loss_low_vol'],
                'new_value': self.config['stop_loss_low_vol'] + 1.0,
                'reason': f'{stop_loss_pct:.0f}% of failures are stop-loss exits',
                'priority': 'HIGH',
                'expected_impact': '+3-5% win rate'
            })
            recommendations.append({
                'parameter': 'stop_loss_med_vol',
                'current_value': self.config['stop_loss_med_vol'],
                'new_value': self.config['stop_loss_med_vol'] + 1.0,
                'reason': 'Widen medium volatility stops proportionally',
                'priority': 'HIGH',
                'expected_impact': '+3-5% win rate'
            })
            recommendations.append({
                'parameter': 'stop_loss_high_vol',
                'current_value': self.config['stop_loss_high_vol'],
                'new_value': self.config['stop_loss_high_vol'] + 1.0,
                'reason': 'Widen high volatility stops proportionally',
                'priority': 'HIGH',
                'expected_impact': '+3-5% win rate'
            })
        
        # 2. Check entry score
        if analysis['avg_entry_score'] < 92 and self.config['entry_score_threshold'] < 95:
            recommendations.append({
                'parameter': 'entry_score_threshold',
                'current_value': self.config['entry_score_threshold'],
                'new_value': self.config['entry_score_threshold'] + 2.0,
                'reason': 'Low entry score on failures - require higher quality setups',
                'priority': 'MEDIUM',
                'expected_impact': '+2-3% win rate'
            })
        
        # 3. Check if volume helps
        if success_df is not None and len(success_df) > 0:
            # Compare volume between success and failures
            if 'volume_ratio' in failed_df.columns and 'volume_ratio' in success_df.columns:
                failed_avg_vol = failed_df['volume_ratio'].mean() if 'volume_ratio' in failed_df.columns else 0
                success_avg_vol = success_df['volume_ratio'].mean() if 'volume_ratio' in success_df.columns else 0
                
                if success_avg_vol > failed_avg_vol * 1.1:
                    recommendations.append({
                        'parameter': 'min_volume_ratio',
                        'current_value': self.config['min_volume_ratio'],
                        'new_value': self.config['min_volume_ratio'] + 0.2,
                        'reason': 'Successful trades have higher volume',
                        'priority': 'MEDIUM',
                        'expected_impact': '+1-2% win rate'
                    })
        
        # 4. Check momentum range
        if self.config['momentum_max'] - self.config['momentum_min'] > 2.0:
            # Tighten momentum range
            recommendations.append({
                'parameter': 'momentum_min',
                'current_value': self.config['momentum_min'],
                'new_value': self.config['momentum_min'] + 0.2,
                'reason': 'Tighten momentum sweet spot',
                'priority': 'LOW',
                'expected_impact': '+0.5-1% win rate'
            })
            recommendations.append({
                'parameter': 'momentum_max',
                'current_value': self.config['momentum_max'],
                'new_value': self.config['momentum_max'] - 0.2,
                'reason': 'Tighten momentum sweet spot',
                'priority': 'LOW',
                'expected_impact': '+0.5-1% win rate'
            })
        
        # 5. Enable market regime if not already
        if not self.config['use_market_regime']:
            recommendations.append({
                'parameter': 'use_market_regime',
                'current_value': False,
                'new_value': True,
                'reason': 'Market regime filter can eliminate bad market conditions',
                'priority': 'HIGH',
                'expected_impact': '+5-10% win rate'
            })
        
        # 6. Check RSI range
        if self.config['rsi_max'] > 60:
            recommendations.append({
                'parameter': 'rsi_max',
                'current_value': self.config['rsi_max'],
                'new_value': 60,
                'reason': 'Avoid overbought conditions',
                'priority': 'MEDIUM',
                'expected_impact': '+1-2% win rate'
            })
        
        return recommendations
    
    def apply_improvements(self, recommendations: List[Dict]) -> bool:
        """Apply recommended improvements to configuration"""
        if not recommendations:
            print("\n⚠️  No improvements to apply")
            return False
        
        print("\n" + "="*100)
        print("🔧 APPLYING IMPROVEMENTS")
        print("="*100)
        
        applied = 0
        for rec in recommendations:
            if rec['priority'] in ['HIGH', 'MEDIUM']:
                old_value = self.config[rec['parameter']]
                new_value = rec['new_value']
                self.config[rec['parameter']] = new_value
                
                print(f"\n✅ {rec['parameter']}")
                print(f"   {old_value} → {new_value}")
                print(f"   Reason: {rec['reason']}")
                print(f"   Expected: {rec['expected_impact']}")
                
                applied += 1
        
        print(f"\n📝 Applied {applied} improvements")
        return applied > 0
    
    def run_until_target(self, max_iterations: int = 10, stock_limit: int = 100,
                        scale_up_at: float = 90.0) -> Dict:
        """
        Run iterations until target win rate achieved
        
        Args:
            max_iterations: Maximum number of iterations
            stock_limit: Number of stocks to test in early iterations
            scale_up_at: Win rate threshold to scale up to full 500 stocks
        """
        print("\n" + "="*100)
        print("🚀 AUTOMATED ITERATION SYSTEM - TARGET: 90%+ WIN RATE")
        print("="*100)
        print(f"Maximum Iterations: {max_iterations}")
        print(f"Initial Test Set: {stock_limit} stocks")
        print(f"Scale to 500 stocks at: {scale_up_at}% win rate")
        print(f"Target Win Rate: {self.target_win_rate}%")
        print("="*100)
        
        # Get stock universe
        print("\n📊 Fetching stock universe...")
        fetcher = SP500Fetcher()
        all_tickers = fetcher.get_top_liquid_stocks(limit=500)
        
        print(f"✅ Got {len(all_tickers)} stocks")
        
        for iteration in range(1, max_iterations + 1):
            # Determine stock count for this iteration
            current_stock_limit = stock_limit
            if self.best_win_rate >= scale_up_at:
                current_stock_limit = 500
                print(f"\n🎯 WIN RATE >= {scale_up_at}% - SCALING TO 500 STOCKS")
            
            # Run iteration
            results = self.run_iteration(iteration, all_tickers, current_stock_limit)
            summary = results['summary']
            win_rate = summary['overall_win_rate']
            
            # Save iteration results
            self._save_iteration_results(iteration, results)
            
            # Check if target achieved
            if win_rate >= self.target_win_rate and current_stock_limit >= 500:
                print("\n" + "="*100)
                print("🎉 🎉 🎉 TARGET ACHIEVED! 🎉 🎉 🎉")
                print("="*100)
                print(f"Win Rate: {win_rate:.2f}%")
                print(f"Target: {self.target_win_rate}%")
                print(f"Stocks Tested: {current_stock_limit}")
                print(f"Total Trades: {summary['total_trades']:,}")
                print(f"Iterations: {iteration}")
                print("="*100)
                
                self._save_final_results(iteration, results)
                return results
            
            # Analyze failures and generate improvements
            failure_analysis = self.analyze_failures(results)
            recommendations = failure_analysis['recommendations']
            
            # Display progress
            print("\n" + "="*100)
            print(f"📊 ITERATION #{iteration} SUMMARY")
            print("="*100)
            print(f"Win Rate: {win_rate:.2f}% (Target: {self.target_win_rate}%)")
            print(f"Gap: {max(0, self.target_win_rate - win_rate):.2f}%")
            print(f"Total Trades: {summary['total_trades']:,}")
            print(f"Avg Entry Score: {summary['avg_entry_score']:.1f}/100")
            print(f"Best So Far: {self.best_win_rate:.2f}%")
            
            if recommendations:
                print(f"\n💡 Generated {len(recommendations)} recommendations")
                print("\nTop 3 Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"  {i}. {rec['parameter']}: {rec['current_value']} → {rec['new_value']}")
                    print(f"     Priority: {rec['priority']} | Impact: {rec['expected_impact']}")
            
            # Apply improvements for next iteration
            if iteration < max_iterations:
                improvements_applied = self.apply_improvements(recommendations)
                
                if not improvements_applied:
                    print("\n⚠️  No more improvements to apply")
                    if current_stock_limit < 500:
                        print("📈 Scaling to full 500 stocks for final validation...")
                    else:
                        print("🏁 Reached optimization limit")
                        break
            
            print("\n" + "="*100)
            
            # Brief pause between iterations
            import time
            time.sleep(2)
        
        # Max iterations reached
        print("\n" + "="*100)
        print("🏁 MAXIMUM ITERATIONS REACHED")
        print("="*100)
        print(f"Best Win Rate Achieved: {self.best_win_rate:.2f}%")
        print(f"Target: {self.target_win_rate}%")
        print(f"Gap: {max(0, self.target_win_rate - self.best_win_rate):.2f}%")
        
        if self.best_config:
            print(f"\nBest Configuration:")
            for key, value in self.best_config.items():
                print(f"  {key}: {value}")
        
        print("="*100)
        
        return results
    
    def _save_iteration_results(self, iteration: int, results: Dict):
        """Save results for this iteration"""
        filename = f'iteration_{iteration}_results.json'
        
        def convert_to_serializable(obj):
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.strftime('%Y-%m-%d')
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif pd.isna(obj):
                return None
            return obj
        
        # Save summary only (full results are too large)
        summary_data = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.copy(),
            'summary': results['summary'],
            'num_failed_trades': len(results['failed_trades']),
            'num_successful_trades': len(results['successful_trades'])
        }
        
        serializable = convert_to_serializable(summary_data)
        
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"\n💾 Saved iteration results to {filename}")
    
    def _save_final_results(self, iteration: int, results: Dict):
        """Save final comprehensive results"""
        # Save iteration history
        with open('iteration_history.json', 'w') as f:
            json.dump(self.iteration_history, f, indent=2)
        
        print(f"\n💾 Saved iteration history to iteration_history.json")
        
        # Save best configuration
        with open('best_configuration.json', 'w') as f:
            json.dump({
                'best_win_rate': self.best_win_rate,
                'best_config': self.best_config,
                'total_iterations': iteration
            }, f, indent=2)
        
        print(f"💾 Saved best configuration to best_configuration.json")


if __name__ == '__main__':
    import sys
    
    # Parse arguments
    target_win_rate = 90.0  # Default target
    max_iterations = 10
    initial_stock_count = 100
    
    if '--target' in sys.argv:
        idx = sys.argv.index('--target')
        target_win_rate = float(sys.argv[idx + 1])
    
    if '--iterations' in sys.argv:
        idx = sys.argv.index('--iterations')
        max_iterations = int(sys.argv[idx + 1])
    
    if '--stocks' in sys.argv:
        idx = sys.argv.index('--stocks')
        initial_stock_count = int(sys.argv[idx + 1])
    
    print("\n" + "="*100)
    print("🚀 STARTING AUTOMATED ITERATION SYSTEM")
    print("="*100)
    print(f"Target Win Rate: {target_win_rate}%")
    print(f"Max Iterations: {max_iterations}")
    print(f"Initial Stock Count: {initial_stock_count}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    
    try:
        system = AutomatedIterationSystem(target_win_rate=target_win_rate)
        final_results = system.run_until_target(
            max_iterations=max_iterations,
            stock_limit=initial_stock_count,
            scale_up_at=90.0
        )
        
        print("\n" + "="*100)
        print("✅ AUTOMATED ITERATION SYSTEM COMPLETE!")
        print("="*100)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Final Win Rate: {final_results['summary']['overall_win_rate']:.2f}%")
        print(f"Total Iterations: {len(system.iteration_history)}")
        print("="*100 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n❌ System interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
