"""
Single Run of Elite Backtest
For manual testing without automated iteration
"""
import sys
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

from sp500_fetcher import SP500Fetcher
from elite_backtester_95pct import EliteBacktester95

def main():
    """Run single elite backtest"""
    
    # Parse arguments
    stock_count = 100  # Default
    use_regime = True  # Default: use market regime filter
    
    if '--stocks' in sys.argv:
        idx = sys.argv.index('--stocks')
        stock_count = int(sys.argv[idx + 1])
    
    if '--no-regime' in sys.argv:
        use_regime = False
    
    print("\n" + "="*100)
    print("🚀 ELITE BACKTEST - SINGLE RUN")
    print("="*100)
    print(f"Configuration A (95% Win Rate Target)")
    print(f"Stocks: {stock_count}")
    print(f"Market Regime Filter: {'ON' if use_regime else 'OFF'}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100 + "\n")
    
    # Get stocks
    print("📊 Fetching stock universe...")
    fetcher = SP500Fetcher()
    all_tickers = fetcher.get_top_liquid_stocks(limit=500)
    test_tickers = all_tickers[:stock_count]
    
    print(f"✅ Testing {len(test_tickers)} stocks\n")
    
    # Run backtest
    backtester = EliteBacktester95()
    results = backtester.run_elite_backtest(
        tickers=test_tickers,
        use_market_regime=use_regime,
        parallel=True
    )
    
    summary = results['summary']
    
    # Display results
    print("\n" + "="*100)
    print("📊 FINAL RESULTS")
    print("="*100)
    print(f"Win Rate: {summary['overall_win_rate']:.2f}%")
    print(f"Total Trades: {summary['total_trades']:,}")
    print(f"Winning Trades: {summary['total_winning_trades']:,}")
    print(f"Losing Trades: {summary['total_losing_trades']:,}")
    print(f"Avg Entry Score: {summary['avg_entry_score']:.1f}/100")
    print(f"Profit Factor: {summary['profit_factor']:.2f}")
    
    # Analyze failures
    if results['failed_trades']:
        print(f"\n📉 Failed Trade Analysis:")
        import pandas as pd
        failed_df = pd.DataFrame(results['failed_trades'])
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
    
    # Assessment
    if summary['overall_win_rate'] >= 95.0:
        print("🎉 ✅ EXCEPTIONAL! Win Rate >= 95%!")
    elif summary['overall_win_rate'] >= 90.0:
        print("🎉 ✅ EXCELLENT! Win Rate >= 90%!")
    elif summary['overall_win_rate'] >= 85.0:
        print("✅ GOOD! Win Rate >= 85%")
    else:
        gap = 90.0 - summary['overall_win_rate']
        print(f"📈 Gap to 90%: {gap:.2f}%")
        print("💡 Consider running automated iteration system for improvements")
    
    return results


if __name__ == '__main__':
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\n❌ Backtest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
