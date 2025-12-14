"""
Quick Test Script for World-Class System
Tests with 10 stocks to validate before full 500-stock backtest
"""
import warnings
warnings.filterwarnings('ignore')

from sp500_fetcher import SP500Fetcher
from worldclass_ml_system import WorldClassMLSystem
from ultra_backtester_40y import UltraBacktester40Year
import config

def quick_test():
    """
    Quick test with 10 stocks to validate system
    """
    print("="*80)
    print("🧪 QUICK TEST - World-Class System Validation")
    print("="*80)
    
    # Test stocks - diverse set
    test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
                   'TSLA', 'JPM', 'JNJ', 'V', 'WMT']
    
    print(f"\nTesting with {len(test_stocks)} stocks: {', '.join(test_stocks)}")
    print("\nThis will take a few minutes...\n")
    
    # Initialize backtester (without ML for speed)
    print("📊 Initializing backtester...")
    backtester = UltraBacktester40Year(ml_system=None)
    
    # Run quick backtest
    print("🚀 Running backtest...")
    results = backtester.run_40year_backtest(
        tickers=test_stocks,
        parallel=True
    )
    
    summary = results['summary']
    
    # Show results
    print("\n" + "="*80)
    print("✅ QUICK TEST RESULTS")
    print("="*80)
    print(f"Stocks Tested: {summary['tickers_tested']}")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Winning Trades: {summary['total_winning_trades']}")
    print(f"Losing Trades: {summary['total_losing_trades']}")
    print(f"Win Rate: {summary['overall_win_rate']:.2f}%")
    print(f"Avg Entry Score: {summary['avg_entry_score']:.1f}/100")
    print(f"Profit Factor: {summary['profit_factor']:.2f}")
    print("="*80)
    
    # Analyze failures
    if backtester.failed_trades:
        print(f"\n🔍 Analyzing {len(backtester.failed_trades)} failures...")
        failure_analysis = backtester.analyze_failures_deep()
    
    # Show per-stock results
    print(f"\n📊 Per-Stock Results:")
    print(f"{'Stock':<8} {'Trades':<8} {'Wins':<8} {'Losses':<8} {'Win Rate':<12} {'Return':<10}")
    print("-"*70)
    
    for ticker, result in results['results'].items():
        if 'error' not in result:
            print(f"{ticker:<8} {result['num_trades']:<8} {result['winning_trades']:<8} "
                  f"{result['losing_trades']:<8} {result['win_rate']:<12.2f}% {result['total_return_pct']:<10.2f}%")
    
    # Assessment
    print("\n" + "="*80)
    if summary['overall_win_rate'] >= 95.0:
        print("🎉 ✅ EXCELLENT! System achieving >95% win rate!")
        print("Ready for full 500-stock backtest.")
    elif summary['overall_win_rate'] >= 90.0:
        print("👍 GOOD! Close to target. System validated.")
        print("Consider running full 500-stock backtest.")
    elif summary['overall_win_rate'] >= 85.0:
        print("📊 PROMISING! Strong performance on test set.")
        print("Review failure analysis before full backtest.")
    else:
        print("⚠️ NEEDS IMPROVEMENT. Review criteria and failure patterns.")
        print("Recommend optimizing before full backtest.")
    print("="*80)
    
    return results


if __name__ == '__main__':
    try:
        results = quick_test()
        print("\n✅ Quick test complete! Review results above.")
        print("\nTo run full 500-stock backtest:")
        print("  python run_worldclass_40year_backtest.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
