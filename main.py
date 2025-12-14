"""
Main Entry Point for Automated Trading Bot
"""
import sys
import argparse
from datetime import datetime, timedelta
from trading_bot import TradingBot

def main():
    """Main function to run the trading bot"""
    parser = argparse.ArgumentParser(description='Automated Trading Bot for US Stocks')
    parser.add_argument('--mode', choices=['analyze', 'trade', 'monitor'], 
                       default='analyze',
                       help='Operation mode: analyze (recommendations only), trade (execute trades), monitor (monitor positions)')
    parser.add_argument('--auto-approve', action='store_true',
                       help='Auto-approve all trades (use with caution!)')
    
    args = parser.parse_args()
    
    # Create bot instance
    bot = TradingBot()
    
    try:
        # Initialize
        if not bot.initialize():
            print("Failed to initialize trading bot")
            return 1
        
        # Run based on mode
        if args.mode == 'analyze':
            print("\nRunning in ENHANCED ANALYSIS mode (no trades will be executed)")
            from enhanced_analyzer import EnhancedAnalyzer
            enhanced = EnhancedAnalyzer()
            results = enhanced.run_enhanced_analysis()
            enhanced.display_results(results)
            
        elif args.mode == 'backtest':
            print("\nRunning in BACKTEST mode")
            from backtester import Backtester
            backtester = Backtester()
            
            tickers = args.tickers if args.tickers else config.POPULAR_TICKERS[:5]
            start_date = args.start_date or (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
            
            print(f"Backtesting from {start_date} to {end_date}")
            for ticker in tickers:
                print(f"\nBacktesting {ticker}...")
                result = backtester.backtest_strategy(ticker, start_date, end_date)
                if 'error' not in result:
                    print(f"  Return: {result['total_return_pct']:.2f}%")
                    print(f"  Win Rate: {result['win_rate']:.2f}%")
                    print(f"  Trades: {result['num_trades']}")
                else:
                    print(f"  Error: {result['error']}")
            
        elif args.mode == 'test':
            print("\nRunning in COMPREHENSIVE TEST mode")
            from backtester import Backtester
            backtester = Backtester()
            
            tickers = args.tickers if args.tickers else config.POPULAR_TICKERS[:5]
            results = backtester.run_comprehensive_test(tickers)
            
            print("\n" + "=" * 80)
            print("TEST SUMMARY")
            print("=" * 80)
            print(f"Average Return: {results['summary']['avg_return_pct']:.2f}%")
            print(f"Average Win Rate: {results['summary']['avg_win_rate']:.2f}%")
            
        elif args.mode == 'forward-test':
            print("\nRunning in FORWARD TEST mode")
            from backtester import Backtester
            backtester = Backtester()
            
            tickers = args.tickers if args.tickers else config.POPULAR_TICKERS[:5]
            for ticker in tickers:
                print(f"\nForward testing {ticker}...")
                result = backtester.forward_test(ticker, days=30)
                if 'error' not in result:
                    print(f"  Return: {result['total_return_pct']:.2f}%")
                    print(f"  Win Rate: {result['win_rate']:.2f}%")
            
        elif args.mode == 'stress-test':
            print("\nRunning in STRESS TEST mode")
            from backtester import Backtester
            backtester = Backtester()
            
            tickers = args.tickers if args.tickers else config.POPULAR_TICKERS[:3]
            for ticker in tickers:
                print(f"\nStress testing {ticker}...")
                results = backtester.stress_test(ticker)
                for scenario, result in results.items():
                    if 'error' not in result:
                        print(f"  {scenario}: {result['total_return_pct']:.2f}% return, {result['win_rate']:.2f}% win rate")
            
        elif args.mode == 'trade':
            print("\nRunning in TRADE mode")
            if args.auto_approve:
                print("⚠️  WARNING: Auto-approve is enabled. All trades will be executed automatically!")
                import config
                config.REQUIRE_APPROVAL = False
            
            bot.run_daily_analysis()
            
        elif args.mode == 'monitor':
            print("\nRunning in MONITOR mode")
            bot.monitor_positions()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        bot.shutdown()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())


