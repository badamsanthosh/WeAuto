"""
WeAuto - Elite Trading System
Main entry point for the trading system

Usage:
    python src/main.py --mode [simulate|backtest|scan]
    python src/main.py --help
"""
import argparse
import sys
from datetime import datetime

# Import core modules
from core.data_analyzer import DataAnalyzer
from core.risk_manager import RiskManager
import core.config as config

# Import ML modules
from ml.optimized_system import FinalOptimizedSystem
from ml.realistic_system import RealisticEliteBacktester

# Import strategy modules
from strategies.trading_bot import TradingBot
from strategies.stock_discovery import StockDiscovery

# Import utils
from utils.sp500_fetcher import SP500Fetcher


def run_backtest(args):
    """Run comprehensive backtest"""
    print("\n" + "="*80)
    print("🚀 RUNNING BACKTEST")
    print("="*80)
    
    # Get stock universe
    fetcher = SP500Fetcher()
    tickers = fetcher.get_top_liquid_stocks(limit=args.stocks)
    
    # Choose system
    if args.config == 'optimized':
        print("Using: Optimized System (Configuration B+)")
        system = FinalOptimizedSystem()
        results = system.run_optimized_backtest(tickers=tickers, parallel=True)
    else:
        print("Using: Realistic System (Configuration B)")
        system = RealisticEliteBacktester()
        results = system.run_realistic_backtest(tickers=tickers, parallel=True)
    
    # Display results
    summary = results['summary']
    print("\n" + "="*80)
    print("📊 BACKTEST RESULTS")
    print("="*80)
    print(f"Win Rate: {summary['overall_win_rate']:.2f}%")
    print(f"Total Trades: {summary['total_trades']:,}")
    print(f"Profit Factor: {summary['profit_factor']:.2f}")
    print("="*80)
    
    return results


def run_simulation(args):
    """Run live simulation"""
    print("\n" + "="*80)
    print("🤖 STARTING TRADING SIMULATION")
    print("="*80)
    print(f"Mode: {config.TRADING_ENV}")
    print(f"Max Positions: {config.MAX_POSITIONS}")
    print(f"Target Gain: {config.TARGET_GAIN_PERCENT}%")
    print("="*80)
    
    bot = TradingBot()
    bot.run(simulate=True)


def run_scan(args):
    """Scan for trading opportunities"""
    print("\n" + "="*80)
    print("🔍 SCANNING FOR OPPORTUNITIES")
    print("="*80)
    
    scanner = StockDiscovery()
    opportunities = scanner.scan_for_opportunities()
    
    if opportunities:
        print(f"\n✅ Found {len(opportunities)} opportunities:")
        for i, opp in enumerate(opportunities[:10], 1):
            print(f"\n{i}. {opp['ticker']}")
            print(f"   Score: {opp['score']:.1f}/100")
            print(f"   Price: ${opp['price']:.2f}")
            print(f"   Target: ${opp['target']:.2f} ({opp['gain_pct']:.1f}%)")
    else:
        print("\n⚠️  No opportunities found at this time")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='WeAuto - Elite Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest on 50 stocks
  python src/main.py --mode backtest --stocks 50
  
  # Run optimized backtest on 500 stocks
  python src/main.py --mode backtest --config optimized --stocks 500
  
  # Start live simulation
  python src/main.py --mode simulate
  
  # Scan for opportunities
  python src/main.py --mode scan
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['backtest', 'simulate', 'scan'],
        default='scan',
        help='Operating mode (default: scan)'
    )
    
    parser.add_argument(
        '--config',
        choices=['realistic', 'optimized'],
        default='realistic',
        help='Backtest configuration (default: realistic)'
    )
    
    parser.add_argument(
        '--stocks',
        type=int,
        default=50,
        help='Number of stocks to test (default: 50)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='WeAuto v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("🤖 WeAuto - Elite Trading System v1.0.0")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        if args.mode == 'backtest':
            run_backtest(args)
        elif args.mode == 'simulate':
            run_simulation(args)
        elif args.mode == 'scan':
            run_scan(args)
        
        print("\n" + "="*80)
        print("✅ COMPLETED SUCCESSFULLY")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
