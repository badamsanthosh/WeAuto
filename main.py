"""
Main Entry Point for Automated Trading Bot
"""
import sys
import argparse
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
            print("\nRunning in ANALYSIS mode (no trades will be executed)")
            recommendations = bot.get_trade_recommendations()
            bot.display_recommendations(recommendations)
            
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


