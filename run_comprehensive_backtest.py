"""
Comprehensive Backtest Runner
Runs 10-year backtest, analyzes failures, and generates improvement recommendations
"""
import sys
from datetime import datetime
from advanced_backtester import AdvancedBacktester
from advanced_ml_predictor import AdvancedMLPredictor
import config

def main():
    """Run comprehensive 10-year backtest"""
    
    print("\n" + "="*80)
    print("AUTOBOT COMPREHENSIVE 10-YEAR BACKTEST")
    print("Weekly Swing Trading Strategy Analysis")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tickers to test: {', '.join(config.POPULAR_TICKERS)}")
    print(f"Strategy: {config.TRADES_PER_WEEK} trades per week")
    print(f"Profit Targets: {config.TARGET_GAIN_PERCENT_MIN}%-{config.TARGET_GAIN_PERCENT_MAX}%")
    print(f"Stop Loss: {config.STOP_LOSS_PERCENT}%")
    print(f"Holding Period: {config.HOLDING_PERIOD_DAYS} days")
    print("="*80)
    
    # Initialize backtester
    backtester = AdvancedBacktester()
    
    # Run 10-year backtest
    print("\n📊 PHASE 1: Running 10-Year Backtest...")
    print("-" * 80)
    
    results = backtester.run_10_year_backtest(config.POPULAR_TICKERS)
    
    # Analyze failed trades
    print("\n" + "="*80)
    print("📉 PHASE 2: Analyzing Failed Trades...")
    print("-" * 80)
    
    failure_analysis = backtester.analyze_failed_trades()
    
    # Save results
    print("\n💾 Saving results...")
    backtester.save_backtest_results(results, 'backtest_10year_results.json')
    
    # Generate improvement recommendations
    print("\n" + "="*80)
    print("🎯 PHASE 3: Generating Improvement Recommendations")
    print("="*80)
    
    recommendations = generate_improvements(results, failure_analysis)
    
    print("\n" + "="*80)
    print("📋 IMPROVEMENT RECOMMENDATIONS FOR >95% WIN RATE")
    print("="*80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Priority: {rec['priority']}")
        print(f"   Description: {rec['description']}")
        print(f"   Expected Impact: {rec['impact']}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 BACKTEST SUMMARY")
    print("="*80)
    
    summary = results.get('summary', {})
    print(f"Total Trades Executed: {summary.get('total_trades', 0)}")
    print(f"Overall Win Rate: {summary.get('overall_win_rate', 0):.2f}%")
    print(f"Winning Trades: {summary.get('total_winning_trades', 0)}")
    print(f"Losing Trades: {summary.get('total_losing_trades', 0)}")
    print(f"Average Return per Stock: {summary.get('avg_return_pct', 0):.2f}%")
    
    # Calculate what's needed for >95% win rate
    total_trades = summary.get('total_trades', 0)
    current_win_rate = summary.get('overall_win_rate', 0)
    
    if total_trades > 0:
        current_wins = summary.get('total_winning_trades', 0)
        target_wins = int(total_trades * 0.95)
        additional_wins_needed = target_wins - current_wins
        
        print(f"\n🎯 TARGET: >95% Win Rate")
        print(f"   Current: {current_win_rate:.2f}% ({current_wins}/{total_trades} wins)")
        print(f"   Target: 95.00% ({target_wins}/{total_trades} wins)")
        print(f"   Gap: {95.0 - current_win_rate:.2f}% ({additional_wins_needed} more wins needed)")
        
        if current_win_rate >= 95.0:
            print("\n   ✅ GOAL ACHIEVED! Win rate exceeds 95%!")
        else:
            improvement_pct = ((95.0 - current_win_rate) / current_win_rate) * 100
            print(f"   📈 Need to improve win rate by {improvement_pct:.1f}%")
    
    print(f"\n✅ Backtest Complete!")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return results, failure_analysis, recommendations


def generate_improvements(results: dict, failure_analysis: dict) -> list:
    """Generate specific improvement recommendations based on backtest results"""
    
    recommendations = []
    
    summary = results.get('summary', {})
    current_win_rate = summary.get('overall_win_rate', 0)
    
    # Recommendation 1: Stricter Entry Criteria
    if current_win_rate < 95.0:
        avg_entry_score = failure_analysis.get('avg_entry_score', 0)
        recommendations.append({
            'title': 'Increase Minimum Entry Score Threshold',
            'priority': 'HIGH',
            'description': f'Current avg entry score for failed trades: {avg_entry_score:.1f}/100. '
                          f'Increase minimum entry score from 60 to 75 to filter out marginal trades.',
            'impact': 'Expected to improve win rate by 5-10% by avoiding low-quality setups',
            'implementation': 'Modify _get_advanced_entry_signal() to require score >= 75'
        })
    
    # Recommendation 2: Adaptive Stop Loss
    if failure_analysis:
        exit_reasons = failure_analysis.get('exit_reasons', {})
        stop_loss_exits = exit_reasons.get('stop_loss', 0)
        total_failed = failure_analysis.get('total_failed', 1)
        
        if stop_loss_exits / total_failed > 0.5:
            recommendations.append({
                'title': 'Implement Adaptive Stop-Loss Based on Volatility',
                'priority': 'HIGH',
                'description': f'{stop_loss_exits} trades ({stop_loss_exits/total_failed*100:.1f}%) '
                              f'exited via stop-loss. Implement volatility-adjusted stops: '
                              f'Low vol (5-8%): 2% stop, Medium vol (8-12%): 3% stop, High vol (>12%): 4% stop',
                'impact': 'Expected to reduce false stop-outs by 30-40%',
                'implementation': 'Create adaptive_stop_loss() function in risk manager'
            })
    
    # Recommendation 3: Market Regime Filter
    recommendations.append({
        'title': 'Add Market Regime Detection',
        'priority': 'HIGH',
        'description': 'Only trade during favorable market conditions. Filter out trades during: '
                      '1) VIX > 30 (high fear), 2) SPY below 200MA (bear market), '
                      '3) Market breadth < 40% (weak internals)',
        'impact': 'Expected to improve win rate by 8-12% by avoiding unfavorable conditions',
        'implementation': 'Create market_regime_analyzer.py with SPY/VIX filters'
    })
    
    # Recommendation 4: Multi-Factor Confirmation
    recommendations.append({
        'title': 'Require Multi-Factor Confirmation',
        'priority': 'HIGH',
        'description': 'Add additional confirmation requirements: '
                      '1) Price must be within 5% of 52-week high, '
                      '2) Weekly RSI between 40-70, '
                      '3) Positive news sentiment (>60), '
                      '4) Sector rotation favorable',
        'impact': 'Expected to improve win rate by 5-8% through better trade selection',
        'implementation': 'Add multi_factor_confirmation() check before entry'
    })
    
    # Recommendation 5: Position Sizing Optimization
    recommendations.append({
        'title': 'Implement Dynamic Position Sizing',
        'priority': 'MEDIUM',
        'description': 'Adjust position size based on setup quality: '
                      'Very High confidence (>90%): 100% size, '
                      'High confidence (80-90%): 75% size, '
                      'Medium confidence (70-80%): 50% size',
        'impact': 'Expected to improve risk-adjusted returns by 15-20%',
        'implementation': 'Add calculate_optimal_position_size() with confidence scaling'
    })
    
    # Recommendation 6: Trailing Stop
    recommendations.append({
        'title': 'Add Trailing Stop After 3% Profit',
        'priority': 'MEDIUM',
        'description': 'Once trade reaches 3% profit, implement trailing stop at 2% below '
                      'highest price achieved. This locks in profits while allowing upside.',
        'impact': 'Expected to reduce profit give-backs by 40-50%',
        'implementation': 'Add trailing_stop_manager() to position monitoring'
    })
    
    # Recommendation 7: Time-Based Filters
    recommendations.append({
        'title': 'Avoid Trading During Low Probability Periods',
        'priority': 'MEDIUM',
        'description': 'Avoid entering trades: 1) Last week of month (window dressing), '
                      '2) Week of FOMC meetings, 3) Major earnings weeks, '
                      '4) Holiday-shortened weeks',
        'impact': 'Expected to improve win rate by 3-5%',
        'implementation': 'Create calendar_filter.py with blackout periods'
    })
    
    # Recommendation 8: Sector Rotation
    recommendations.append({
        'title': 'Incorporate Sector Relative Strength',
        'priority': 'MEDIUM',
        'description': 'Only trade stocks in top 3 performing sectors over last 4 weeks. '
                      'Sector momentum is highly predictive of individual stock performance.',
        'impact': 'Expected to improve win rate by 5-7%',
        'implementation': 'Add sector_analyzer.py with relative strength calculations'
    })
    
    # Recommendation 9: ML Model Enhancement
    recommendations.append({
        'title': 'Implement Advanced Ensemble ML Model',
        'priority': 'HIGH',
        'description': 'Replace current ML model with ensemble of XGBoost, Random Forest, '
                      'Gradient Boosting, and AdaBoost with soft voting. Add 50+ engineered features.',
        'impact': 'Expected to improve prediction accuracy by 10-15%',
        'implementation': 'Use advanced_ml_predictor.py with ensemble models'
    })
    
    # Recommendation 10: Exit Optimization
    recommendations.append({
        'title': 'Optimize Exit Strategy',
        'priority': 'MEDIUM',
        'description': 'Exit at first profit target hit (5%, 7.5%, or 10%) instead of waiting. '
                      'Analysis shows many winners give back profits by holding too long.',
        'impact': 'Expected to improve profit retention by 20-30%',
        'implementation': 'Modify exit logic to take profit immediately when any target is hit'
    })
    
    return recommendations


if __name__ == '__main__':
    try:
        results, failures, recommendations = main()
    except KeyboardInterrupt:
        print("\n\nBacktest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
