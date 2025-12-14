"""
Enhanced Analysis Module
Implements the new analyze mode with multi-source discovery and ranking
"""
from typing import List, Dict, Optional
from datetime import datetime

from stock_discovery import StockDiscovery
from volatility_analyzer import VolatilityAnalyzer
from probability_scorer import ProbabilityScorer
from stock_predictor import StockPredictor
from data_analyzer import DataAnalyzer
import config

class EnhancedAnalyzer:
    """Enhanced analysis with multi-source discovery and ranking"""
    
    def __init__(self):
        self.stock_discovery = StockDiscovery()
        self.vol_analyzer = VolatilityAnalyzer()
        self.probability_scorer = ProbabilityScorer()
        self.predictor = StockPredictor()
        self.data_analyzer = DataAnalyzer()
    
    def discover_top_trending_stocks(self, top_n: int = 10) -> List[Dict]:
        """
        Discover top trending stocks from multiple sources
        
        Args:
            top_n: Number of top stocks to return
            
        Returns:
            List of top trending stocks with metadata
        """
        print("\n" + "=" * 80)
        print("MULTI-SOURCE STOCK DISCOVERY")
        print("=" * 80)
        
        # Discover from all sources
        sources = self.stock_discovery.discover_all_sources(limit_per_source=15)
        
        print("\nDiscovery Results:")
        for source, tickers in sources.items():
            print(f"  {source.replace('_', ' ').title()}: {len(tickers)} stocks")
            if tickers:
                print(f"    {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}")
        
        # Get combined trending (stocks appearing in multiple sources)
        combined = self.stock_discovery.get_combined_trending(min_sources=1)
        
        print(f"\nCombined unique stocks: {len(combined)}")
        
        # Rank by volatility and volume
        print("\nRanking stocks by volatility and volume...")
        ranked = self.vol_analyzer.rank_stocks_by_volatility(combined)
        
        # Take top N
        top_stocks = ranked[:top_n]
        
        print(f"\nTop {top_n} stocks by volatility/volume:")
        for i, stock in enumerate(top_stocks, 1):
            print(f"  {i}. {stock['ticker']}: "
                  f"Vol={stock['volatility_score']:.1f}, "
                  f"Vol_Vol={stock['volume_score']:.1f}, "
                  f"Combined={stock['combined_score']:.1f}")
        
        return top_stocks
    
    def analyze_with_probability(self, tickers: List[str]) -> List[Dict]:
        """
        Analyze stocks and calculate probability/conviction
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            List of stocks with probability and conviction
        """
        print("\n" + "=" * 80)
        print("PROBABILITY AND CONVICTION ANALYSIS")
        print("=" * 80)
        
        results = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Analyzing {ticker}...")
            
            score_data = self.probability_scorer.get_comprehensive_score(ticker)
            if score_data:
                results.append(score_data)
                print(f"  Probability: {score_data['probability_percent']:.2f}%")
                print(f"  Conviction: {score_data['conviction_level']}")
            else:
                print(f"  ⚠️  Could not analyze {ticker}")
        
        # Sort by probability
        results.sort(key=lambda x: x['combined_probability'], reverse=True)
        
        return results
    
    def get_top_5_suggestions(self, top_10_stocks: List[Dict]) -> List[Dict]:
        """
        Get top 5 stock suggestions with entry points and profit targets
        
        Args:
            top_10_stocks: List of top 10 stocks with probability data
            
        Returns:
            List of top 5 suggestions with entry/exit points
        """
        print("\n" + "=" * 80)
        print("TOP 5 STOCK SUGGESTIONS")
        print("=" * 80)
        
        suggestions = []
        
        for stock_data in top_10_stocks[:5]:
            ticker = stock_data.get('ticker') or stock_data.get('symbol', '')
            
            # Get current price and calculate entry/exit
            try:
                data = self.data_analyzer.get_historical_data(ticker, years=0.1)
                if data is None or data.empty:
                    continue
                
                current_price = data['Close'].iloc[-1]
                
                # Calculate entry point (current price or slightly below)
                entry_price = current_price * 0.998  # 0.2% below current
                
                # Calculate profit targets
                target_5pct = entry_price * 1.05
                target_3pct = entry_price * 1.03
                target_2pct = entry_price * 1.02
                
                # Stop loss
                stop_loss = entry_price * (1 - config.STOP_LOSS_PERCENT / 100)
                
                # Get probability and conviction
                prob_data = self.probability_scorer.get_comprehensive_score(ticker)
                
                suggestion = {
                    'ticker': ticker,
                    'current_price': current_price,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target_2pct': target_2pct,
                    'target_3pct': target_3pct,
                    'target_5pct': target_5pct,
                    'max_profit_potential': ((target_5pct - entry_price) / entry_price) * 100,
                    'probability': prob_data.get('probability_percent', 0) if prob_data else 0,
                    'conviction': prob_data.get('conviction_level', 'LOW') if prob_data else 'LOW',
                    'ml_score': prob_data.get('ml_score', 'LOW') if prob_data else 'LOW',
                    'ma_signal': prob_data.get('ma_signal', 'NEUTRAL') if prob_data else 'NEUTRAL',
                }
                
                suggestions.append(suggestion)
                
            except Exception as e:
                print(f"  ⚠️  Error processing {ticker}: {e}")
                continue
        
        # Sort by probability
        suggestions.sort(key=lambda x: x['probability'], reverse=True)
        
        return suggestions
    
    def run_enhanced_analysis(self) -> Dict:
        """
        Run complete enhanced analysis
        
        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "=" * 80)
        print("ENHANCED INTRADAY TRADING ANALYSIS")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Step 1: Discover top 10 trending stocks
        top_10 = self.discover_top_trending_stocks(top_n=10)
        top_10_tickers = [s['ticker'] for s in top_10]
        
        # Step 2: Calculate probability and conviction
        probability_results = self.analyze_with_probability(top_10_tickers)
        
        # Step 3: Get top 5 suggestions
        top_5_suggestions = self.get_top_5_suggestions(probability_results)
        
        return {
            'top_10_trending': top_10,
            'probability_analysis': probability_results,
            'top_5_suggestions': top_5_suggestions,
            'analysis_date': datetime.now().isoformat()
        }
    
    def display_results(self, results: Dict):
        """
        Display analysis results in formatted way
        
        Args:
            results: Analysis results dictionary
        """
        print("\n" + "=" * 80)
        print("ANALYSIS RESULTS SUMMARY")
        print("=" * 80)
        
        # Top 10 Trending
        print("\n📊 TOP 10 TRENDING STOCKS:")
        print("-" * 80)
        for i, stock in enumerate(results['top_10_trending'], 1):
            print(f"{i:2d}. {stock['ticker']:6s} | "
                  f"Vol Score: {stock['volatility_score']:5.1f} | "
                  f"Vol_Vol Score: {stock['volume_score']:5.1f} | "
                  f"Combined: {stock['combined_score']:5.1f}")
        
        # Probability Analysis
        print("\n🎯 PROBABILITY & CONVICTION ANALYSIS:")
        print("-" * 80)
        for stock in results['probability_analysis']:
            conviction_emoji = {
                'VERY_HIGH': '🔥',
                'HIGH': '✅',
                'MEDIUM': '⚡',
                'LOW': '⚠️',
                'VERY_LOW': '❌'
            }.get(stock['conviction_level'], '❓')
            
            print(f"{stock['ticker']:6s} | "
                  f"Probability: {stock['probability_percent']:5.2f}% | "
                  f"Conviction: {conviction_emoji} {stock['conviction_level']:10s} | "
                  f"ML: {stock['ml_score']:4s} | MA: {stock['ma_signal']:10s}")
        
        # Top 5 Suggestions
        print("\n💡 TOP 5 TRADING SUGGESTIONS:")
        print("-" * 80)
        for i, suggestion in enumerate(results['top_5_suggestions'], 1):
            print(f"\n{i}. {suggestion['ticker']}")
            print(f"   Current Price: ${suggestion['current_price']:.2f}")
            print(f"   Entry Price: ${suggestion['entry_price']:.2f}")
            print(f"   Stop Loss: ${suggestion['stop_loss']:.2f}")
            print(f"   Targets: 2%=${suggestion['target_2pct']:.2f} | "
                  f"3%=${suggestion['target_3pct']:.2f} | "
                  f"5%=${suggestion['target_5pct']:.2f}")
            print(f"   Max Profit Potential: {suggestion['max_profit_potential']:.2f}%")
            print(f"   Probability: {suggestion['probability']:.2f}%")
            print(f"   Conviction: {suggestion['conviction']}")
            print(f"   ML Score: {suggestion['ml_score']} | MA Signal: {suggestion['ma_signal']}")

