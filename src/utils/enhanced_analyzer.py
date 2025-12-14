"""
Enhanced Analysis Module
Implements the new analyze mode with multi-source discovery and ranking
UPDATED FOR WEEKLY SWING TRADING STRATEGY
"""
from typing import List, Dict, Optional
from datetime import datetime

from stock_discovery import StockDiscovery
from volatility_analyzer import VolatilityAnalyzer
from probability_scorer import ProbabilityScorer
from stock_predictor import StockPredictor
from core.data_analyzer import DataAnalyzer
from news_sentiment_analyzer import NewsSentimentAnalyzer
from core import config

class EnhancedAnalyzer:
    """Enhanced analysis with multi-source discovery and ranking for WEEKLY SWING TRADING"""
    
    def __init__(self):
        self.stock_discovery = StockDiscovery()
        self.vol_analyzer = VolatilityAnalyzer()
        self.probability_scorer = ProbabilityScorer()
        self.predictor = StockPredictor()
        self.data_analyzer = DataAnalyzer()
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.use_weekly = True  # Use weekly analysis for swing trading
    
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
        
        # Rank by volatility and volume (WEEKLY for swing trading)
        print("\nRanking stocks by weekly volatility and volume for swing trading...")
        ranked = self.vol_analyzer.rank_stocks_by_volatility(combined, weekly=True)
        
        # Take top N
        top_stocks = ranked[:top_n]
        
        print(f"\nTop {top_n} stocks by weekly volatility/volume:")
        for i, stock in enumerate(top_stocks, 1):
            vol_display = stock.get('avg_weekly_volatility', stock.get('avg_intraday_volatility', 0))
            print(f"  {i}. {stock['ticker']}: "
                  f"Weekly_Vol={vol_display:.1f}%, "
                  f"Vol_Score={stock['volatility_score']:.1f}, "
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
    
    def rank_stocks_by_sentiment(self, tickers: List[str]) -> List[Dict]:
        """
        Rank stocks by news sentiment score
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            List of stocks ranked by sentiment score (highest first)
        """
        print("\n" + "=" * 80)
        print("NEWS SENTIMENT ANALYSIS")
        print("=" * 80)
        
        ranked = self.sentiment_analyzer.rank_stocks_by_sentiment(tickers)
        
        print(f"\nTop stocks by sentiment:")
        for i, stock in enumerate(ranked[:10], 1):
            print(f"  {i}. {stock['ticker']}: "
                  f"Sentiment={stock['sentiment_score']:.1f}/100 "
                  f"({stock['sentiment_label']}), "
                  f"Articles={stock['num_articles']}, "
                  f"Confidence={stock['confidence']}")
        
        return ranked
    
    def discover_profitable_stocks(self, max_stocks: int = 10000, target_count: int = 10) -> List[Dict]:
        """
        Discovery logic: Run until profitable stocks are found for WEEKLY SWING TRADING.
        - First analyzes top 10 trending stocks (by weekly volatility)
        - If not enough found, continues exploring next best available stocks
        - Probability should be more than 90%
        - Conviction should be HIGH or VERY_HIGH
        - Focus on stocks suitable for 5-10% weekly gains
        - Stop when target_count stocks are found OR max_stocks reached
        - Maximum number of stocks to analyze is max_stocks
        
        Args:
            max_stocks: Maximum number of stocks to analyze
            target_count: Number of profitable stocks to find
            
        Returns:
            List of profitable stocks meeting criteria for weekly swing trading
        """
        print("\n" + "=" * 80)
        print("PROFITABLE STOCK DISCOVERY - WEEKLY SWING TRADING")
        print("=" * 80)
        print(f"Target: Find {target_count} stocks with >90% probability and HIGH/VERY_HIGH conviction")
        print(f"Weekly profit target: {config.TARGET_GAIN_PERCENT_MIN}% - {config.TARGET_GAIN_PERCENT_MAX}%")
        print(f"Holding period: {config.HOLDING_PERIOD_DAYS} trading days (1 week)")
        print(f"Maximum stocks to analyze: {max_stocks}")
        print("=" * 80)
        
        profitable_stocks = []
        analyzed_count = 0
        batch_size = 50
        
        # Step 1: Start with top 10 trending stocks
        print("\nStep 1: Analyzing top 10 trending stocks first...")
        top_10_trending = self.discover_top_trending_stocks(top_n=10)
        top_10_tickers = [s['ticker'] for s in top_10_trending]
        
        print(f"Analyzing top 10 trending stocks: {', '.join(top_10_tickers)}")
        
        # Analyze top 10 first
        for ticker in top_10_tickers:
            if len(profitable_stocks) >= target_count:
                break
            
            analyzed_count += 1
            print(f"[{analyzed_count}/{max_stocks}] "
                  f"Analyzing {ticker} (Top 10)... "
                  f"(Found: {len(profitable_stocks)}/{target_count})", end='\r')
            
            try:
                score_data = self.probability_scorer.get_comprehensive_score(ticker)
                
                if not score_data:
                    continue
                
                probability = score_data.get('probability_percent', 0)
                conviction = score_data.get('conviction_level', 'LOW')
                
                if probability > 90.0 and conviction in ['HIGH', 'VERY_HIGH']:
                    sentiment_data = self.sentiment_analyzer.calculate_sentiment_score(ticker)
                    
                    stock_info = {
                        **score_data,
                        'sentiment_score': sentiment_data.get('sentiment_score', 50.0) if sentiment_data else 50.0,
                        'sentiment_label': sentiment_data.get('sentiment_label', 'NEUTRAL') if sentiment_data else 'NEUTRAL',
                    }
                    
                    profitable_stocks.append(stock_info)
                    print(f"\n✅ Found in Top 10: {ticker} - Probability: {probability:.2f}%, "
                          f"Conviction: {conviction}, Sentiment: {stock_info['sentiment_score']:.1f}")
            
            except Exception as e:
                continue
        
        # Step 2: If not enough found, expand search to all discovered stocks
        if len(profitable_stocks) < target_count:
            print(f"\n\nStep 2: Only found {len(profitable_stocks)}/{target_count} in top 10. Expanding search...")
            
            # Discover stocks from all sources
            print("Discovering stocks from all sources...")
            sources = self.stock_discovery.discover_all_sources(limit_per_source=100)
            combined = self.stock_discovery.get_combined_trending(min_sources=1)
            
            # Add popular tickers
            all_candidates = list(set(combined + config.POPULAR_TICKERS))
            
            # Remove already analyzed top 10
            all_candidates = [t for t in all_candidates if t not in top_10_tickers]
            
            # Add sentiment-ranked stocks (analyze top 200 for sentiment)
            print("Analyzing sentiment for additional candidates...")
            try:
                sentiment_ranked = self.sentiment_analyzer.rank_stocks_by_sentiment(all_candidates[:200])
                sentiment_tickers = [s['ticker'] for s in sentiment_ranked[:50]]
                all_candidates = list(set(all_candidates + sentiment_tickers))
                # Remove already analyzed
                all_candidates = [t for t in all_candidates if t not in top_10_tickers]
            except Exception as e:
                print(f"Warning: Sentiment analysis skipped: {e}")
            
            print(f"Total additional candidate stocks: {len(all_candidates)}")
            print(f"Continuing analysis...\n")
            
            # Analyze remaining stocks in batches
            remaining_to_analyze = min(len(all_candidates), max_stocks - analyzed_count)
            
            for i in range(0, remaining_to_analyze, batch_size):
                if len(profitable_stocks) >= target_count:
                    break
                
                batch = all_candidates[i:min(i+batch_size, remaining_to_analyze)]
                
                for ticker in batch:
                    if len(profitable_stocks) >= target_count:
                        break
                    
                    if analyzed_count >= max_stocks:
                        print(f"\n⚠️  Reached maximum analysis limit ({max_stocks} stocks)")
                        break
                    
                    analyzed_count += 1
                    print(f"[{analyzed_count}/{max_stocks}] "
                          f"Analyzing {ticker}... "
                          f"(Found: {len(profitable_stocks)}/{target_count})", end='\r')
                    
                    try:
                        score_data = self.probability_scorer.get_comprehensive_score(ticker)
                        
                        if not score_data:
                            continue
                        
                        probability = score_data.get('probability_percent', 0)
                        conviction = score_data.get('conviction_level', 'LOW')
                        
                        if probability > 90.0 and conviction in ['HIGH', 'VERY_HIGH']:
                            sentiment_data = self.sentiment_analyzer.calculate_sentiment_score(ticker)
                            
                            stock_info = {
                                **score_data,
                                'sentiment_score': sentiment_data.get('sentiment_score', 50.0) if sentiment_data else 50.0,
                                'sentiment_label': sentiment_data.get('sentiment_label', 'NEUTRAL') if sentiment_data else 'NEUTRAL',
                            }
                            
                            profitable_stocks.append(stock_info)
                            print(f"\n✅ Found: {ticker} - Probability: {probability:.2f}%, "
                                  f"Conviction: {conviction}, Sentiment: {stock_info['sentiment_score']:.1f}")
                    
                    except Exception as e:
                        continue
                
                if len(profitable_stocks) >= target_count or analyzed_count >= max_stocks:
                    break
        
        print(f"\n\nDiscovery complete!")
        print(f"Analyzed: {analyzed_count} stocks")
        print(f"Found: {len(profitable_stocks)} profitable stocks (Target: {target_count})")
        
        if len(profitable_stocks) < target_count:
            print(f"⚠️  Warning: Only found {len(profitable_stocks)}/{target_count} profitable stocks")
            print(f"   Consider adjusting criteria or expanding stock universe")
        
        # Sort by probability (highest first)
        profitable_stocks.sort(key=lambda x: x.get('probability_percent', 0), reverse=True)
        
        return profitable_stocks[:target_count] if len(profitable_stocks) > 0 else profitable_stocks
    
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
                
                # Calculate entry point (current price or slightly below for swing entry)
                entry_price = current_price * 0.995  # 0.5% below current (better swing entry)
                
                # Calculate profit targets for WEEKLY SWING TRADING
                target_10pct = entry_price * (1 + config.TARGET_GAIN_PERCENT_MAX / 100)  # Max target
                target_7pct = entry_price * (1 + config.TARGET_GAIN_PERCENT / 100)  # Default target
                target_5pct = entry_price * (1 + config.TARGET_GAIN_PERCENT_MIN / 100)  # Min target
                
                # Stop loss (wider for weekly trades)
                stop_loss = entry_price * (1 - config.STOP_LOSS_PERCENT / 100)
                
                # Get probability and conviction
                prob_data = self.probability_scorer.get_comprehensive_score(ticker)
                
                # Get sentiment data
                sentiment_data = self.sentiment_analyzer.calculate_sentiment_score(ticker)
                
                suggestion = {
                    'ticker': ticker,
                    'current_price': current_price,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target_5pct': target_5pct,  # Minimum weekly target
                    'target_7pct': target_7pct,  # Default weekly target
                    'target_10pct': target_10pct,  # Maximum weekly target
                    'min_profit_potential': ((target_5pct - entry_price) / entry_price) * 100,
                    'default_profit_potential': ((target_7pct - entry_price) / entry_price) * 100,
                    'max_profit_potential': ((target_10pct - entry_price) / entry_price) * 100,
                    'holding_period_days': config.HOLDING_PERIOD_DAYS,
                    'trading_style': 'WEEKLY_SWING',
                    'probability': prob_data.get('probability_percent', 0) if prob_data else 0,
                    'conviction': prob_data.get('conviction_level', 'LOW') if prob_data else 'LOW',
                    'ml_score': prob_data.get('ml_score', 'LOW') if prob_data else 'LOW',
                    'ma_signal': prob_data.get('ma_signal', 'NEUTRAL') if prob_data else 'NEUTRAL',
                    'sentiment_score': sentiment_data.get('sentiment_score', 50.0) if sentiment_data else 50.0,
                    'sentiment_label': sentiment_data.get('sentiment_label', 'NEUTRAL') if sentiment_data else 'NEUTRAL',
                }
                
                suggestions.append(suggestion)
                
            except Exception as e:
                print(f"  ⚠️  Error processing {ticker}: {e}")
                continue
        
        # Sort by probability
        suggestions.sort(key=lambda x: x['probability'], reverse=True)
        
        return suggestions
    
    def run_enhanced_analysis(self, use_profitable_discovery: bool = False) -> Dict:
        """
        Run complete enhanced analysis for WEEKLY SWING TRADING
        
        Args:
            use_profitable_discovery: If True, use new discovery logic to find profitable stocks
        
        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "=" * 80)
        print("ENHANCED WEEKLY SWING TRADING ANALYSIS")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Strategy: 2 trades per week (1 buy, 1 sell)")
        print(f"Profit Target: {config.TARGET_GAIN_PERCENT_MIN}% - {config.TARGET_GAIN_PERCENT_MAX}%")
        print(f"Holding Period: {config.HOLDING_PERIOD_DAYS} trading days")
        print("=" * 80)
        
        if use_profitable_discovery:
            # Use new discovery logic
            profitable_stocks = self.discover_profitable_stocks(max_stocks=10000, target_count=10)
            profitable_tickers = [s['ticker'] for s in profitable_stocks]
            
            # Get top 5 suggestions from profitable stocks
            top_5_suggestions = self.get_top_5_suggestions(profitable_stocks)
            
            # Get sentiment ranking for profitable stocks
            sentiment_ranked = self.rank_stocks_by_sentiment(profitable_tickers)
            
            return {
                'profitable_stocks': profitable_stocks,
                'sentiment_ranking': sentiment_ranked,
                'top_5_suggestions': top_5_suggestions,
                'analysis_date': datetime.now().isoformat()
            }
        else:
            # Original analysis flow
            # Step 1: Discover top 10 trending stocks
            top_10 = self.discover_top_trending_stocks(top_n=10)
            top_10_tickers = [s['ticker'] for s in top_10]
            
            # Step 2: Rank by sentiment
            sentiment_ranked = self.rank_stocks_by_sentiment(top_10_tickers)
            
            # Step 3: Calculate probability and conviction
            probability_results = self.analyze_with_probability(top_10_tickers)
            
            # Step 4: Get top 5 suggestions
            top_5_suggestions = self.get_top_5_suggestions(probability_results)
            
            return {
                'top_10_trending': top_10,
                'sentiment_ranking': sentiment_ranked,
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
        
        # Top 10 Trending (if available - for original analysis mode)
        if 'top_10_trending' in results and results['top_10_trending']:
            print("\n📊 TOP 10 TRENDING STOCKS:")
            print("-" * 80)
            for i, stock in enumerate(results['top_10_trending'], 1):
                print(f"{i:2d}. {stock['ticker']:6s} | "
                      f"Vol Score: {stock['volatility_score']:5.1f} | "
                      f"Vol_Vol Score: {stock['volume_score']:5.1f} | "
                      f"Combined: {stock['combined_score']:5.1f}")
        
        # Probability Analysis (if available - for original analysis mode)
        if 'probability_analysis' in results and results['probability_analysis']:
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
        
        # Sentiment Ranking (if available)
        if 'sentiment_ranking' in results and results['sentiment_ranking']:
            print("\n📰 NEWS SENTIMENT RANKING:")
            print("-" * 80)
            for i, stock in enumerate(results['sentiment_ranking'][:10], 1):
                sentiment_emoji = {
                    'VERY_POSITIVE': '🚀',
                    'POSITIVE': '✅',
                    'NEUTRAL': '➡️',
                    'NEGATIVE': '⚠️',
                    'VERY_NEGATIVE': '❌'
                }.get(stock.get('sentiment_label', 'NEUTRAL'), '❓')
                
                print(f"{i:2d}. {stock['ticker']:6s} | "
                      f"Sentiment: {sentiment_emoji} {stock.get('sentiment_score', 0):5.1f}/100 | "
                      f"Label: {stock.get('sentiment_label', 'NEUTRAL'):15s} | "
                      f"Articles: {stock.get('num_articles', 0):2d} | "
                      f"Confidence: {stock.get('confidence', 'LOW'):6s}")
        
        # Profitable Stocks (if using new discovery)
        if 'profitable_stocks' in results and results['profitable_stocks']:
            print("\n🎯 PROFITABLE STOCKS DISCOVERY (>90% Probability, HIGH/VERY_HIGH Conviction):")
            print("-" * 80)
            for i, stock in enumerate(results['profitable_stocks'], 1):
                conviction_emoji = {
                    'VERY_HIGH': '🔥',
                    'HIGH': '✅',
                    'MEDIUM': '⚡',
                    'LOW': '⚠️',
                    'VERY_LOW': '❌'
                }.get(stock.get('conviction_level', 'LOW'), '❓')
                
                print(f"{i:2d}. {stock['ticker']:6s} | "
                      f"Probability: {stock.get('probability_percent', 0):5.2f}% | "
                      f"Conviction: {conviction_emoji} {stock.get('conviction_level', 'LOW'):10s} | "
                      f"Sentiment: {stock.get('sentiment_score', 50):5.1f}/100")
        
        # Top 5 Suggestions
        print("\n💡 TOP 5 WEEKLY SWING TRADING SUGGESTIONS:")
        print("-" * 80)
        for i, suggestion in enumerate(results['top_5_suggestions'], 1):
            print(f"\n{i}. {suggestion['ticker']} - {suggestion.get('trading_style', 'WEEKLY_SWING')}")
            print(f"   Current Price: ${suggestion['current_price']:.2f}")
            print(f"   Entry Price: ${suggestion['entry_price']:.2f}")
            print(f"   Stop Loss: ${suggestion['stop_loss']:.2f} (-{config.STOP_LOSS_PERCENT}%)")
            print(f"   Weekly Targets:")
            print(f"     • Minimum (5%): ${suggestion['target_5pct']:.2f}")
            if 'target_7pct' in suggestion:
                print(f"     • Default (7.5%): ${suggestion['target_7pct']:.2f}")
            if 'target_10pct' in suggestion:
                print(f"     • Maximum (10%): ${suggestion['target_10pct']:.2f}")
            print(f"   Profit Potential: {suggestion.get('min_profit_potential', 0):.2f}% - {suggestion['max_profit_potential']:.2f}%")
            print(f"   Holding Period: {suggestion.get('holding_period_days', 5)} trading days")
            print(f"   Probability: {suggestion['probability']:.2f}%")
            print(f"   Conviction: {suggestion['conviction']}")
            print(f"   ML Score: {suggestion['ml_score']} | MA Signal: {suggestion['ma_signal']}")
            if 'sentiment_score' in suggestion:
                print(f"   Sentiment: {suggestion['sentiment_score']:.1f}/100 ({suggestion.get('sentiment_label', 'NEUTRAL')})")

