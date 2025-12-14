# AutoBot - Weekly Swing Trading Transformation

## 🎉 Major Update: Version 3.0

**Date:** December 15, 2025  
**Transformation:** Intraday Trading → **Weekly Swing Trading Strategy**

---

## Summary of Changes

AutoBot has been completely transformed from an **intraday trading system** (5% same-day gains) to a **weekly swing trading system** (5-10% gains over 5 trading days).

---

## Key Changes

### 1. Trading Strategy
- **Before:** Intraday trading - buy and sell same day, target 5% gain
- **After:** Weekly swing trading - hold for 5 days (1 week), target 5-10% gain
- **Trades per week:** 2 trades (1 buy, 1 sell)
- **Holding period:** 5 trading days

### 2. Profit Targets
- **Minimum:** 5% (take profit if reached mid-week)
- **Default:** 7.5% (target profit)
- **Maximum:** 10% (exit immediately when reached)
- **Strategy:** Exit when profit target is achieved, don't wait for end of week

### 3. Risk Management
- **Stop Loss:** Increased from 1% to 3% (wider to accommodate weekly volatility)
- **Position Size:** Increased from $10,000 to $50,000 per position
- **Max Positions:** 2 (1 buy, 1 sell per week)

### 4. Technical Analysis Updates

#### Data Analyzer (`data_analyzer.py`)
- ✅ Added weekly gain calculations (5-day forward returns)
- ✅ Added weekly volatility metrics
- ✅ Added weekly momentum indicators
- ✅ Added weekly RSI (5-day)
- ✅ Added swing strength indicator
- ✅ New method: `get_weekly_patterns()` for analyzing weekly gain patterns

#### Volatility Analyzer (`volatility_analyzer.py`)
- ✅ Added `calculate_weekly_volatility()` method
- ✅ Weekly volatility score (5-15% ideal range for swing trading)
- ✅ Updated `rank_stocks_by_volatility()` to support weekly analysis
- ✅ New swing trading suitability scoring

#### Probability Scorer (`probability_scorer.py`)
- ✅ Updated to calculate probability of weekly gains
- ✅ New method: `_calculate_weekly_technical_score()`
- ✅ Weighted scoring for weekly trading: ML(35%), MA(25%), Technical(20%), Volatility(10%), Swing(10%)
- ✅ Analyzes weekly RSI, weekly momentum, swing strength

#### Stock Predictor (`stock_predictor.py`)
- ✅ ML model now predicts 5-10% weekly gains (not intraday)
- ✅ Added weekly features: Weekly_Gain, Weekly_Momentum, Weekly_RSI, Weekly_Volatility, Swing_Strength
- ✅ Training target changed to `Target_Hit_Min` (5%+ weekly gain)
- ✅ Predictions include weekly-specific data

#### Enhanced Analyzer (`enhanced_analyzer.py`)
- ✅ Discovery logic updated for weekly swing trading
- ✅ Uses weekly volatility ranking
- ✅ Top 5 suggestions show weekly profit targets (5%, 7.5%, 10%)
- ✅ Displays 5-day holding period recommendation
- ✅ Updated output formatting for weekly trading

### 5. Configuration (`config.py`)
New parameters added:
```python
TRADING_TIMEFRAME = 'WEEKLY'
TARGET_GAIN_PERCENT_MIN = 5.0
TARGET_GAIN_PERCENT_MAX = 10.0
TARGET_GAIN_PERCENT = 7.5
TRADES_PER_WEEK = 2
HOLDING_PERIOD_DAYS = 5
STOP_LOSS_PERCENT = 3.0
TAKE_PROFIT_MIN = 5.0
TAKE_PROFIT_MAX = 10.0
```

### 6. Documentation (`AutoBot.Md`)
- ✅ Complete rewrite reflecting weekly swing trading
- ✅ Updated all examples to show weekly trading
- ✅ New best practices for weekly trading
- ✅ Updated profit targets, holding periods, risk management
- ✅ Added weekly trading strategy summary

---

## How to Use

### Run Weekly Analysis (Recommended First)
```bash
python main.py --mode analyze
```

**What you'll see:**
- Weekly swing trading analysis
- Stocks ranked by weekly volatility
- Probability of 5-10% weekly gains
- Top 5 weekly trading suggestions with:
  - Entry prices
  - Weekly profit targets (5%, 7.5%, 10%)
  - 5-day holding period
  - 3% stop-loss levels

### Execute Weekly Trades
```bash
python main.py --mode trade
```

**Trading flow:**
1. System analyzes and recommends 2 stocks for the week
2. You approve each trade manually
3. Hold positions for up to 5 trading days
4. Exit when 5-10% profit is achieved OR at end of week
5. 3% stop-loss automatically protects against losses

---

## Best Practices for Weekly Trading

### Entry Timing
- **Best days:** Monday/Tuesday (start of week)
- Enter positions when MA signals are strong
- Check weekly volatility patterns

### Monitoring
- **Mid-week:** Check progress toward 5-10% targets
- Adjust stops if needed
- Monitor news and market conditions

### Exit Strategy
- **Take 5% profit** if reached by mid-week
- **Hold for 7.5%** if momentum is strong
- **Exit at 10%** immediately
- Close all positions by Friday if targets not hit

### Position Management
- Maximum 2 positions per week (1 buy, 1 sell)
- Don't exceed $50,000 per position
- Use 3% stop-loss on all trades
- Diversify across different sectors

---

## Trading Strategy Comparison

| Feature | Intraday (Old) | Weekly Swing (New) |
|---------|---------------|-------------------|
| Timeframe | Same day | 5 trading days |
| Profit Target | 5% | 5-10% |
| Stop Loss | 1-2% | 3% |
| Trades | Multiple per day | 2 per week |
| Position Size | $10,000 | $50,000 |
| Volatility | Intraday 2-8% | Weekly 5-15% |
| Best Suited | Day traders | Swing traders |
| Monitoring | Constant | Daily check |

---

## Files Modified

1. ✅ `config.py` - Weekly trading parameters
2. ✅ `data_analyzer.py` - Weekly metrics and indicators
3. ✅ `volatility_analyzer.py` - Weekly volatility analysis
4. ✅ `probability_scorer.py` - Weekly probability scoring
5. ✅ `enhanced_analyzer.py` - Weekly discovery logic
6. ✅ `stock_predictor.py` - Weekly ML predictions
7. ✅ `main.py` - Weekly trading mode
8. ✅ `AutoBot.Md` - Complete documentation update

---

## Backward Compatibility

The system retains legacy methods for intraday analysis:
- `calculate_intraday_volatility()` still available
- `get_intraday_patterns()` still available
- Can switch between weekly/intraday by setting `weekly=False` parameter

However, **weekly trading is now the default** for all new analysis.

---

## Testing Recommendations

Before live trading:

1. **Test in simulation mode** for 2-4 weeks (8-16 trades)
2. **Verify weekly profit targets** are achievable
3. **Monitor position holding** for full 5-day periods
4. **Test stop-loss** functionality at 3%
5. **Validate exit strategy** when targets are hit

```bash
# Start with simulation
echo "TRADING_ENV=SIMULATE" >> .env

# Run weekly analysis
python main.py --mode analyze

# Test with trades (paper trading)
python main.py --mode trade
```

---

## Support & Questions

If you have questions about the weekly trading strategy:
1. Review `AutoBot.Md` for complete documentation
2. Check configuration in `config.py`
3. Test in simulation mode first
4. Adjust parameters based on your risk tolerance

---

## Next Steps

1. ✅ Review updated documentation (`AutoBot.Md`)
2. ✅ Update `.env` file if needed
3. ✅ Run analysis mode to see weekly recommendations
4. ✅ Test in simulation mode for 2-4 weeks
5. ✅ Adjust position sizes and risk parameters
6. ✅ Start live trading when comfortable

---

**Happy Swing Trading! 🚀📈**

*Remember: Always take profit when 5-10% target is achieved. Don't wait for end of week!*
