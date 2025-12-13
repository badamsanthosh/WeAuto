# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up Moomoo

1. **Install OpenD Gateway:**
   - Visit: https://openapi.moomoo.com/moomoo-api-doc/en/opend/opend-intro.html
   - Download OpenD for your operating system
   - Install and start OpenD (it should run on localhost:11111)

2. **Install Moomoo Python SDK:**
   - Check if available: `pip install moomoo-api`
   - Or download from Moomoo OpenAPI website

3. **Test Connection:**
   ```bash
   python test_connection.py
   ```

## Step 3: Configure

Create a `.env` file in the project root:

```env
MOOMOO_HOST=127.0.0.1
MOOMOO_PORT=11111
MOOMOO_USERNAME=your_moomoo_username
MOOMOO_PASSWORD=your_moomoo_password
TRADING_ENV=SIMULATE
```

**Important**: Start with `TRADING_ENV=SIMULATE` for paper trading!

## Step 4: Run Your First Analysis

```bash
python main.py --mode analyze
```

This will:
- Analyze 40 years of historical data
- Train ML models
- Generate trade recommendations
- **NOT execute any trades**

## Step 5: Review Recommendations

The system will display:
- Top stock picks with highest probability of 5% intraday gains
- Current prices and target prices
- Stop-loss and take-profit levels
- Confidence scores and technical indicators

## Step 6: Execute Trades (When Ready)

Once you're comfortable with the recommendations:

```bash
python main.py --mode trade
```

For each recommendation, you'll be asked to approve before execution.

## Step 7: Monitor Positions

```bash
python main.py --mode monitor
```

This checks open positions and suggests exits when stop-loss or take-profit is hit.

## Example Output

```
================================================================================
DAILY TRADING ANALYSIS - 2025-01-13 09:30:00
================================================================================

Analyzing stocks for intraday opportunities...

Model trained successfully!
Train accuracy: 0.8234
Test accuracy: 0.7891

================================================================================
TRADE RECOMMENDATIONS
================================================================================

Recommendation #1:
  Ticker: AAPL
  Action: BUY
  Current Price: $185.50
  Target Price: $194.78
  Stop Loss: $181.79
  Take Profit: $194.78
  Quantity: 53 shares
  Position Value: $9831.50
  Probability: 78.50%
  Confidence: HIGH
  RSI: 45.23
  Volume Ratio: 1.85
  Expected Gain: 5.00%

================================================================================
APPROVAL REQUEST
================================================================================
Ticker: AAPL
Action: BUY
Price: $185.50
Quantity: 53 shares
Total Value: $9831.50
Target: $194.78 (5.00% gain)
Stop Loss: $181.79
Confidence: HIGH (78.50%)
================================================================================

Do you approve this trade? (yes/no):
```

## Tips

1. **Start Small**: Begin with small position sizes in simulation mode
2. **Review Recommendations**: Always review the analysis before approving
3. **Market Conditions**: The system works best during normal market hours
4. **Risk Management**: Adjust stop-loss and take-profit in `config.py` based on your risk tolerance
5. **Model Training**: First run may take 5-10 minutes to download and analyze historical data

## Troubleshooting

### "Moomoo package not found"
- Install Moomoo SDK: `pip install moomoo-api`
- Or download from Moomoo website

### "Failed to connect to Moomoo"
- Ensure OpenD is running
- Check host/port in `.env` match OpenD settings
- Verify credentials

### "No trade recommendations"
- Market may be closed
- No stocks meet confidence threshold (adjust in `config.py`)
- Insufficient historical data (wait for first-time data download)

## Next Steps

- Customize stock universe in `config.py`
- Adjust ML model parameters for better predictions
- Set up automated daily runs with cron/scheduler
- Integrate with additional data sources

Happy Trading! 🚀


