# AutoBot - Automated Trading Bot Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Initial Setup](#initial-setup)
4. [Configuration](#configuration)
5. [Running the Bot](#running-the-bot)
6. [Understanding Output](#understanding-output)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)
9. [Advanced Usage](#advanced-usage)

---

## System Overview

AutoBot is an automated trading system that:
- **Discovers trending stocks** from multiple sources (Yahoo Finance, Moomoo, Twitter, Options, News)
- Analyzes 40 years of historical market data
- Uses Machine Learning (XGBoost/Random Forest) to predict stocks with 5% intraday gain potential
- Implements 50/250 Moving Average strategy as primary filter
- Analyzes volatility and volume for optimal intraday trading
- Calculates probability of profit and conviction levels
- Executes trades through Moomoo exchange with manual approval
- Manages risk with stop-loss and take-profit mechanisms
- Comprehensive testing framework (backtesting, forward testing, stress testing)

**Key Features:**
- ✅ **Multi-Source Stock Discovery** (Yahoo Finance, Moomoo, Twitter, Options, News)
- ✅ **Volatility & Volume Analysis** for intraday trading optimization
- ✅ **Probability & Conviction Scoring** (VERY_HIGH, HIGH, MEDIUM, LOW, VERY_LOW)
- ✅ **50/250 Moving Average Golden Cross Strategy**
- ✅ **RSI Overbought/Oversold Detection**
- ✅ **ML-based stock prediction**
- ✅ **Top 10 Trending Stocks** with ranking
- ✅ **Top 5 Trading Suggestions** with entry points and profit targets
- ✅ **Real-time market data integration**
- ✅ **Automated trade execution** (with approval)
- ✅ **Risk management and position monitoring**
- ✅ **Paper trading support** (simulation mode)
- ✅ **Comprehensive Testing Framework** (backtest, forward-test, stress-test)

---

## Prerequisites

Before starting, ensure you have:

### 1. System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python Version**: Python 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: At least 2GB free space for data cache
- **Internet Connection**: Stable connection for data fetching

### 2. Accounts and Access
- **Moomoo Trading Account**: Active account with OpenAPI access enabled
- **Moomoo OpenAPI Access**: Request API access from Moomoo (if not already enabled)

### 3. Software Requirements
- **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
- **pip**: Python package manager (usually included with Python)
- **Git** (optional): For version control

---

## Initial Setup

### Step 1: Verify Python Installation

Open your terminal/command prompt and verify Python is installed:

```bash
python --version
# or
python3 --version
```

**Expected Output:**
```
Python 3.8.0
```
(or any version 3.8 or higher)

If Python is not installed:
1. Visit [python.org/downloads](https://www.python.org/downloads/)
2. Download Python 3.8 or higher for your operating system
3. Run the installer
4. **Important**: Check "Add Python to PATH" during installation

### Step 2: Navigate to Project Directory

```bash
cd /path/to/auto_trade
# Example:
# cd /Users/santhoshbadam/Documents/development/auto_trade
# or on Windows:
# cd C:\Users\YourName\Documents\auto_trade
```

### Step 3: Create Virtual Environment (Recommended)

**Why?** Isolates project dependencies from system Python.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Verify activation:**
Your terminal prompt should show `(venv)` prefix:
```
(venv) user@computer:~/auto_trade$
```

### Step 4: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected Output:**
```
Collecting pandas>=2.0.0
  Downloading pandas-2.1.0-cp38-cp38-win_amd64.whl (10.5 MB)
...
Successfully installed pandas-2.0.0 numpy-1.24.0 ...
```

**Troubleshooting:**
- If `pip` command not found: Use `python -m pip` instead
- If installation fails: Update pip first: `python -m pip install --upgrade pip`
- For permission errors: Use `--user` flag: `pip install --user -r requirements.txt`

### Step 5: Install Moomoo OpenAPI SDK

**Option A: Install from PyPI (if available)**
```bash
pip install moomoo-api
```

**Option B: Manual Installation**
1. Visit [Moomoo OpenAPI Documentation](https://openapi.moomoo.com/moomoo-api-doc/en/)
2. Navigate to "SDK Downloads" section
3. Download Python SDK for your operating system
4. Extract and install:
   ```bash
   cd moomoo-python-sdk
   pip install .
   ```

**Verify Installation:**
```bash
python -c "import moomoo; print('Moomoo SDK installed successfully')"
```

### Step 6: Install and Configure OpenD Gateway

OpenD is required for Moomoo API communication.

**Step 6.1: Download OpenD**
1. Visit: https://openapi.moomoo.com/moomoo-api-doc/en/opend/opend-intro.html
2. Download OpenD for your operating system:
   - **Windows**: `OpenD_Windows.exe`
   - **macOS**: `OpenD_macOS.dmg` or `.pkg`
   - **Linux**: `OpenD_Linux.tar.gz`

**Step 6.2: Install OpenD**

**Windows:**
1. Run `OpenD_Windows.exe`
2. Follow installation wizard
3. OpenD will be installed in Program Files

**macOS:**
1. Open `OpenD_macOS.dmg`
2. Drag OpenD to Applications folder
3. Open Applications → OpenD
4. If security warning appears: System Preferences → Security → Allow

**Linux:**
```bash
tar -xzf OpenD_Linux.tar.gz
cd OpenD
./OpenD
```

**Step 6.3: Start OpenD**

**Windows:**
- Double-click OpenD icon on desktop
- Or run from Start Menu: `OpenD`

**macOS:**
- Open Applications → OpenD
- Or run from terminal:
  ```bash
  /Applications/OpenD.app/Contents/MacOS/OpenD
  ```

**Linux:**
```bash
./OpenD
```

**Step 6.4: Verify OpenD is Running**

OpenD should show:
- Status: "Connected" or "Running"
- Port: 11111 (default)
- Host: 127.0.0.1 (localhost)

**Keep OpenD running** - The bot requires OpenD to be active.

### Step 7: Create Environment Configuration File

Create a `.env` file in the project root directory:

**On macOS/Linux:**
```bash
touch .env
nano .env
```

**On Windows:**
```bash
notepad .env
```

**Add the following content:**
```env
# Moomoo API Configuration
MOOMOO_HOST=127.0.0.1
MOOMOO_PORT=11111
MOOMOO_USERNAME=your_moomoo_username
MOOMOO_PASSWORD=your_moomoo_password

# Trading Environment
# Use 'SIMULATE' for paper trading (RECOMMENDED FOR FIRST RUN)
# Use 'REAL' for live trading (ONLY AFTER THOROUGH TESTING)
TRADING_ENV=SIMULATE
```

**Important:**
- Replace `your_moomoo_username` with your actual Moomoo username
- Replace `your_moomoo_password` with your actual Moomoo password
- **ALWAYS start with `TRADING_ENV=SIMULATE`** for paper trading
- Never commit `.env` file to version control (it's in `.gitignore`)

**Save the file:**
- **nano**: Press `Ctrl+X`, then `Y`, then `Enter`
- **notepad**: Press `Ctrl+S`, then close

### Step 8: Test Moomoo Connection

Before running the bot, test the connection:

```bash
python test_connection.py
```

**Expected Output (Success):**
```
Testing Moomoo API Connection...
Host: 127.0.0.1
Port: 11111
Trading Environment: SIMULATE
------------------------------------------------------------
✅ Successfully connected to Moomoo API!

Fetching account information...
✅ Account info retrieved:
   {'account_id': '...', 'total_assets': 100000.0, ...}

Testing market data retrieval...
✅ Current price for AAPL: $185.50

Fetching current positions...
✅ No open positions (or positions query successful)

✅ Connection test completed successfully!
```

**If Connection Fails:**
- Ensure OpenD is running
- Check `.env` file credentials
- Verify OpenD port matches `MOOMOO_PORT` in `.env`
- See [Troubleshooting](#troubleshooting) section

---

## Configuration

### Understanding Configuration Files

The bot uses two configuration files:

1. **`.env`** - Environment variables (credentials, API settings)
2. **`config.py`** - Trading parameters and strategy settings

### Basic Configuration (config.py)

Open `config.py` in a text editor to customize:

#### Trading Parameters
```python
TARGET_GAIN_PERCENT = 5.0  # Target intraday gain (5%)
MAX_POSITIONS = 2  # Maximum concurrent positions
MAX_POSITION_SIZE = 10000  # Maximum position size in dollars
```

#### Risk Management
```python
STOP_LOSS_PERCENT = 2.0  # Stop loss at 2% below entry
TAKE_PROFIT_PERCENT = 5.0  # Take profit at 5% above entry
```

#### Moving Average Strategy (PRIMARY)
```python
MA_FAST = 50  # 50-day moving average
MA_SLOW = 250  # 250-day moving average
MA_STRATEGY_ENABLED = True  # Enable MA filtering
REQUIRE_GOLDEN_CROSS = True  # Require 50MA > 250MA
REQUIRE_PRICE_ABOVE_MA50 = True  # Require price above 50MA
MIN_MA_SEPARATION_PERCENT = 1.0  # Minimum MA separation
```

#### ML Model Settings
```python
MODEL_TYPE = 'xgboost'  # Options: 'xgboost', 'random_forest', 'gradient_boosting'
MIN_CONFIDENCE_SCORE = 0.7  # Minimum 70% confidence to consider trade
```

#### Approval Settings
```python
REQUIRE_APPROVAL = True  # Require manual approval (RECOMMENDED)
```

**Save changes** after editing `config.py`.

### Stock Universe Configuration

Edit the list of stocks to analyze in `config.py`:

```python
POPULAR_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX',
    'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'LYFT',
    # Add your preferred tickers here
]
```

---

## Running the Bot

### Mode 1: Enhanced Analysis Mode (Recommended First Run)

**Purpose:** Get comprehensive trade recommendations with multi-source discovery WITHOUT executing trades.

**Command:**
```bash
python main.py --mode analyze
```

**What It Does:**
1. **Multi-Source Stock Discovery:**
   - Discovers trending stocks from Yahoo Finance
   - Gets top stocks from Moomoo
   - Analyzes Twitter/X trending mentions
   - Identifies unusual options activity
   - Scans financial news for trending stocks
   
2. **Volatility & Volume Analysis:**
   - Ranks stocks by intraday volatility
   - Analyzes volume patterns and liquidity
   - Identifies stocks with optimal volatility for intraday trading
   
3. **Probability & Conviction Scoring:**
   - Calculates probability of profit using ML models
   - Evaluates 50/250 MA strategy alignment
   - Analyzes technical indicators (RSI, MACD, Bollinger Bands)
   - Assigns conviction levels (VERY_HIGH, HIGH, MEDIUM, LOW, VERY_LOW)
   
4. **Top 10 Trending Stocks:**
   - Displays top 10 stocks sorted by volatility and volume
   - Shows probability and conviction for each
   
5. **Top 5 Trading Suggestions:**
   - Selects best 5 stocks from top 10
   - Provides entry points and profit targets (2%, 3%, 5%)
   - Shows stop-loss levels
   - Displays maximum profit potential
   
6. **Does NOT execute any trades**

**Expected Output:**
```
============================================================
Initializing Automated Trading Bot
============================================================

1. Connecting to Moomoo API...
✅ Successfully connected to Moomoo API

2. Analyzing market trends with 50/250 MA strategy...
  SPY: BULL market, ✅ Golden Cross, MA Separation: 2.15%, RSI: 58.23
  QQQ: BULL market, ✅ Golden Cross, MA Separation: 3.42%, RSI: 62.15
  DIA: BULL market, ✅ Golden Cross, MA Separation: 1.89%, RSI: 55.67

3. Training ML model on historical data...
Training model on 10 tickers...
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
  ML Probability: 78.50%

  📊 MOVING AVERAGE STRATEGY (50/250 MA):
     MA Strategy Score: 85.0/100
     MA Signal: STRONG_BUY
     Golden Cross (50MA > 250MA): ✅ YES
     50-Day MA: $180.50
     250-Day MA: $175.20
     MA Separation: 3.02%
     Distance from 50MA: 2.5%
     Trend Strength: 0.95
     Combined Score (ML + MA): 82.5%

  📈 Technical Indicators:
     RSI: 45.23
     Volume Ratio: 1.85
     Expected Gain: 5.00%
```

**First Run Notes:**
- Initial run may take **5-10 minutes** to download historical data
- Data is cached in `data_cache/` directory for faster subsequent runs
- Model training occurs on first run and when new data is available

### Mode 2: Trade Mode (With Approval)

**Purpose:** Get recommendations AND execute trades after manual approval.

**Command:**
```bash
python main.py --mode trade
```

**What It Does:**
1. Performs all analysis steps (same as analyze mode)
2. For each recommendation, requests your approval
3. Validates risk management criteria
4. Executes approved trades through Moomoo API

**Approval Process:**
```
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
MA Signal: STRONG_BUY
================================================================================

Do you approve this trade? (yes/no):
```

**Response Options:**
- Type `yes` or `y` to approve and execute
- Type `no` or `n` to skip this trade
- Press `Ctrl+C` to cancel all remaining trades

**After Approval:**
```
Executing BUY order for AAPL...
✅ Order executed successfully!
   Order details: {'order_id': '12345', 'status': 'SUBMITTED', ...}
```

### Mode 3: Monitor Mode

**Purpose:** Monitor existing positions and check for exit signals.

**Command:**
```bash
python main.py --mode monitor
```

**What It Does:**
1. Connects to Moomoo API
2. Retrieves all open positions
3. Checks current prices
4. Evaluates stop-loss and take-profit conditions
5. Requests approval to close positions if exit conditions are met

**Expected Output:**
```
================================================================================
MONITORING POSITIONS
================================================================================

Position: AAPL
  Entry: $185.50
  Current: $194.20
  Quantity: 53
  P&L: $461.10 (4.69%)
  ⚠️  EXIT SIGNAL: TAKE_PROFIT
  Requesting approval to close position...
```

### Mode 4: Auto-Approve Mode (⚠️ USE WITH EXTREME CAUTION)

**Purpose:** Execute trades automatically WITHOUT approval.

**Command:**
```bash
python main.py --mode trade --auto-approve
```

**⚠️ WARNING:**
- **ALL trades will be executed automatically**
- No manual review or approval
- Only use after thorough testing in simulation mode
- Ensure risk parameters are properly configured
- Monitor closely during first few runs

**Recommended Usage:**
1. Test extensively in `SIMULATE` mode first
2. Start with small position sizes
3. Monitor first few auto-approved trades closely
4. Have stop-loss and take-profit properly configured

---

## Understanding Output

### Enhanced Analysis Output

**Multi-Source Discovery:**
```
Discovery Results:
  Yahoo Finance: 15 stocks
  Moomoo: 10 stocks
  Twitter: 10 stocks
  Unusual Options: 8 stocks
  News: 7 stocks
```

**Top 10 Trending Stocks:**
```
1. AAPL | Vol Score: 85.2 | Vol_Vol Score: 92.1 | Combined: 88.1
2. TSLA | Vol Score: 78.5 | Vol_Vol Score: 88.3 | Combined: 82.4
```

**Probability & Conviction:**
```
AAPL | Probability: 82.50% | Conviction: 🔥 VERY_HIGH | ML: HIGH | MA: STRONG_BUY
TSLA | Probability: 75.30% | Conviction: ✅ HIGH | ML: MEDIUM | MA: BUY
```

**Top 5 Trading Suggestions:**
```
1. AAPL
   Current Price: $185.50
   Entry Price: $185.13
   Stop Loss: $181.79
   Targets: 2%=$188.83 | 3%=$190.68 | 5%=$194.38
   Max Profit Potential: 5.00%
   Probability: 82.50%
   Conviction: VERY_HIGH
```

### Market Trend Analysis

```
SPY: BULL market, ✅ Golden Cross, MA Separation: 2.15%, RSI: 58.23
```

**Interpretation:**
- **BULL market**: Price above 250MA and 50MA > 250MA
- **✅ Golden Cross**: 50-day MA is above 250-day MA (bullish)
- **MA Separation**: Percentage difference between MAs (higher = stronger trend)
- **RSI**: Relative Strength Index (30-70 is normal range)
- **RSI Signal**: OVERSOLD (≤30), NEUTRAL (30-70), OVERBOUGHT (≥70)

### Trade Recommendation Breakdown

#### Basic Information
- **Ticker**: Stock symbol (e.g., AAPL)
- **Action**: BUY or SELL
- **Current Price**: Real-time market price
- **Target Price**: Price for 5% gain (take-profit level)
- **Stop Loss**: Price for 2% loss (risk management)
- **Quantity**: Number of shares to buy
- **Position Value**: Total dollar amount

#### Moving Average Strategy (PRIMARY)
- **MA Strategy Score**: 0-100 score based on MA criteria
- **MA Signal**: STRONG_BUY, BUY, NEUTRAL, or AVOID
- **Golden Cross**: Whether 50MA > 250MA
- **50-Day MA**: Current 50-day moving average price
- **250-Day MA**: Current 250-day moving average price
- **MA Separation**: Percentage difference between MAs
- **Distance from 50MA**: How far price is from 50MA
- **Combined Score**: Weighted combination of ML + MA (60% ML, 40% MA)

#### Technical Indicators
- **ML Probability**: Machine learning prediction (0-100%)
- **Confidence**: HIGH (≥80%), MEDIUM (≥60%), or LOW (<60%)
- **RSI**: Relative Strength Index (momentum indicator)
- **Volume Ratio**: Current volume vs average volume

### Approval Request Details

When a trade is proposed, review:
1. **Ticker and Action**: Correct stock and direction?
2. **Price**: Current market price reasonable?
3. **Quantity**: Position size acceptable?
4. **Total Value**: Within your risk tolerance?
5. **Target**: 5% gain achievable?
6. **Stop Loss**: Acceptable loss level?
7. **Confidence**: ML probability high enough?
8. **MA Signal**: MA strategy confirms trade?

---

## Troubleshooting

### Issue: "Python not found" or "python: command not found"

**Solution:**
- Verify Python is installed: `python --version`
- Use `python3` instead of `python` on macOS/Linux
- Add Python to PATH environment variable
- On Windows: Reinstall Python with "Add to PATH" checked

### Issue: "pip: command not found"

**Solution:**
```bash
python -m pip install -r requirements.txt
# or
python3 -m pip install -r requirements.txt
```

### Issue: "Failed to connect to Moomoo API"

**Checklist:**
1. ✅ Is OpenD running? (Check system tray/Applications)
2. ✅ Is OpenD listening on port 11111?
3. ✅ Are credentials in `.env` correct?
4. ✅ Does `MOOMOO_HOST` match OpenD host (usually 127.0.0.1)?
5. ✅ Does `MOOMOO_PORT` match OpenD port (usually 11111)?

**Debug Steps:**
```bash
# Test connection separately
python test_connection.py

# Check OpenD status
# Windows: Check system tray icon
# macOS: Check Applications → OpenD
# Linux: Check if process is running: ps aux | grep OpenD
```

**Common Fixes:**
- Restart OpenD
- Check firewall settings (allow OpenD)
- Verify `.env` file format (no extra spaces, correct quotes)
- Try `TRADING_ENV=SIMULATE` first

### Issue: "Moomoo package not found"

**Solution:**
```bash
# Try installing from PyPI
pip install moomoo-api

# Or download SDK manually from Moomoo website
# Then install: pip install /path/to/moomoo-sdk
```

### Issue: "No trade recommendations"

**Possible Causes:**
1. **Market is closed**: Bot works best during market hours
2. **No stocks meet criteria**: Adjust `MIN_CONFIDENCE_SCORE` in `config.py`
3. **MA strategy too strict**: Set `REQUIRE_PRICE_ABOVE_MA250 = False`
4. **Insufficient data**: Wait for first-time data download to complete

**Solutions:**
```python
# In config.py, lower confidence threshold:
MIN_CONFIDENCE_SCORE = 0.6  # Lower from 0.7

# Or relax MA requirements:
REQUIRE_PRICE_ABOVE_MA250 = False
MIN_MA_SEPARATION_PERCENT = 0.5  # Lower from 1.0
```

### Issue: "Model training failed"

**Possible Causes:**
- Insufficient historical data
- Invalid ticker symbols
- Network connectivity issues

**Solutions:**
1. Check internet connection
2. Verify tickers in `config.POPULAR_TICKERS` are valid
3. Reduce `HISTORICAL_YEARS` in `config.py` (e.g., from 40 to 10)
4. Delete `data_cache/` and retry

### Issue: "Order execution failed"

**Check:**
1. Is account funded? (for real trading)
2. Are you in simulation mode? (check `TRADING_ENV` in `.env`)
3. Is market open? (orders can't execute when market is closed)
4. Are position limits exceeded? (check `MAX_POSITIONS` in `config.py`)

### Issue: Slow Performance

**Optimization Tips:**
1. Reduce number of tickers in `POPULAR_TICKERS`
2. Reduce `HISTORICAL_YEARS` (e.g., 10 instead of 40)
3. Use cached data (don't delete `data_cache/` unnecessarily)
4. Close other applications to free up RAM

### Issue: Data Cache Issues

**Clear Cache:**
```bash
# Delete cache directory
rm -rf data_cache/  # macOS/Linux
# or
rmdir /s data_cache  # Windows

# Bot will re-download data on next run
```

---

## Best Practices

### 1. Start with Simulation Mode

**ALWAYS** begin with paper trading:
```env
TRADING_ENV=SIMULATE
```

Test for at least 1-2 weeks before considering live trading.

### 2. Review Recommendations Carefully

Before approving any trade:
- ✅ Check current market conditions
- ✅ Verify stock fundamentals
- ✅ Review news and events
- ✅ Confirm MA strategy signals
- ✅ Validate risk parameters

### 3. Start with Small Positions

Even in simulation, use realistic position sizes:
```python
# In config.py
MAX_POSITION_SIZE = 1000  # Start small
MAX_POSITIONS = 1  # One position at a time initially
```

### 4. Monitor Regularly

- Check positions daily
- Review trade performance weekly
- Adjust parameters based on results
- Keep logs of all trades

### 5. Risk Management

**Never:**
- ❌ Risk more than you can afford to lose
- ❌ Disable stop-loss
- ❌ Override approval system without testing
- ❌ Trade during high volatility without adjustment

**Always:**
- ✅ Use stop-loss (default: 2%)
- ✅ Set take-profit (default: 5%)
- ✅ Limit position sizes
- ✅ Diversify (don't put all capital in one stock)

### 6. Keep OpenD Running

- OpenD must be active for bot to connect
- Set OpenD to start on system boot (optional)
- Monitor OpenD connection status

### 7. Regular Updates

- Update Python packages: `pip install --upgrade -r requirements.txt`
- Check for Moomoo API updates
- Review and adjust strategy parameters monthly

### 8. Documentation

- Keep notes on parameter changes
- Document successful trade patterns
- Record lessons learned
- Track performance metrics

---

## Advanced Usage

### Custom Stock Universe

Edit `config.py`:
```python
POPULAR_TICKERS = [
    # Add your preferred stocks
    'YOUR_TICKER_1',
    'YOUR_TICKER_2',
    # ... more tickers
]
```

### Adjusting ML Model

Change model type in `config.py`:
```python
MODEL_TYPE = 'xgboost'  # Fast, good accuracy
# or
MODEL_TYPE = 'random_forest'  # More interpretable
# or
MODEL_TYPE = 'gradient_boosting'  # Good for complex patterns
```

### Custom Risk Parameters

```python
# In config.py
STOP_LOSS_PERCENT = 1.5  # Tighter stop-loss
TAKE_PROFIT_PERCENT = 6.0  # Higher target
MAX_POSITION_SIZE = 5000  # Smaller positions
MAX_POSITIONS = 3  # More concurrent positions
```

### MA Strategy Customization

```python
# In config.py
MA_FAST = 50  # 50-day MA
MA_SLOW = 250  # 250-day MA
REQUIRE_GOLDEN_CROSS = True  # Must have golden cross
REQUIRE_PRICE_ABOVE_MA50 = True  # Price must be above 50MA
REQUIRE_PRICE_ABOVE_MA250 = False  # Optional: above 250MA
MIN_MA_SEPARATION_PERCENT = 1.0  # Minimum 1% separation
```

### Automated Scheduling

**Linux/macOS (cron):**
```bash
# Edit crontab
crontab -e

# Add line to run daily at 9:30 AM (market open)
30 9 * * 1-5 cd /path/to/auto_trade && /path/to/venv/bin/python main.py --mode trade
```

**Windows (Task Scheduler):**
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: Daily at 9:30 AM
4. Action: Start a program
5. Program: `C:\path\to\venv\Scripts\python.exe`
6. Arguments: `main.py --mode trade`
7. Start in: `C:\path\to\auto_trade`

### Logging and Monitoring

The bot outputs to console. To save logs:

**Linux/macOS:**
```bash
python main.py --mode trade 2>&1 | tee trading_log_$(date +%Y%m%d).log
```

**Windows:**
```cmd
python main.py --mode trade > trading_log_%date%.log 2>&1
```

### Performance Analysis

Track your trades:
1. Export Moomoo trade history
2. Compare bot recommendations vs actual results
3. Analyze win rate, average gain/loss
4. Adjust parameters based on performance

---

## Quick Reference

### Essential Commands

```bash
# Test connection
python test_connection.py

# Enhanced analysis mode (no trades)
python main.py --mode analyze

# Trade mode (with approval)
python main.py --mode trade

# Monitor positions
python main.py --mode monitor

# Backtest strategy
python main.py --mode backtest --tickers AAPL MSFT --start-date 2024-01-01 --end-date 2024-12-31

# Comprehensive testing
python main.py --mode test --tickers AAPL MSFT GOOGL

# Forward testing
python main.py --mode forward-test --tickers AAPL MSFT

# Stress testing
python main.py --mode stress-test --tickers AAPL

# Auto-approve (⚠️ dangerous)
python main.py --mode trade --auto-approve
```

### Execution Modes Summary

| Mode | Command | Purpose |
|------|---------|---------|
| **analyze** | `--mode analyze` | Enhanced multi-source analysis with top 10 trending and top 5 suggestions |
| **trade** | `--mode trade` | Execute trades with manual approval |
| **monitor** | `--mode monitor` | Monitor existing positions |
| **backtest** | `--mode backtest --tickers TICKER --start-date DATE --end-date DATE` | Historical backtesting |
| **test** | `--mode test --tickers TICKER` | Comprehensive testing (backtest + forward + stress) |
| **forward-test** | `--mode forward-test --tickers TICKER` | Out-of-sample forward testing |
| **stress-test** | `--mode stress-test --tickers TICKER` | Stress testing under extreme conditions |

### Configuration Files

- **`.env`**: Credentials and API settings
- **`config.py`**: Trading parameters and strategy

### Important Directories

- **`data_cache/`**: Cached historical data (can be deleted to refresh)
- **`.env`**: Environment variables (keep secure, don't commit)

### Key Settings

- **`TRADING_ENV`**: `SIMULATE` (paper) or `REAL` (live)
- **`REQUIRE_APPROVAL`**: `True` (safe) or `False` (auto-execute)
- **`MA_STRATEGY_ENABLED`**: `True` (use MA filter) or `False` (ML only)

---

## Support and Resources

### Documentation
- **README.md**: General overview
- **QUICKSTART.md**: Quick start guide
- **MA_STRATEGY_GUIDE.md**: Moving Average strategy details
- **AutoBot.Md**: This comprehensive guide

### External Resources
- **Moomoo OpenAPI**: https://openapi.moomoo.com/
- **Python Documentation**: https://docs.python.org/
- **yfinance Documentation**: https://pypi.org/project/yfinance/

### Getting Help

1. Check this documentation first
2. Review troubleshooting section
3. Check Moomoo API documentation
4. Review code comments in source files
5. Test in simulation mode before asking for help

---

## Risk Disclaimer

⚠️ **IMPORTANT WARNINGS:**

1. **Trading Risk**: Stock trading involves substantial risk of loss. You may lose money.

2. **No Guarantees**: Past performance does not guarantee future results. The bot's predictions are not guaranteed.

3. **Test First**: Always test in simulation mode before live trading.

4. **Your Responsibility**: You are responsible for all trading decisions and outcomes.

5. **Not Financial Advice**: This bot is a tool, not financial advice. Consult a financial advisor.

6. **Regulatory Compliance**: Ensure compliance with all applicable trading regulations.

7. **System Failures**: Technical issues, network problems, or API failures may cause losses.

8. **Market Conditions**: The bot may not perform well in all market conditions.

**Use at your own risk. Trade responsibly.**

---

## Conclusion

You now have a complete guide to set up and run AutoBot. Remember:

1. ✅ Start with simulation mode
2. ✅ Test thoroughly before live trading
3. ✅ Review all recommendations
4. ✅ Manage risk carefully
5. ✅ Monitor performance regularly
6. ✅ Adjust parameters as needed

**Happy Trading! 🚀**

---

*Last Updated: 2025-01-13*
*Version: 1.0*


