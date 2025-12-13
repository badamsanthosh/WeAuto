# Automated Trading Bot for US Stocks

A professional automated trading system that identifies US stocks with high probability of achieving 5% intraday gains and executes trades through Moomoo exchange with manual approval.

## Features

- **Historical Data Analysis**: Analyzes 40 years of market data to identify patterns and trends
- **Machine Learning Predictions**: Uses XGBoost/Random Forest models to predict stocks likely to gain 5% intraday
- **Technical Indicators**: Comprehensive technical analysis using RSI, MACD, Bollinger Bands, and more
- **Risk Management**: Built-in position sizing, stop-loss, and take-profit mechanisms
- **Moomoo Integration**: Seamless integration with Moomoo OpenAPI for trade execution
- **Approval System**: Manual approval required before executing trades (configurable)
- **Position Monitoring**: Automatic monitoring of open positions with exit signals

## Prerequisites

1. **Python 3.8+**
2. **Moomoo Account** with OpenAPI access
3. **OpenD Gateway** installed and running (required for Moomoo API)

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd auto_trade
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Moomoo OpenD and SDK:**
   - Download and install OpenD from [Moomoo OpenAPI Documentation](https://openapi.moomoo.com/moomoo-api-doc/en/opend/opend-intro.html)
   - Start OpenD on your machine (default: localhost:11111)
   - Install Moomoo Python SDK:
     ```bash
     # Option 1: If available on PyPI
     pip install moomoo-api
     
     # Option 2: Download SDK from Moomoo OpenAPI website and install manually
     # Follow instructions at: https://openapi.moomoo.com/moomoo-api-doc/en/
     ```
   
4. **Test Moomoo Connection:**
   ```bash
   python test_connection.py
   ```

5. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your Moomoo credentials and settings.
   
   **Note**: If `.env.example` doesn't exist, create a `.env` file with:
   ```
   MOOMOO_HOST=127.0.0.1
   MOOMOO_PORT=11111
   MOOMOO_USERNAME=your_username
   MOOMOO_PASSWORD=your_password
   TRADING_ENV=SIMULATE
   ```

## Configuration

Edit `config.py` to customize:

- **Trading Parameters**: Target gain percentage, max positions, position sizes
- **Risk Management**: Stop-loss, take-profit percentages
- **Stock Universe**: List of tickers to analyze
- **ML Model**: Model type (xgboost, random_forest, gradient_boosting)
- **Approval Settings**: Enable/disable manual approval

## Usage

### 1. Analysis Mode (Recommended for first run)
Get trade recommendations without executing trades:
```bash
python main.py --mode analyze
```

### 2. Trade Mode
Get recommendations and execute trades with approval:
```bash
python main.py --mode trade
```

### 3. Monitor Mode
Monitor existing positions and check for exit signals:
```bash
python main.py --mode monitor
```

### 4. Auto-approve Mode (Use with extreme caution!)
```bash
python main.py --mode trade --auto-approve
```

## How It Works

### 1. Data Analysis
- Fetches 40 years of historical data for major US stocks
- Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Analyzes market trends from major indices (SPY, QQQ, DIA)

### 2. Stock Selection
- ML model trained on historical data identifies patterns preceding 5% intraday gains
- Scores stocks based on probability of achieving target gain
- Filters by confidence threshold and risk criteria

### 3. Trade Recommendations
For each selected stock, the system provides:
- **Current Price**: Real-time market price
- **Target Price**: Price for 5% gain (take-profit)
- **Stop Loss Price**: Risk management exit point
- **Quantity**: Position size based on risk management
- **Confidence Score**: ML model probability
- **Technical Indicators**: RSI, volume ratios, etc.

### 4. Approval & Execution
- Displays detailed trade information
- Requests manual approval (if enabled)
- Validates risk management criteria
- Executes trade through Moomoo API upon approval

### 5. Position Monitoring
- Continuously monitors open positions
- Checks for stop-loss and take-profit triggers
- Requests approval before closing positions

## Project Structure

```
auto_trade/
├── config.py              # Configuration settings
├── data_analyzer.py       # Historical data analysis
├── stock_predictor.py     # ML-based stock prediction
├── moomoo_integration.py  # Moomoo API integration
├── risk_manager.py        # Risk management logic
├── trading_bot.py         # Main trading bot orchestrator
├── main.py                # Entry point
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Risk Disclaimer

⚠️ **IMPORTANT**: This is a trading system for educational and research purposes. 

- Trading stocks involves substantial risk of loss
- Past performance does not guarantee future results
- Always test in simulation mode before live trading
- Never invest more than you can afford to lose
- The system requires manual approval by default - use auto-approve with extreme caution
- Ensure compliance with all regulatory requirements

## Model Training

The ML model is automatically trained on first run using historical data from popular US stocks. Training may take several minutes depending on:
- Number of tickers analyzed
- Years of historical data
- System performance

Model performance metrics are displayed during training.

## Troubleshooting

### Moomoo Connection Issues
- Ensure OpenD is running on your machine
- Check that `MOOMOO_HOST` and `MOOMOO_PORT` in `.env` match OpenD settings
- Verify Moomoo credentials are correct

### Data Fetching Issues
- Historical data is cached in `data_cache/` directory
- Delete cache files to force fresh data download
- Check internet connection for yfinance data access

### Model Training Issues
- Ensure sufficient historical data is available
- Check that tickers in `config.POPULAR_TICKERS` are valid
- Reduce `HISTORICAL_YEARS` if training is too slow

## Advanced Configuration

### Custom Stock Universe
Edit `config.py` to add/remove tickers:
```python
POPULAR_TICKERS = ['AAPL', 'MSFT', 'GOOGL', ...]  # Your custom list
```

### Model Parameters
Adjust ML model settings in `stock_predictor.py`:
- XGBoost: n_estimators, max_depth, learning_rate
- Random Forest: n_estimators, max_depth
- Gradient Boosting: n_estimators, max_depth, learning_rate

### Risk Management
Modify risk parameters in `config.py`:
- `MAX_POSITION_SIZE`: Maximum dollar amount per position
- `STOP_LOSS_PERCENT`: Stop-loss percentage
- `TAKE_PROFIT_PERCENT`: Take-profit percentage
- `MAX_POSITIONS`: Maximum concurrent positions

## License

This project is for educational purposes. Use at your own risk.

## Support

For issues related to:
- **Moomoo API**: Refer to [Moomoo OpenAPI Documentation](https://openapi.moomoo.com/)
- **Trading Strategy**: Review code comments and adjust parameters
- **Data Issues**: Check yfinance documentation and data sources

---

**Happy Trading! Remember: Always trade responsibly and within your risk tolerance.**

