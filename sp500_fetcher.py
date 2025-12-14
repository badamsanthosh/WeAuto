"""
S&P 500 Stock List Fetcher
Fetches top 500 US stocks from multiple sources
"""
import pandas as pd
import requests
from typing import List
import warnings
warnings.filterwarnings('ignore')

class SP500Fetcher:
    """Fetch S&P 500 stock symbols"""
    
    def __init__(self):
        self.symbols = []
        
    def get_sp500_symbols(self) -> List[str]:
        """
        Get S&P 500 symbols from Wikipedia
        Returns list of stock symbols
        """
        try:
            # Try to get from Wikipedia
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]
            symbols = df['Symbol'].tolist()
            
            # Clean symbols (replace dots with dashes for Yahoo Finance)
            symbols = [s.replace('.', '-') for s in symbols]
            
            print(f"✅ Fetched {len(symbols)} S&P 500 symbols from Wikipedia")
            self.symbols = symbols
            return symbols
            
        except Exception as e:
            print(f"⚠️ Could not fetch from Wikipedia: {e}")
            print("Using fallback list of major stocks...")
            return self.get_fallback_symbols()
    
    def get_fallback_symbols(self) -> List[str]:
        """
        Fallback list of major US stocks if Wikipedia fetch fails
        Top liquid stocks across all sectors
        """
        symbols = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL',
            'ADBE', 'CRM', 'AMD', 'INTC', 'CSCO', 'QCOM', 'TXN', 'IBM', 'INTU', 'NOW',
            'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'NXPI', 'FTNT',
            
            # Healthcare
            'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'DHR', 'BMY',
            'AMGN', 'GILD', 'CVS', 'CI', 'ISRG', 'REGN', 'VRTX', 'ZTS', 'SYK', 'BSX',
            
            # Financials
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'SPGI', 'AXP',
            'USB', 'PNC', 'TFC', 'COF', 'BK', 'MMC', 'AON', 'ICE', 'CME', 'MCO',
            
            # Consumer Discretionary
            'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'ABNB',
            'MAR', 'CMG', 'ORLY', 'YUM', 'GM', 'F', 'ROST', 'DHI', 'LEN', 'HLT',
            
            # Consumer Staples
            'WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'MO', 'CL', 'MDLZ', 'KHC',
            'GIS', 'KMB', 'SYY', 'HSY', 'K', 'CAG', 'STZ', 'TSN', 'CHD', 'CLX',
            
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'WMB',
            'KMI', 'HAL', 'HES', 'DVN', 'FANG', 'BKR', 'MRO', 'APA', 'CTRA', 'EQT',
            
            # Industrials
            'BA', 'HON', 'UPS', 'RTX', 'CAT', 'GE', 'LMT', 'DE', 'MMM', 'UNP',
            'ADP', 'UBER', 'FDX', 'WM', 'GD', 'NOC', 'ITW', 'TT', 'ETN', 'PH',
            
            # Materials
            'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'VMC',
            'MLM', 'CTVA', 'IFF', 'PPG', 'ALB', 'EMN', 'CE', 'FMC', 'MOS', 'CF',
            
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'WELL', 'DLR', 'O', 'VICI',
            'AVB', 'EQR', 'WY', 'SBAC', 'ARE', 'INVH', 'VTR', 'ESS', 'MAA', 'KIM',
            
            # Communication Services
            'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'EA',
            'TTWO', 'MTCH', 'PARA', 'FOXA', 'FOX', 'NWSA', 'NWS', 'OMC', 'IPG', 'LYV',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ED',
            'PEG', 'ES', 'FE', 'EIX', 'ETR', 'AWK', 'DTE', 'PPL', 'AEE', 'CMS',
            
            # Additional high-volume stocks
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'BND',
            'IEMG', 'VNQ', 'GLD', 'SLV', 'USO', 'TLT', 'IEF', 'LQD', 'HYG', 'EMB',
            
            # Growth stocks
            'SHOP', 'SQ', 'PYPL', 'ROKU', 'ZM', 'SNOW', 'DDOG', 'CRWD', 'NET', 'OKTA',
            'TWLO', 'MDB', 'PANW', 'ZS', 'WDAY', 'TEAM', 'DOCU', 'COUP', 'BILL', 'PATH',
            
            # Biotechnology
            'MRNA', 'BNTX', 'BIIB', 'SGEN', 'ALNY', 'BGNE', 'SRPT', 'BMRN', 'EXEL', 'INCY',
            'JAZZ', 'NBIX', 'HALO', 'UTHR', 'RARE', 'FOLD', 'IONS', 'BLUE', 'PTCT', 'ARVN',
            
            # Semiconductors
            'NVDA', 'AMD', 'INTC', 'QCOM', 'TXN', 'AVGO', 'MU', 'ADI', 'LRCX', 'KLAC',
            'AMAT', 'MRVL', 'NXPI', 'MCHP', 'SWKS', 'QRVO', 'ON', 'MPWR', 'ENTG', 'ALGM',
            
            # Software
            'MSFT', 'ORCL', 'ADBE', 'CRM', 'INTU', 'NOW', 'WDAY', 'TEAM', 'DDOG', 'SNOW',
            'PANW', 'FTNT', 'ZS', 'CRWD', 'NET', 'OKTA', 'MDB', 'SPLK', 'ANSS', 'SNPS',
            
            # E-commerce & Retail
            'AMZN', 'WMT', 'HD', 'COST', 'TGT', 'LOW', 'TJX', 'DG', 'DLTR', 'ROST',
            'ETSY', 'W', 'CHWY', 'FTCH', 'REAL', 'RH', 'BBWI', 'ULTA', 'BBY', 'GPS',
            
            # Financial Technology
            'V', 'MA', 'PYPL', 'SQ', 'FIS', 'FISV', 'ADP', 'PAYX', 'BR', 'TRU',
            'GPN', 'JKHY', 'WEX', 'FOUR', 'BILL', 'STNE', 'NU', 'AFRM', 'UPST', 'SOFI',
            
            # Cloud & AI
            'MSFT', 'GOOGL', 'AMZN', 'ORCL', 'CRM', 'NOW', 'SNOW', 'DDOG', 'MDB', 'NET',
            'PLTR', 'AI', 'PATH', 'S', 'ESTC', 'DBX', 'BOX', 'ZM', 'TWLO', 'DOCN',
            
            # Electric Vehicles & Clean Energy
            'TSLA', 'RIVN', 'LCID', 'FSR', 'NIO', 'XPEV', 'LI', 'PLUG', 'FCEL', 'BLDP',
            'BE', 'QS', 'BLNK', 'CHPT', 'RUN', 'ENPH', 'SEDG', 'NOVA', 'ARRY', 'CSIQ',
            
            # Cybersecurity
            'PANW', 'CRWD', 'ZS', 'FTNT', 'NET', 'OKTA', 'S', 'TENB', 'VRNS', 'QLYS',
            'RPD', 'CYBR', 'FEYE', 'SAIL', 'CHKP', 'AVGO', 'CSCO', 'JNPR', 'NTCT', 'PFPT'
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_symbols = []
        for symbol in symbols:
            if symbol not in seen:
                seen.add(symbol)
                unique_symbols.append(symbol)
        
        print(f"✅ Using fallback list of {len(unique_symbols)} major US stocks")
        self.symbols = unique_symbols
        return unique_symbols
    
    def get_top_liquid_stocks(self, limit: int = 500) -> List[str]:
        """
        Get top liquid stocks (combination of S&P 500 and fallback)
        
        Args:
            limit: Maximum number of stocks to return
        
        Returns:
            List of stock symbols
        """
        # Try S&P 500 first
        try:
            symbols = self.get_sp500_symbols()
        except:
            symbols = []
        
        # If we don't have enough, add from fallback
        if len(symbols) < limit:
            fallback = self.get_fallback_symbols()
            # Add unique symbols from fallback
            for symbol in fallback:
                if symbol not in symbols and len(symbols) < limit:
                    symbols.append(symbol)
        
        # Limit to requested number
        symbols = symbols[:limit]
        
        print(f"✅ Final list: {len(symbols)} stocks for backtesting")
        return symbols
    
    def save_symbols_to_file(self, filename: str = 'sp500_symbols.txt'):
        """Save symbols to file"""
        if not self.symbols:
            self.get_top_liquid_stocks()
        
        with open(filename, 'w') as f:
            for symbol in self.symbols:
                f.write(f"{symbol}\n")
        
        print(f"💾 Saved {len(self.symbols)} symbols to {filename}")


if __name__ == '__main__':
    fetcher = SP500Fetcher()
    symbols = fetcher.get_top_liquid_stocks(500)
    
    print(f"\nSample of fetched stocks:")
    print(", ".join(symbols[:20]))
    print(f"... and {len(symbols) - 20} more")
    
    fetcher.save_symbols_to_file()
