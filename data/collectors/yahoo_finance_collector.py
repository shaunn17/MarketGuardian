"""
Yahoo Finance Data Collector

This module provides functionality to collect historical stock data
from Yahoo Finance using the yfinance library.
"""

import yfinance as yf
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class YahooFinanceCollector:
    """Collector for Yahoo Finance stock data."""
    
    def __init__(self):
        self.session = None
    
    def get_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            # Add symbol column
            data['Symbol'] = symbol
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Rename columns to standard format
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Symbol']
            
            # Select relevant columns
            data = data[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            logger.info(f"Successfully collected {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_stocks(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch data for multiple stocks.
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Combined DataFrame with data for all symbols
        """
        all_data = []
        
        for symbol in symbols:
            data = self.get_stock_data(symbol, period, interval)
            if not data.empty:
                all_data.append(data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get additional information about a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            logger.error(f"Error getting info for {symbol}: {str(e)}")
            return {}
    
    def get_market_cap(self, symbol: str) -> Optional[float]:
        """Get market capitalization for a stock."""
        info = self.get_stock_info(symbol)
        return info.get('marketCap')
    
    def get_sector(self, symbol: str) -> Optional[str]:
        """Get sector information for a stock."""
        info = self.get_stock_info(symbol)
        return info.get('sector')


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create collector
    collector = YahooFinanceCollector()
    
    # Test single stock
    print("Testing single stock collection...")
    aapl_data = collector.get_stock_data("AAPL", period="6mo")
    print(f"AAPL data shape: {aapl_data.shape}")
    print(aapl_data.head())
    
    # Test multiple stocks
    print("\nTesting multiple stocks collection...")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    multi_data = collector.get_multiple_stocks(symbols, period="3mo")
    print(f"Multi-stock data shape: {multi_data.shape}")
    print(multi_data.head())
    
    # Test stock info
    print("\nTesting stock info...")
    info = collector.get_stock_info("AAPL")
    print(f"AAPL sector: {info.get('sector')}")
    print(f"AAPL market cap: {info.get('marketCap')}")
