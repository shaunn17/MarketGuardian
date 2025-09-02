"""
Cryptocurrency Data Collector

This module provides functionality to collect historical cryptocurrency data
from Binance and CoinGecko APIs.
"""

import ccxt
import requests
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)


class BinanceCollector:
    """Collector for Binance cryptocurrency data."""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': '',  # Not needed for public data
            'secret': '',  # Not needed for public data
            'sandbox': False,
            'rateLimit': 1200,  # Respect rate limits
        })
    
    def get_crypto_data(
        self,
        symbol: str,
        timeframe: str = '1d',
        limit: int = 1000,
        since: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical cryptocurrency data from Binance.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT', 'ETH/USDT')
            timeframe: Timeframe ('1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M')
            limit: Number of candles to fetch (max 1000)
            since: Timestamp in milliseconds since epoch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            
            if not ohlcv:
                logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Convert timestamp to datetime
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add symbol column
            df['Symbol'] = symbol
            
            # Select and reorder columns
            df = df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            logger.info(f"Successfully collected {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_cryptos(
        self,
        symbols: List[str],
        timeframe: str = '1d',
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch data for multiple cryptocurrency pairs.
        
        Args:
            symbols: List of trading pair symbols
            timeframe: Data timeframe
            limit: Number of candles per symbol
            
        Returns:
            Combined DataFrame with data for all symbols
        """
        all_data = []
        
        for symbol in symbols:
            data = self.get_crypto_data(symbol, timeframe, limit)
            if not data.empty:
                all_data.append(data)
            # Add delay to respect rate limits
            time.sleep(0.1)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            return pd.DataFrame()
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols."""
        try:
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            logger.error(f"Error getting available symbols: {str(e)}")
            return []


class CoinGeckoCollector:
    """Collector for CoinGecko cryptocurrency data."""
    
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = requests.Session()
    
    def get_crypto_data(
        self,
        coin_id: str,
        days: int = 365,
        vs_currency: str = 'usd'
    ) -> pd.DataFrame:
        """
        Fetch historical cryptocurrency data from CoinGecko.
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
            days: Number of days of data (1, 7, 14, 30, 90, 180, 365, max)
            vs_currency: Target currency ('usd', 'eur', 'btc', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': vs_currency,
                'days': days,
                'interval': 'daily' if days > 90 else 'hourly'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or 'prices' not in data:
                logger.warning(f"No data found for coin {coin_id}")
                return pd.DataFrame()
            
            # Extract price data
            prices = data['prices']
            volumes = data.get('total_volumes', [])
            
            # Create DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add volume data if available
            if volumes:
                vol_df = pd.DataFrame(volumes, columns=['timestamp', 'Volume'])
                df = df.merge(vol_df, on='timestamp', how='left')
            else:
                df['Volume'] = 0
            
            # For CoinGecko, we only have close prices, so we'll use them for OHLC
            df['Open'] = df['Close'].shift(1).fillna(df['Close'])
            df['High'] = df['Close']  # Approximation
            df['Low'] = df['Close']   # Approximation
            
            # Add symbol column
            df['Symbol'] = coin_id.upper()
            
            # Select and reorder columns
            df = df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            logger.info(f"Successfully collected {len(df)} records for {coin_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting data for {coin_id}: {str(e)}")
            return pd.DataFrame()
    
    def get_coin_list(self) -> List[Dict[str, Any]]:
        """Get list of available coins."""
        try:
            url = f"{self.base_url}/coins/list"
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting coin list: {str(e)}")
            return []
    
    def get_trending_coins(self) -> List[str]:
        """Get list of trending coin IDs."""
        try:
            url = f"{self.base_url}/search/trending"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            trending = []
            for coin in data.get('coins', []):
                trending.append(coin['item']['id'])
            
            return trending
        except Exception as e:
            logger.error(f"Error getting trending coins: {str(e)}")
            return []


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test Binance collector
    print("Testing Binance collector...")
    binance_collector = BinanceCollector()
    btc_data = binance_collector.get_crypto_data("BTC/USDT", timeframe="1d", limit=100)
    print(f"BTC/USDT data shape: {btc_data.shape}")
    print(btc_data.head())
    
    # Test CoinGecko collector
    print("\nTesting CoinGecko collector...")
    coingecko_collector = CoinGeckoCollector()
    eth_data = coingecko_collector.get_crypto_data("ethereum", days=30)
    print(f"Ethereum data shape: {eth_data.shape}")
    print(eth_data.head())
    
    # Test trending coins
    print("\nTesting trending coins...")
    trending = coingecko_collector.get_trending_coins()
    print(f"Trending coins: {trending[:5]}")
