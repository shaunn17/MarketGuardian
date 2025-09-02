"""
Forex Data Collector

This module provides functionality to collect historical forex data
from free FX data sources.
"""

import requests
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)


class FXCollector:
    """Collector for forex data from free sources."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_fx_data_from_alpha_vantage(
        self,
        from_currency: str,
        to_currency: str,
        api_key: Optional[str] = None,
        outputsize: str = 'compact'
    ) -> pd.DataFrame:
        """
        Fetch forex data from Alpha Vantage (free tier available).
        
        Args:
            from_currency: Base currency (e.g., 'USD', 'EUR')
            to_currency: Quote currency (e.g., 'EUR', 'GBP')
            api_key: Alpha Vantage API key (optional, has free tier)
            outputsize: 'compact' (last 100 data points) or 'full' (full dataset)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Use demo API key if none provided (limited requests)
            if not api_key:
                api_key = 'demo'
                logger.warning("Using demo API key. For production, get a free API key from Alpha Vantage.")
            
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'FX_DAILY',
                'from_symbol': from_currency,
                'to_symbol': to_currency,
                'apikey': api_key,
                'outputsize': outputsize
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return pd.DataFrame()
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage note: {data['Note']}")
                return pd.DataFrame()
            
            time_series = data.get('Time Series (FX)', {})
            if not time_series:
                logger.warning(f"No data found for {from_currency}/{to_currency}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                df_data.append({
                    'Date': pd.to_datetime(date_str),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': 0  # FX doesn't have volume
                })
            
            df = pd.DataFrame(df_data)
            df['Symbol'] = f"{from_currency}/{to_currency}"
            df = df.sort_values('Date').reset_index(drop=True)
            
            logger.info(f"Successfully collected {len(df)} records for {from_currency}/{to_currency}")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting FX data for {from_currency}/{to_currency}: {str(e)}")
            return pd.DataFrame()
    
    def get_fx_data_from_exchangerate_api(
        self,
        base_currency: str = 'USD',
        target_currencies: List[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch forex data from ExchangeRate-API (free tier available).
        
        Args:
            base_currency: Base currency
            target_currencies: List of target currencies
            days: Number of days to fetch
            
        Returns:
            DataFrame with exchange rate data
        """
        try:
            if target_currencies is None:
                target_currencies = ['EUR', 'GBP', 'JPY', 'CAD', 'AUD']
            
            all_data = []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            for target_currency in target_currencies:
                url = f"https://api.exchangerate-api.com/v4/history/{base_currency}"
                params = {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d')
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                rates = data.get('rates', {})
                
                for date_str, rate_data in rates.items():
                    if target_currency in rate_data:
                        all_data.append({
                            'Date': pd.to_datetime(date_str),
                            'Symbol': f"{base_currency}/{target_currency}",
                            'Open': rate_data[target_currency],
                            'High': rate_data[target_currency],  # Approximation
                            'Low': rate_data[target_currency],   # Approximation
                            'Close': rate_data[target_currency],
                            'Volume': 0
                        })
                
                # Add delay to respect rate limits
                time.sleep(0.1)
            
            if all_data:
                df = pd.DataFrame(all_data)
                df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
                logger.info(f"Successfully collected {len(df)} records for FX data")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error collecting FX data from ExchangeRate-API: {str(e)}")
            return pd.DataFrame()
    
    def get_fx_data_from_fixer(
        self,
        base_currency: str = 'USD',
        target_currencies: List[str] = None,
        days: int = 30,
        access_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch forex data from Fixer.io (free tier available).
        
        Args:
            base_currency: Base currency
            target_currencies: List of target currencies
            days: Number of days to fetch
            access_key: Fixer.io access key (optional, has free tier)
            
        Returns:
            DataFrame with exchange rate data
        """
        try:
            if not access_key:
                logger.warning("Fixer.io requires an API key. Get a free one at https://fixer.io/")
                return pd.DataFrame()
            
            if target_currencies is None:
                target_currencies = ['EUR', 'GBP', 'JPY', 'CAD', 'AUD']
            
            all_data = []
            end_date = datetime.now()
            
            for i in range(days):
                current_date = end_date - timedelta(days=i)
                date_str = current_date.strftime('%Y-%m-%d')
                
                url = f"http://data.fixer.io/api/{date_str}"
                params = {
                    'access_key': access_key,
                    'base': base_currency,
                    'symbols': ','.join(target_currencies)
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('success'):
                    rates = data.get('rates', {})
                    for target_currency, rate in rates.items():
                        all_data.append({
                            'Date': current_date,
                            'Symbol': f"{base_currency}/{target_currency}",
                            'Open': rate,
                            'High': rate,  # Approximation
                            'Low': rate,   # Approximation
                            'Close': rate,
                            'Volume': 0
                        })
                
                # Add delay to respect rate limits
                time.sleep(0.1)
            
            if all_data:
                df = pd.DataFrame(all_data)
                df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
                logger.info(f"Successfully collected {len(df)} records for FX data")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error collecting FX data from Fixer.io: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_fx_pairs(
        self,
        pairs: List[str],
        days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch data for multiple forex pairs.
        
        Args:
            pairs: List of forex pairs (e.g., ['USD/EUR', 'USD/GBP'])
            days: Number of days to fetch
            
        Returns:
            Combined DataFrame with data for all pairs
        """
        all_data = []
        
        for pair in pairs:
            try:
                from_currency, to_currency = pair.split('/')
                data = self.get_fx_data_from_alpha_vantage(from_currency, to_currency)
                if not data.empty:
                    all_data.append(data)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error processing pair {pair}: {str(e)}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            return pd.DataFrame()


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create collector
    fx_collector = FXCollector()
    
    # Test Alpha Vantage (demo key)
    print("Testing Alpha Vantage collector...")
    usd_eur_data = fx_collector.get_fx_data_from_alpha_vantage('USD', 'EUR')
    print(f"USD/EUR data shape: {usd_eur_data.shape}")
    if not usd_eur_data.empty:
        print(usd_eur_data.head())
    
    # Test ExchangeRate-API
    print("\nTesting ExchangeRate-API collector...")
    fx_data = fx_collector.get_fx_data_from_exchangerate_api('USD', ['EUR', 'GBP'], days=7)
    print(f"FX data shape: {fx_data.shape}")
    if not fx_data.empty:
        print(fx_data.head())
    
    # Test multiple pairs
    print("\nTesting multiple pairs...")
    pairs = ['USD/EUR', 'USD/GBP', 'USD/JPY']
    multi_fx_data = fx_collector.get_multiple_fx_pairs(pairs, days=7)
    print(f"Multi-pair data shape: {multi_fx_data.shape}")
    if not multi_fx_data.empty:
        print(multi_fx_data.head())
