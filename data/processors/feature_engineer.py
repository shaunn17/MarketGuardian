"""
Feature Engineering for Financial Data

This module provides functionality to create additional features
from raw OHLCV financial data for anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats

logger = logging.getLogger(__name__)


class FinancialFeatureEngineer:
    """Feature engineering for financial time series data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional price features
        """
        df = df.copy()
        
        # Basic price features
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Range_Pct'] = df['Price_Range'] / df['Open'] * 100
        df['Body_Size'] = abs(df['Close'] - df['Open'])
        df['Body_Size_Pct'] = df['Body_Size'] / df['Open'] * 100
        df['Upper_Shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['Lower_Shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
        
        # Price position within the day's range
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['Price_Position'] = df['Price_Position'].fillna(0.5)  # Handle division by zero
        
        # Gap features (if we have previous day's close)
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Pct'] = df['Gap'] / df['Close'].shift(1) * 100
        
        return df
    
    def create_volume_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create volume-based features.
        
        Args:
            df: DataFrame with OHLCV data
            windows: List of window sizes for rolling calculations
            
        Returns:
            DataFrame with additional volume features
        """
        df = df.copy()
        
        # Volume statistics
        for window in windows:
            df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
            df[f'Volume_Std_{window}'] = df['Volume'].rolling(window=window).std()
            df[f'Volume_Ratio_{window}'] = df['Volume'] / df[f'Volume_MA_{window}']
            df[f'Volume_ZScore_{window}'] = (df['Volume'] - df[f'Volume_MA_{window}']) / df[f'Volume_Std_{window}']
        
        # Volume-weighted average price (VWAP)
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
        
        # Price-volume relationship
        df['Price_Volume_Trend'] = df['Close'].pct_change() * df['Volume']
        
        return df
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        df = df.copy()
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_Ratio_{window}'] = df['Close'] / df[f'MA_{window}']
        
        # Exponential moving averages
        for span in [12, 26]:
            df[f'EMA_{span}'] = df['Close'].ewm(span=span).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic Oscillator
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        return df
    
    def create_returns_features(self, df: pd.DataFrame, windows: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Create returns-based features.
        
        Args:
            df: DataFrame with OHLCV data
            windows: List of window sizes for returns calculation
            
        Returns:
            DataFrame with returns features
        """
        df = df.copy()
        
        # Simple returns
        for window in windows:
            df[f'Return_{window}d'] = df['Close'].pct_change(window)
            df[f'Log_Return_{window}d'] = np.log(df['Close'] / df['Close'].shift(window))
        
        # Volatility (rolling standard deviation of returns)
        for window in windows:
            df[f'Volatility_{window}d'] = df['Return_1d'].rolling(window=window).std() * np.sqrt(252)
        
        # Sharpe ratio (simplified)
        for window in windows:
            returns = df['Return_1d'].rolling(window=window)
            df[f'Sharpe_{window}d'] = returns.mean() / returns.std() * np.sqrt(252)
        
        # Maximum drawdown
        for window in windows:
            rolling_max = df['Close'].rolling(window=window).max()
            df[f'Drawdown_{window}d'] = (df['Close'] - rolling_max) / rolling_max
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: DataFrame with OHLCV data and Date column
            
        Returns:
            DataFrame with time features
        """
        df = df.copy()
        
        # Ensure Date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Basic time features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['Quarter'] = df['Date'].dt.quarter
        
        # Cyclical encoding for time features
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        # Market session indicators (simplified)
        df['Is_Monday'] = (df['DayOfWeek'] == 0).astype(int)
        df['Is_Friday'] = (df['DayOfWeek'] == 4).astype(int)
        df['Is_Month_End'] = (df['Date'].dt.is_month_end).astype(int)
        df['Is_Quarter_End'] = (df['Date'].dt.is_quarter_end).astype(int)
        
        return df
    
    def create_correlation_features(self, df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """
        Create correlation-based features for multiple symbols.
        
        Args:
            df: DataFrame with data for multiple symbols
            symbols: List of symbols to calculate correlations
            
        Returns:
            DataFrame with correlation features
        """
        df = df.copy()
        
        # Pivot to get price data for each symbol
        price_pivot = df.pivot(index='Date', columns='Symbol', values='Close')
        
        # Calculate rolling correlations
        for window in [5, 10, 20]:
            corr_matrix = price_pivot.rolling(window=window).corr()
            
            # For each symbol, calculate average correlation with others
            for symbol in symbols:
                if symbol in corr_matrix.index.get_level_values(1):
                    symbol_corrs = corr_matrix.loc[(slice(None), symbol), :]
                    # Remove self-correlation
                    symbol_corrs = symbol_corrs.drop(columns=[symbol])
                    avg_corr = symbol_corrs.mean(axis=1)
                    
                    # Add to original dataframe
                    corr_series = avg_corr.droplevel(1)
                    df.loc[df['Symbol'] == symbol, f'Avg_Correlation_{window}d'] = corr_series.reindex(
                        df[df['Symbol'] == symbol]['Date']
                    ).values
        
        return df
    
    def create_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specifically designed for anomaly detection.
        
        Args:
            df: DataFrame with financial features
            
        Returns:
            DataFrame with anomaly detection features
        """
        df = df.copy()
        
        # Z-scores for key metrics
        key_metrics = ['Close', 'Volume', 'Price_Range_Pct', 'Return_1d']
        for metric in key_metrics:
            if metric in df.columns:
                df[f'{metric}_ZScore'] = stats.zscore(df[metric].fillna(0))
        
        # Percentile ranks
        for metric in key_metrics:
            if metric in df.columns:
                df[f'{metric}_Percentile'] = df[metric].rank(pct=True)
        
        # Outlier indicators
        for metric in key_metrics:
            if metric in df.columns:
                q1 = df[metric].quantile(0.25)
                q3 = df[metric].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df[f'{metric}_Outlier'] = ((df[metric] < lower_bound) | (df[metric] > upper_bound)).astype(int)
        
        # Volatility regime indicators
        if 'Volatility_20d' in df.columns:
            vol_ma = df['Volatility_20d'].rolling(window=50).mean()
            df['High_Volatility'] = (df['Volatility_20d'] > vol_ma * 1.5).astype(int)
            df['Low_Volatility'] = (df['Volatility_20d'] < vol_ma * 0.5).astype(int)
        
        return df
    
    def engineer_all_features(
        self,
        df: pd.DataFrame,
        include_time_features: bool = True,
        include_correlation_features: bool = False,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Raw DataFrame with OHLCV data
            include_time_features: Whether to include time-based features
            include_correlation_features: Whether to include correlation features
            symbols: List of symbols for correlation features
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Apply all feature engineering steps
        df = self.create_price_features(df)
        df = self.create_volume_features(df)
        df = self.create_technical_indicators(df)
        df = self.create_returns_features(df)
        
        if include_time_features:
            df = self.create_time_features(df)
        
        if include_correlation_features and symbols:
            df = self.create_correlation_features(df, symbols)
        
        df = self.create_anomaly_features(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # Forward fill for price data
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].ffill()
        
        # Fill volume with 0
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
        
        # Fill technical indicators with forward fill
        technical_columns = [col for col in df.columns if any(indicator in col for indicator in 
                           ['MA_', 'EMA_', 'BB_', 'RSI', 'MACD', 'Stoch_', 'VWAP'])]
        for col in technical_columns:
            df[col] = df[col].ffill()
        
        # Fill remaining numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def prepare_for_ml(self, df: pd.DataFrame, target_column: str = None) -> tuple:
        """
        Prepare data for machine learning models.
        
        Args:
            df: DataFrame with engineered features
            target_column: Name of target column (if any)
            
        Returns:
            Tuple of (features_df, target_series, feature_names)
        """
        # Select numeric features only
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove non-feature columns
        exclude_columns = ['Date', 'Symbol']
        if target_column:
            exclude_columns.append(target_column)
        
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Create features DataFrame
        features_df = df[feature_columns].copy()
        
        # Handle any remaining missing values
        features_df = features_df.fillna(0)
        
        # Get target if specified
        target_series = None
        if target_column and target_column in df.columns:
            target_series = df[target_column]
        
        return features_df, target_series, feature_columns


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Symbol': 'AAPL',
        'Open': 100 + np.random.randn(100).cumsum(),
        'High': 105 + np.random.randn(100).cumsum(),
        'Low': 95 + np.random.randn(100).cumsum(),
        'Close': 100 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000000, 10000000, 100)
    })
    
    # Create feature engineer
    engineer = FinancialFeatureEngineer()
    
    # Engineer features
    engineered_data = engineer.engineer_all_features(sample_data)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Engineered data shape: {engineered_data.shape}")
    print(f"New features created: {len(engineered_data.columns) - len(sample_data.columns)}")
    
    # Prepare for ML
    features_df, _, feature_names = engineer.prepare_for_ml(engineered_data)
    print(f"Features for ML: {len(feature_names)}")
    print(f"Feature names: {feature_names[:10]}...")  # Show first 10 features
