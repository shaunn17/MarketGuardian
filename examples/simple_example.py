"""
Simple Example: Basic Anomaly Detection

This script provides a simple example of how to use the Financial Anomaly Detection system
with minimal code for quick testing.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from data.collectors.yahoo_finance_collector import YahooFinanceCollector
from data.processors.feature_engineer import FinancialFeatureEngineer
from models.isolation_forest import IsolationForestAnomalyDetector
from models.autoencoder import AutoencoderAnomalyDetector


def main():
    """Simple example of anomaly detection."""
    print("Starting Simple Anomaly Detection Example")
    
    # Step 1: Collect data
    print("\nCollecting stock data...")
    collector = YahooFinanceCollector()
    data = collector.get_stock_data("AAPL", period="1y")
    
    if data.empty:
        print("No data collected. Please check your internet connection.")
        return
    
    print(f"Collected {len(data)} records for AAPL")
    
    # Step 2: Engineer features
    print("\nEngineering features...")
    engineer = FinancialFeatureEngineer()
    features = engineer.engineer_all_features(data)
    print(f"Generated {len(features.columns)} features")
    
    # Step 3: Prepare data for ML
    features_df, _, feature_names = engineer.prepare_for_ml(features)
    print(f"Prepared {len(feature_names)} features for ML")
    
    # Step 4: Train Isolation Forest
    print("\nTraining Isolation Forest...")
    isolation_forest = IsolationForestAnomalyDetector(contamination=0.1)
    isolation_forest.fit(features_df)
    print("Isolation Forest trained")
    
    # Step 5: Detect anomalies
    print("\nDetecting anomalies...")
    predictions, scores, metadata = isolation_forest.detect_anomalies(features_df)
    print(f"Detected {metadata['n_anomalies']} anomalies ({metadata['anomaly_rate']:.2%})")
    
    # Step 6: Visualize results
    print("\nCreating visualization...")
    fig = isolation_forest.plot_anomalies(
        features_df,
        dates=features['Date'],
        prices=features['Close']
    )
    plt.title("AAPL Stock Price with Detected Anomalies")
    plt.show()
    
    # Step 7: Show results
    print("\nResults Summary:")
    print(f"  Total records: {len(data)}")
    print(f"  Anomalies detected: {metadata['n_anomalies']}")
    print(f"  Anomaly rate: {metadata['anomaly_rate']:.2%}")
    print(f"  Threshold used: {metadata['threshold_used']}")
    
    # Show some anomaly details
    anomaly_indices = np.where(predictions == -1)[0]
    if len(anomaly_indices) > 0:
        print(f"\nAnomaly Details (showing first 5):")
        anomaly_data = features.iloc[anomaly_indices[:5]]
        for idx, row in anomaly_data.iterrows():
            print(f"  Date: {row['Date'].date()}, Price: ${row['Close']:.2f}, Score: {scores[idx]:.3f}")
    
    print("\nExample completed successfully!")
    print("Try running the full analysis with 'python examples/run_analysis.py'")


if __name__ == "__main__":
    main()
