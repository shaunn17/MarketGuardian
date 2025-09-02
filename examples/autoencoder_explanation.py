"""
Autoencoder Anomaly Detection Explanation

This script demonstrates how to interpret autoencoder anomaly scores
and provides clear visualizations of what the scores mean.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collectors.yahoo_finance_collector import YahooFinanceCollector
from data.processors.feature_engineer import FinancialFeatureEngineer
from models.autoencoder import AutoencoderAnomalyDetector


def explain_autoencoder_scores():
    """Explain how autoencoder anomaly scores work."""
    
    print("🔍 AUTOENCODER ANOMALY DETECTION EXPLANATION")
    print("=" * 60)
    
    # Step 1: Collect data
    print("\n1. Collecting sample data...")
    collector = YahooFinanceCollector()
    data = collector.get_stock_data('AAPL', period='6mo')
    print(f"   Collected {len(data)} records")
    
    # Step 2: Engineer features
    print("\n2. Engineering features...")
    engineer = FinancialFeatureEngineer()
    features = engineer.engineer_all_features(data)
    features_df, _, _ = engineer.prepare_for_ml(features)
    print(f"   Generated {features_df.shape[1]} features")
    
    # Step 3: Train autoencoder
    print("\n3. Training autoencoder...")
    model = AutoencoderAnomalyDetector(
        encoding_dim=16,
        hidden_dims=[32, 16],
        epochs=30,
        batch_size=32
    )
    model.fit(features_df, verbose=True)
    
    # Step 4: Get reconstruction errors
    print("\n4. Calculating reconstruction errors...")
    reconstruction_errors = model.reconstruction_errors(features_df)
    
    # Step 5: Detect anomalies
    print("\n5. Detecting anomalies...")
    predictions, scores, metadata = model.detect_anomalies(features_df)
    
    # Step 6: Explain the results
    print("\n" + "=" * 60)
    print("📊 UNDERSTANDING THE RESULTS")
    print("=" * 60)
    
    print(f"\n🔢 Reconstruction Error Statistics:")
    print(f"   Min error: {reconstruction_errors.min():.6f}")
    print(f"   Max error: {reconstruction_errors.max():.6f}")
    print(f"   Mean error: {reconstruction_errors.mean():.6f}")
    print(f"   Std error: {reconstruction_errors.std():.6f}")
    print(f"   Threshold: {metadata['threshold_used']:.6f}")
    
    print(f"\n🎯 Anomaly Detection Results:")
    print(f"   Total data points: {len(predictions)}")
    print(f"   Anomalies detected: {metadata['n_anomalies']}")
    print(f"   Normal points: {metadata['n_normal']}")
    print(f"   Anomaly rate: {metadata['anomaly_rate']:.2%}")
    
    print(f"\n💡 How to Interpret Scores:")
    print(f"   • Low scores (0.0-0.3): Normal data - easy to reconstruct")
    print(f"   • Medium scores (0.3-0.6): Somewhat unusual - harder to reconstruct")
    print(f"   • High scores (0.6+): Anomalous - very hard to reconstruct")
    print(f"   • Threshold ({metadata['threshold_used']:.3f}): Decision boundary")
    
    # Step 7: Create comprehensive visualization
    create_autoencoder_visualization(
        data, features_df, reconstruction_errors, 
        predictions, scores, metadata
    )
    
    return model, features_df, reconstruction_errors, predictions, scores, metadata


def create_autoencoder_visualization(data, features_df, reconstruction_errors, 
                                   predictions, scores, metadata):
    """Create comprehensive visualization of autoencoder results."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Autoencoder Anomaly Detection - Complete Analysis', fontsize=16, fontweight='bold')
    
    # 1. Time series with anomalies
    ax1 = axes[0, 0]
    dates = pd.to_datetime(data['Date'])
    prices = data['Close'].values
    
    # Plot normal points
    normal_mask = predictions == 1
    anomaly_mask = predictions == -1
    
    ax1.plot(dates[normal_mask], prices[normal_mask], 'b-', alpha=0.7, label='Normal')
    ax1.scatter(dates[anomaly_mask], prices[anomaly_mask], 
               color='red', s=50, alpha=0.8, label='Anomalies', zorder=5)
    
    ax1.set_title('Price Chart with Detected Anomalies')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reconstruction error distribution
    ax2 = axes[0, 1]
    ax2.hist(reconstruction_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(metadata['threshold_used'], color='red', linestyle='--', 
               label=f'Threshold: {metadata["threshold_used"]:.3f}')
    ax2.set_title('Reconstruction Error Distribution')
    ax2.set_xlabel('Reconstruction Error')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Reconstruction errors over time
    ax3 = axes[1, 0]
    ax3.plot(dates, reconstruction_errors, 'g-', alpha=0.7, label='Reconstruction Error')
    ax3.axhline(metadata['threshold_used'], color='red', linestyle='--', 
               label=f'Threshold: {metadata["threshold_used"]:.3f}')
    
    # Highlight anomalies
    anomaly_dates = dates[anomaly_mask]
    anomaly_errors = reconstruction_errors[anomaly_mask]
    ax3.scatter(anomaly_dates, anomaly_errors, color='red', s=50, alpha=0.8, 
               label='Anomalies', zorder=5)
    
    ax3.set_title('Reconstruction Errors Over Time')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Reconstruction Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Score vs Error scatter plot
    ax4 = axes[1, 1]
    colors = ['red' if pred == -1 else 'blue' for pred in predictions]
    ax4.scatter(reconstruction_errors, scores, c=colors, alpha=0.6)
    ax4.axvline(metadata['threshold_used'], color='red', linestyle='--', 
               label=f'Threshold: {metadata["threshold_used"]:.3f}')
    ax4.set_title('Reconstruction Error vs Anomaly Score')
    ax4.set_xlabel('Reconstruction Error')
    ax4.set_ylabel('Anomaly Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed explanation
    print(f"\n📈 VISUALIZATION EXPLANATION:")
    print(f"   Top Left: Price chart with red dots showing detected anomalies")
    print(f"   Top Right: Histogram of reconstruction errors with threshold line")
    print(f"   Bottom Left: Reconstruction errors over time with anomalies highlighted")
    print(f"   Bottom Right: Scatter plot showing relationship between errors and scores")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   • Anomalies occur when the autoencoder struggles to reconstruct data")
    print(f"   • Higher reconstruction error = more anomalous")
    print(f"   • The threshold separates normal from anomalous data")
    print(f"   • Look for patterns in when anomalies occur")


def analyze_anomaly_patterns(data, predictions, reconstruction_errors):
    """Analyze patterns in detected anomalies."""
    
    print(f"\n🔍 ANOMALY PATTERN ANALYSIS:")
    print("=" * 40)
    
    # Get anomaly data
    anomaly_mask = predictions == -1
    anomaly_data = data[anomaly_mask].copy()
    
    if len(anomaly_data) > 0:
        print(f"\n📅 Anomaly Dates:")
        for i, (idx, row) in enumerate(anomaly_data.iterrows()):
            error = reconstruction_errors[idx]
            print(f"   {i+1}. {row['Date']} - Price: ${row['Close']:.2f}, Error: {error:.4f}")
        
        print(f"\n📊 Anomaly Statistics:")
        print(f"   Average price during anomalies: ${anomaly_data['Close'].mean():.2f}")
        print(f"   Price range during anomalies: ${anomaly_data['Close'].min():.2f} - ${anomaly_data['Close'].max():.2f}")
        print(f"   Average volume during anomalies: {anomaly_data['Volume'].mean():,.0f}")
        
        # Check for clustering
        anomaly_dates = pd.to_datetime(anomaly_data['Date'])
        date_diffs = anomaly_dates.diff().dt.days
        consecutive_anomalies = (date_diffs == 1).sum()
        
        print(f"\n🔗 Anomaly Clustering:")
        print(f"   Consecutive anomaly days: {consecutive_anomalies}")
        print(f"   Clustering rate: {consecutive_anomalies/len(anomaly_data):.2%}")
    else:
        print("   No anomalies detected!")


def main():
    """Main function to run the autoencoder explanation."""
    
    try:
        # Run the explanation
        model, features_df, reconstruction_errors, predictions, scores, metadata = explain_autoencoder_scores()
        
        # Analyze patterns
        data = pd.read_csv('temp_data.csv') if os.path.exists('temp_data.csv') else None
        if data is not None:
            analyze_anomaly_patterns(data, predictions, reconstruction_errors)
        
        print(f"\n✅ Autoencoder explanation completed!")
        print(f"\n💡 Key Takeaways:")
        print(f"   • Autoencoder learns normal patterns and flags deviations")
        print(f"   • Reconstruction error measures how 'unusual' data is")
        print(f"   • Higher error = more anomalous")
        print(f"   • Threshold determines what counts as anomalous")
        print(f"   • Look for patterns in anomaly timing and magnitude")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
