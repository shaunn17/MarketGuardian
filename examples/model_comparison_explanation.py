"""
Model Comparison Explanation

This script explains the differences between the three anomaly detection models
and shows how to interpret their results.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collectors.yahoo_finance_collector import YahooFinanceCollector
from data.processors.feature_engineer import FinancialFeatureEngineer
from models.isolation_forest import IsolationForestAnomalyDetector
from models.autoencoder import AutoencoderAnomalyDetector


def explain_model_differences():
    """Explain the differences between anomaly detection models."""
    
    print("🤖 ANOMALY DETECTION MODELS EXPLAINED")
    print("=" * 60)
    
    # Collect sample data
    print("\n📊 Collecting sample data...")
    collector = YahooFinanceCollector()
    data = collector.get_stock_data('AAPL', period='3mo')
    
    # Engineer features
    print("🔧 Engineering features...")
    engineer = FinancialFeatureEngineer()
    features = engineer.engineer_all_features(data)
    features_df, _, _ = engineer.prepare_for_ml(features)
    
    # Train models
    print("\n🚀 Training models...")
    
    # 1. Isolation Forest
    print("   Training Isolation Forest...")
    isolation_forest = IsolationForestAnomalyDetector(contamination=0.1)
    isolation_forest.fit(features_df)
    
    # 2. Autoencoder
    print("   Training Autoencoder...")
    autoencoder = AutoencoderAnomalyDetector(
        encoding_dim=16,
        hidden_dims=[32, 16],
        epochs=20,
        batch_size=32
    )
    autoencoder.fit(features_df, verbose=False)
    
    # Get predictions
    print("\n🔍 Detecting anomalies...")
    
    # Isolation Forest predictions
    if_pred, if_scores, if_meta = isolation_forest.detect_anomalies(features_df)
    
    # Autoencoder predictions
    ae_pred, ae_scores, ae_meta = autoencoder.detect_anomalies(features_df)
    
    # Create comparison visualization
    create_model_comparison_plot(
        data, features_df, 
        (if_pred, if_scores, if_meta, "Isolation Forest"),
        (ae_pred, ae_scores, ae_meta, "Autoencoder")
    )
    
    # Explain the differences
    explain_model_characteristics(if_meta, ae_meta)
    
    return data, features_df, (if_pred, if_scores, if_meta), (ae_pred, ae_scores, ae_meta)


def create_model_comparison_plot(data, features_df, isolation_forest_results, autoencoder_results):
    """Create a comparison plot of different models."""
    
    if_pred, if_scores, if_meta, if_name = isolation_forest_results
    ae_pred, ae_scores, ae_meta, ae_name = autoencoder_results
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Anomaly Detection Models Comparison', fontsize=16, fontweight='bold')
    
    dates = pd.to_datetime(data['Date'])
    prices = data['Close'].values
    
    # 1. Isolation Forest results
    ax1 = axes[0, 0]
    if_normal = if_pred == 1
    if_anomaly = if_pred == -1
    
    ax1.plot(dates[if_normal], prices[if_normal], 'b-', alpha=0.7, label='Normal')
    ax1.scatter(dates[if_anomaly], prices[if_anomaly], 
               color='red', s=50, alpha=0.8, label='Anomalies', zorder=5)
    ax1.set_title(f'{if_name} Results\n({if_meta["n_anomalies"]} anomalies, {if_meta["anomaly_rate"]:.1%})')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Autoencoder results
    ax2 = axes[0, 1]
    ae_normal = ae_pred == 1
    ae_anomaly = ae_pred == -1
    
    ax2.plot(dates[ae_normal], prices[ae_normal], 'b-', alpha=0.7, label='Normal')
    ax2.scatter(dates[ae_anomaly], prices[ae_anomaly], 
               color='red', s=50, alpha=0.8, label='Anomalies', zorder=5)
    ax2.set_title(f'{ae_name} Results\n({ae_meta["n_anomalies"]} anomalies, {ae_meta["anomaly_rate"]:.1%})')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Score distributions
    ax3 = axes[1, 0]
    ax3.hist(if_scores, bins=30, alpha=0.7, label='Isolation Forest', color='blue')
    ax3.set_title('Isolation Forest Score Distribution')
    ax3.set_xlabel('Isolation Score (lower = more anomalous)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Autoencoder reconstruction errors
    ax4 = axes[1, 1]
    ax4.hist(ae_scores, bins=30, alpha=0.7, label='Autoencoder', color='green')
    ax4.axvline(ae_meta['threshold_used'], color='red', linestyle='--', 
               label=f'Threshold: {ae_meta["threshold_used"]:.3f}')
    ax4.set_title('Autoencoder Reconstruction Error Distribution')
    ax4.set_xlabel('Reconstruction Error (higher = more anomalous)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def explain_model_characteristics(if_meta, ae_meta):
    """Explain the characteristics of each model."""
    
    print(f"\n📋 MODEL CHARACTERISTICS:")
    print("=" * 50)
    
    print(f"\n🌲 ISOLATION FOREST:")
    print(f"   • How it works: Builds random trees to isolate data points")
    print(f"   • Anomalies: Points that are easy to isolate (few splits needed)")
    print(f"   • Score meaning: Lower scores = more anomalous")
    print(f"   • Best for: General anomaly detection, fast processing")
    print(f"   • Results: {if_meta['n_anomalies']} anomalies ({if_meta['anomaly_rate']:.1%})")
    
    print(f"\n🧠 AUTOENCODER:")
    print(f"   • How it works: Neural network learns to reconstruct normal data")
    print(f"   • Anomalies: Data that's hard to reconstruct accurately")
    print(f"   • Score meaning: Higher reconstruction error = more anomalous")
    print(f"   • Best for: Complex patterns, sequence anomalies")
    print(f"   • Results: {ae_meta['n_anomalies']} anomalies ({ae_meta['anomaly_rate']:.1%})")
    print(f"   • Threshold: {ae_meta['threshold_used']:.4f}")
    
    print(f"\n🔗 GRAPH NEURAL NETWORK (GNN):")
    print(f"   • How it works: Models relationships between different assets")
    print(f"   • Anomalies: Unusual correlations or relationship patterns")
    print(f"   • Score meaning: Relationship-based anomaly scores")
    print(f"   • Best for: Multi-asset analysis, correlation anomalies")
    print(f"   • Note: Returns predictions per asset, not per data point")
    
    print(f"\n💡 WHEN TO USE WHICH MODEL:")
    print(f"   • Isolation Forest: Quick analysis, general anomalies")
    print(f"   • Autoencoder: Complex patterns, time series anomalies")
    print(f"   • GNN: Multi-asset portfolios, correlation analysis")


def analyze_score_interpretation():
    """Explain how to interpret different score types."""
    
    print(f"\n🎯 SCORE INTERPRETATION GUIDE:")
    print("=" * 40)
    
    print(f"\n🌲 Isolation Forest Scores:")
    print(f"   • Range: 0.0 to 1.0")
    print(f"   • 0.0-0.3: Very anomalous (high confidence)")
    print(f"   • 0.3-0.5: Moderately anomalous")
    print(f"   • 0.5-1.0: Normal behavior")
    print(f"   • Lower scores = more anomalous")
    
    print(f"\n🧠 Autoencoder Scores (Reconstruction Error):")
    print(f"   • Range: 0.0 to ∞ (typically 0.0-2.0)")
    print(f"   • 0.0-0.3: Normal data (easy to reconstruct)")
    print(f"   • 0.3-0.6: Somewhat unusual")
    print(f"   • 0.6+: Very anomalous (hard to reconstruct)")
    print(f"   • Higher scores = more anomalous")
    
    print(f"\n🔗 GNN Scores:")
    print(f"   • Range: Varies based on model architecture")
    print(f"   • Interpretation: Relationship-based anomaly scores")
    print(f"   • Context: Compare scores across different assets")
    print(f"   • Note: One score per asset, not per data point")


def main():
    """Main function to run the model comparison explanation."""
    
    try:
        # Run the explanation
        data, features_df, if_results, ae_results = explain_model_differences()
        
        # Analyze score interpretation
        analyze_score_interpretation()
        
        print(f"\n✅ Model comparison explanation completed!")
        print(f"\n🚀 Next Steps:")
        print(f"   1. Try the interactive dashboard: streamlit run dashboard/app.py")
        print(f"   2. Run simple example: python examples/simple_example.py")
        print(f"   3. Experiment with different models and parameters")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
