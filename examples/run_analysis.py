"""
Example Script: Complete Anomaly Detection Analysis

This script demonstrates how to use the Financial Anomaly Detection system
to perform a complete analysis from data collection to anomaly detection.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from data.collectors.yahoo_finance_collector import YahooFinanceCollector
from data.collectors.crypto_collector import BinanceCollector, CoinGeckoCollector
from data.processors.feature_engineer import FinancialFeatureEngineer
from models.isolation_forest import IsolationForestAnomalyDetector
from models.autoencoder import AutoencoderAnomalyDetector
from models.gnn_anomaly import GNNAnomalyDetector
from utils.model_evaluator import AnomalyDetectionEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_stock_data():
    """Collect stock data from Yahoo Finance."""
    logger.info("Collecting stock data from Yahoo Finance...")
    
    collector = YahooFinanceCollector()
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    data = collector.get_multiple_stocks(
        symbols=symbols,
        period='6mo',
        interval='1d'
    )
    
    logger.info(f"Collected {len(data)} records for {len(symbols)} symbols")
    return data


def collect_crypto_data():
    """Collect cryptocurrency data from Binance."""
    logger.info("Collecting cryptocurrency data from Binance...")
    
    collector = BinanceCollector()
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    data = collector.get_multiple_cryptos(
        symbols=symbols,
        timeframe='1d',
        limit=200
    )
    
    logger.info(f"Collected {len(data)} records for {len(symbols)} trading pairs")
    return data


def engineer_features(data):
    """Engineer features from raw data."""
    logger.info("Engineering features...")
    
    engineer = FinancialFeatureEngineer()
    
    # Get symbols for correlation features
    symbols = data['Symbol'].unique().tolist()
    
    # Generate all features
    features = engineer.engineer_all_features(
        data,
        include_time_features=True,
        include_correlation_features=True,
        symbols=symbols
    )
    
    logger.info(f"Generated {len(features.columns)} features")
    return features


def train_models(features):
    """Train anomaly detection models."""
    logger.info("Training anomaly detection models...")
    
    # Prepare data for ML
    engineer = FinancialFeatureEngineer()
    features_df, _, feature_names = engineer.prepare_for_ml(features)
    
    models = {}
    
    # Train Isolation Forest
    logger.info("Training Isolation Forest...")
    isolation_forest = IsolationForestAnomalyDetector(contamination=0.1)
    isolation_forest.fit(features_df)
    models['Isolation Forest'] = isolation_forest
    
    # Train Autoencoder
    logger.info("Training Autoencoder...")
    autoencoder = AutoencoderAnomalyDetector(
        encoding_dim=32,
        hidden_dims=[64, 32],
        epochs=50,
        batch_size=32
    )
    autoencoder.fit(features_df, verbose=True)
    models['Autoencoder'] = autoencoder
    
    # Train GNN
    logger.info("Training Graph Neural Network...")
    gnn = GNNAnomalyDetector(
        model_type='GCN',
        hidden_dim=32,
        output_dim=16,
        epochs=50
    )
    gnn.fit(features, verbose=True)
    models['GNN'] = gnn
    
    logger.info("All models trained successfully")
    return models


def detect_anomalies(models, features):
    """Detect anomalies using trained models."""
    logger.info("Detecting anomalies...")
    
    results = {}
    
    # Prepare data for ML models
    engineer = FinancialFeatureEngineer()
    features_df, _, _ = engineer.prepare_for_ml(features)
    
    for model_name, model in models.items():
        logger.info(f"Detecting anomalies with {model_name}...")
        
        if model_name == 'GNN':
            # GNN needs the full features dataframe
            predictions, scores, metadata = model.detect_anomalies(features)
        else:
            # Other models need the prepared features
            predictions, scores, metadata = model.detect_anomalies(features_df)
        
        results[model_name] = {
            'predictions': predictions,
            'scores': scores,
            'metadata': metadata
        }
        
        logger.info(f"{model_name}: Detected {metadata['n_anomalies']} anomalies ({metadata['anomaly_rate']:.2%})")
    
    return results


def evaluate_models(results, features):
    """Evaluate model performance."""
    logger.info("Evaluating models...")
    
    evaluator = AnomalyDetectionEvaluator()
    
    # Create synthetic true labels for evaluation (in practice, you'd have real labels)
    # For demonstration, we'll create labels based on extreme price movements
    price_changes = features.groupby('Symbol')['Close'].pct_change().abs()
    threshold = price_changes.quantile(0.95)  # Top 5% as anomalies
    y_true = (price_changes > threshold).astype(int).values
    
    # Evaluate each model
    for model_name, result in results.items():
        evaluator.evaluate_model(
            model_name=model_name,
            y_true=y_true,
            y_pred=result['predictions'],
            anomaly_scores=result['scores'],
            metadata=result['metadata']
        )
    
    # Compare models
    comparison = evaluator.compare_models()
    logger.info("Model comparison completed")
    
    return evaluator, comparison


def visualize_results(results, features):
    """Visualize anomaly detection results."""
    logger.info("Creating visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Anomaly Detection Results', fontsize=16)
    
    # Plot 1: Price data with anomalies (Isolation Forest)
    ax1 = axes[0, 0]
    model_name = 'Isolation Forest'
    if model_name in results:
        result = results[model_name]
        predictions = result['predictions']
        
        for symbol in features['Symbol'].unique():
            symbol_data = features[features['Symbol'] == symbol]
            symbol_predictions = predictions[:len(symbol_data)]
            
            # Plot normal points
            normal_mask = symbol_predictions == 1
            if np.any(normal_mask):
                ax1.plot(symbol_data[normal_mask]['Date'], symbol_data[normal_mask]['Close'], 
                        'b-', alpha=0.7, label=f'{symbol} (Normal)')
            
            # Plot anomaly points
            anomaly_mask = symbol_predictions == -1
            if np.any(anomaly_mask):
                ax1.scatter(symbol_data[anomaly_mask]['Date'], symbol_data[anomaly_mask]['Close'], 
                           color='red', s=50, alpha=0.8, label=f'{symbol} (Anomaly)')
        
        ax1.set_title(f'{model_name} - Price with Anomalies')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Anomaly scores (Autoencoder)
    ax2 = axes[0, 1]
    model_name = 'Autoencoder'
    if model_name in results:
        result = results[model_name]
        scores = result['scores']
        
        ax2.plot(features['Date'], scores, 'g-', alpha=0.7, label='Anomaly Score')
        ax2.axhline(y=np.percentile(scores, 95), color='red', linestyle='--', alpha=0.8, label='95th Percentile')
        ax2.set_title(f'{model_name} - Anomaly Scores')
        ax2.set_ylabel('Anomaly Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Model comparison
    ax3 = axes[1, 0]
    model_names = list(results.keys())
    anomaly_counts = [results[name]['metadata']['n_anomalies'] for name in model_names]
    
    bars = ax3.bar(model_names, anomaly_counts, color=['blue', 'green', 'orange'])
    ax3.set_title('Anomalies Detected by Model')
    ax3.set_ylabel('Number of Anomalies')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, anomaly_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    # Plot 4: Anomaly rate comparison
    ax4 = axes[1, 1]
    anomaly_rates = [results[name]['metadata']['anomaly_rate'] for name in model_names]
    
    bars = ax4.bar(model_names, anomaly_rates, color=['blue', 'green', 'orange'])
    ax4.set_title('Anomaly Rate by Model')
    ax4.set_ylabel('Anomaly Rate')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, rate in zip(bars, anomaly_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{rate:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Visualizations saved to 'anomaly_detection_results.png'")


def main():
    """Main function to run the complete analysis."""
    logger.info("Starting Financial Anomaly Detection Analysis")
    
    try:
        # Step 1: Collect data
        logger.info("=" * 50)
        logger.info("STEP 1: DATA COLLECTION")
        logger.info("=" * 50)
        
        # Collect stock data
        stock_data = collect_stock_data()
        
        # Collect crypto data
        crypto_data = collect_crypto_data()
        
        # Combine data
        data = pd.concat([stock_data, crypto_data], ignore_index=True)
        logger.info(f"Total data collected: {len(data)} records")
        
        # Step 2: Feature engineering
        logger.info("=" * 50)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("=" * 50)
        
        features = engineer_features(data)
        
        # Step 3: Model training
        logger.info("=" * 50)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("=" * 50)
        
        models = train_models(features)
        
        # Step 4: Anomaly detection
        logger.info("=" * 50)
        logger.info("STEP 4: ANOMALY DETECTION")
        logger.info("=" * 50)
        
        results = detect_anomalies(models, features)
        
        # Step 5: Model evaluation
        logger.info("=" * 50)
        logger.info("STEP 5: MODEL EVALUATION")
        logger.info("=" * 50)
        
        evaluator, comparison = evaluate_models(results, features)
        
        # Step 6: Visualization
        logger.info("=" * 50)
        logger.info("STEP 6: VISUALIZATION")
        logger.info("=" * 50)
        
        visualize_results(results, features)
        
        # Step 7: Generate report
        logger.info("=" * 50)
        logger.info("STEP 7: REPORT GENERATION")
        logger.info("=" * 50)
        
        report = evaluator.generate_report('anomaly_detection_report.txt')
        logger.info("Report generated: 'anomaly_detection_report.txt'")
        
        # Print summary
        logger.info("=" * 50)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 50)
        
        print("\n" + "=" * 60)
        print("FINANCIAL ANOMALY DETECTION ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Data collected: {len(data)} records")
        print(f"Features generated: {len(features.columns)}")
        print(f"Models trained: {len(models)}")
        print("\nModel Performance:")
        print(comparison[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
        print("\nAnomaly Detection Results:")
        for model_name, result in results.items():
            print(f"  {model_name}: {result['metadata']['n_anomalies']} anomalies ({result['metadata']['anomaly_rate']:.2%})")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
