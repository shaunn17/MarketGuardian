"""
Test Pipeline for Financial Anomaly Detection

This script tests the complete pipeline to ensure all components work correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from data.collectors.yahoo_finance_collector import YahooFinanceCollector
from data.processors.feature_engineer import FinancialFeatureEngineer
from models.isolation_forest import IsolationForestAnomalyDetector
from models.autoencoder import AutoencoderAnomalyDetector
from models.gnn_anomaly import GNNAnomalyDetector
from utils.model_evaluator import AnomalyDetectionEvaluator


class TestFinancialAnomalyDetection(unittest.TestCase):
    """Test cases for the Financial Anomaly Detection system."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample financial data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Date': dates,
            'Symbol': 'AAPL',
            'Open': 100 + np.random.randn(100).cumsum(),
            'High': 105 + np.random.randn(100).cumsum(),
            'Low': 95 + np.random.randn(100).cumsum(),
            'Close': 100 + np.random.randn(100).cumsum(),
            'Volume': np.random.randint(1000000, 10000000, 100)
        })
        
        # Create multi-symbol data for GNN testing
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        multi_data = []
        for symbol in symbols:
            symbol_data = self.sample_data.copy()
            symbol_data['Symbol'] = symbol
            multi_data.append(symbol_data)
        self.multi_symbol_data = pd.concat(multi_data, ignore_index=True)
    
    def test_feature_engineering(self):
        """Test feature engineering functionality."""
        print("Testing feature engineering...")
        
        engineer = FinancialFeatureEngineer()
        features = engineer.engineer_all_features(self.sample_data)
        
        # Check that features were created
        self.assertGreater(len(features.columns), len(self.sample_data.columns))
        
        # Check that specific features exist
        expected_features = ['Price_Range', 'Volume_MA_5', 'MA_20', 'RSI', 'Return_1d']
        for feature in expected_features:
            self.assertIn(feature, features.columns)
        
        print("Feature engineering test passed")
    
    def test_isolation_forest(self):
        """Test Isolation Forest model."""
        print("Testing Isolation Forest...")
        
        # Engineer features
        engineer = FinancialFeatureEngineer()
        features = engineer.engineer_all_features(self.sample_data)
        features_df, _, _ = engineer.prepare_for_ml(features)
        
        # Train model
        model = IsolationForestAnomalyDetector(contamination=0.1)
        model.fit(features_df)
        
        # Test predictions
        predictions, scores, metadata = model.detect_anomalies(features_df)
        
        # Check outputs
        self.assertEqual(len(predictions), len(features_df))
        self.assertEqual(len(scores), len(features_df))
        self.assertIn('n_anomalies', metadata)
        self.assertIn('anomaly_rate', metadata)
        
        print("Isolation Forest test passed")
    
    def test_autoencoder(self):
        """Test Autoencoder model."""
        print("Testing Autoencoder...")
        
        # Engineer features
        engineer = FinancialFeatureEngineer()
        features = engineer.engineer_all_features(self.sample_data)
        features_df, _, _ = engineer.prepare_for_ml(features)
        
        # Train model
        model = AutoencoderAnomalyDetector(
            encoding_dim=16,
            hidden_dims=[32, 16],
            epochs=10,  # Reduced for testing
            batch_size=16
        )
        model.fit(features_df, verbose=False)
        
        # Test predictions
        predictions, scores, metadata = model.detect_anomalies(features_df)
        
        # Check outputs
        self.assertEqual(len(predictions), len(features_df))
        self.assertEqual(len(scores), len(features_df))
        self.assertIn('n_anomalies', metadata)
        self.assertIn('anomaly_rate', metadata)
        
        print("Autoencoder test passed")
    
    def test_gnn(self):
        """Test Graph Neural Network model."""
        print("Testing Graph Neural Network...")
        
        # Engineer features
        engineer = FinancialFeatureEngineer()
        features = engineer.engineer_all_features(self.multi_symbol_data)
        
        # Train model
        model = GNNAnomalyDetector(
            model_type='GCN',
            hidden_dim=16,
            output_dim=8,
            epochs=10  # Reduced for testing
        )
        model.fit(features, verbose=False)
        
        # Test predictions
        predictions, scores, metadata = model.detect_anomalies(features)
        
        # Check outputs - GNN returns predictions per symbol, not per row
        unique_symbols = features['Symbol'].unique()
        self.assertEqual(len(predictions), len(unique_symbols))  # Should match number of symbols
        self.assertEqual(len(scores), len(unique_symbols))  # Should match number of symbols
        self.assertIn('n_anomalies', metadata)
        self.assertIn('anomaly_rate', metadata)
        
        print("Graph Neural Network test passed")
    
    def test_model_evaluator(self):
        """Test model evaluation functionality."""
        print("Testing model evaluator...")
        
        # Create sample data
        y_true = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
        y_pred = np.random.choice([-1, 1], size=100, p=[0.1, 0.9])
        scores = np.random.rand(100)
        
        # Test evaluator
        evaluator = AnomalyDetectionEvaluator()
        result = evaluator.evaluate_model(
            model_name='Test Model',
            y_true=y_true,
            y_pred=y_pred,
            anomaly_scores=scores
        )
        
        # Check results
        self.assertIn('classification_metrics', result)
        self.assertIn('basic_metrics', result)
        self.assertIn('ranking_metrics', result)
        
        print("Model evaluator test passed")
    
    @patch('yfinance.Ticker')
    def test_yahoo_finance_collector(self, mock_ticker):
        """Test Yahoo Finance collector with mocked data."""
        print("Testing Yahoo Finance collector...")
        
        # Mock the yfinance response with proper index
        dates = pd.date_range('2023-01-01', periods=3, freq='D')
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000000, 1100000, 1200000],
            'Dividends': [0, 0, 0],
            'Stock Splits': [0, 0, 0]
        }, index=dates)
        mock_ticker.return_value.history.return_value = mock_data
        
        # Test collector
        collector = YahooFinanceCollector()
        data = collector.get_stock_data('AAPL', period='1mo')
        
        # Check that data was processed correctly
        self.assertIn('Symbol', data.columns)
        self.assertIn('Date', data.columns)
        
        print("Yahoo Finance collector test passed")
    
    def test_end_to_end_pipeline(self):
        """Test the complete end-to-end pipeline."""
        print("Testing end-to-end pipeline...")
        
        # Step 1: Feature engineering
        engineer = FinancialFeatureEngineer()
        features = engineer.engineer_all_features(self.sample_data)
        features_df, _, _ = engineer.prepare_for_ml(features)
        
        # Step 2: Train multiple models
        models = {}
        
        # Isolation Forest
        isolation_forest = IsolationForestAnomalyDetector(contamination=0.1)
        isolation_forest.fit(features_df)
        models['Isolation Forest'] = isolation_forest
        
        # Autoencoder
        autoencoder = AutoencoderAnomalyDetector(
            encoding_dim=16,
            hidden_dims=[32, 16],
            epochs=10,
            batch_size=16
        )
        autoencoder.fit(features_df, verbose=False)
        models['Autoencoder'] = autoencoder
        
        # Step 3: Detect anomalies
        results = {}
        for model_name, model in models.items():
            predictions, scores, metadata = model.detect_anomalies(features_df)
            results[model_name] = {
                'predictions': predictions,
                'scores': scores,
                'metadata': metadata
            }
        
        # Step 4: Evaluate models
        evaluator = AnomalyDetectionEvaluator()
        
        # Create synthetic true labels for evaluation
        y_true = np.random.choice([0, 1], size=len(features_df), p=[0.9, 0.1])
        
        for model_name, result in results.items():
            evaluator.evaluate_model(
                model_name=model_name,
                y_true=y_true,
                y_pred=result['predictions'],
                anomaly_scores=result['scores']
            )
        
        # Step 5: Compare models
        comparison = evaluator.compare_models()
        
        # Check that comparison was successful
        self.assertGreater(len(comparison), 0)
        self.assertIn('Model', comparison.columns)
        self.assertIn('Accuracy', comparison.columns)
        
        print("End-to-end pipeline test passed")


def run_tests():
    """Run all tests."""
    print("Running Financial Anomaly Detection Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestFinancialAnomalyDetection)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
