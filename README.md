# Financial Anomaly Detection System

A comprehensive machine learning system for detecting unusual trading patterns and potential fraud in financial markets (equities, crypto, FX). This project provides a complete end-to-end solution for financial anomaly detection using state-of-the-art machine learning techniques.

## 🚀 Features

- **Multi-source Data Collection**: Yahoo Finance (equities), Binance (crypto), CoinGecko (crypto), and FX data
- **Advanced ML Models**: Isolation Forest, Autoencoder, and Graph Neural Networks
- **Interactive Dashboard**: Streamlit-based web interface with real-time visualization
- **Comprehensive Feature Engineering**: 50+ technical indicators and financial features
- **Model Evaluation**: Multiple metrics for unsupervised anomaly detection
- **Real-time Detection**: Support for streaming data analysis
- **Extensible Architecture**: Easy to add new data sources and models

## 📊 Supported Data Sources

- **Equities**: Yahoo Finance API (free)
- **Cryptocurrency**: Binance API, CoinGecko API (free tiers available)
- **Forex**: Alpha Vantage, ExchangeRate-API, Fixer.io (free tiers available)

## 🤖 Machine Learning Models

1. **Isolation Forest**: Fast anomaly detection using tree-based isolation
2. **Autoencoder**: Reconstruction-based anomaly detection using neural networks
3. **Graph Neural Network**: Correlation-aware anomaly detection for multiple assets

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd anomaly-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python tests/test_pipeline.py
   ```

## 🚀 Quick Start

### Option 1: Interactive Dashboard
```bash
streamlit run dashboard/app.py
```

### Option 2: Simple Example
```bash
python examples/simple_example.py
```

### Option 3: Complete Analysis
```bash
python examples/run_analysis.py
```

## 📁 Project Structure

```
├── data/                   # Data collection and processing
│   ├── collectors/         # API collectors for different data sources
│   │   ├── yahoo_finance_collector.py
│   │   ├── crypto_collector.py
│   │   └── fx_collector.py
│   └── processors/         # Data preprocessing and feature engineering
│       └── feature_engineer.py
├── models/                 # Machine learning models
│   ├── isolation_forest.py
│   ├── autoencoder.py
│   └── gnn_anomaly.py
├── utils/                  # Utility functions
│   └── model_evaluator.py
├── dashboard/              # Streamlit dashboard
│   └── app.py
├── examples/               # Example scripts
│   ├── simple_example.py
│   └── run_analysis.py
├── tests/                  # Unit tests
│   └── test_pipeline.py
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 📈 Usage Examples

### Basic Usage
```python
from data.collectors.yahoo_finance_collector import YahooFinanceCollector
from data.processors.feature_engineer import FinancialFeatureEngineer
from models.isolation_forest import IsolationForestAnomalyDetector

# Collect data
collector = YahooFinanceCollector()
data = collector.get_stock_data("AAPL", period="1y")

# Engineer features
engineer = FinancialFeatureEngineer()
features = engineer.engineer_all_features(data)
features_df, _, _ = engineer.prepare_for_ml(features)

# Train model
model = IsolationForestAnomalyDetector(contamination=0.1)
model.fit(features_df)

# Detect anomalies
predictions, scores, metadata = model.detect_anomalies(features_df)
print(f"Detected {metadata['n_anomalies']} anomalies")
```

### Advanced Usage with Multiple Models
```python
from models.autoencoder import AutoencoderAnomalyDetector
from models.gnn_anomaly import GNNAnomalyDetector
from utils.model_evaluator import AnomalyDetectionEvaluator

# Train multiple models
models = {
    'Isolation Forest': IsolationForestAnomalyDetector(),
    'Autoencoder': AutoencoderAnomalyDetector(),
    'GNN': GNNAnomalyDetector()
}

# Train and evaluate
evaluator = AnomalyDetectionEvaluator()
for name, model in models.items():
    model.fit(features_df)
    predictions, scores, metadata = model.detect_anomalies(features_df)
    evaluator.evaluate_model(name, y_true, predictions, scores)

# Compare models
comparison = evaluator.compare_models()
print(comparison)
```

## 🔧 Configuration

### Model Parameters
- **Isolation Forest**: `contamination`, `n_estimators`, `max_samples`
- **Autoencoder**: `encoding_dim`, `hidden_dims`, `epochs`, `learning_rate`
- **GNN**: `model_type`, `hidden_dim`, `num_layers`, `heads`

### Feature Engineering
- **Price Features**: Range, body size, shadows, gaps
- **Volume Features**: Moving averages, ratios, z-scores
- **Technical Indicators**: MA, EMA, MACD, RSI, Bollinger Bands
- **Returns Features**: Simple returns, log returns, volatility
- **Time Features**: Cyclical encoding, market session indicators

## 📊 Dashboard Features

The Streamlit dashboard provides:
- **Data Collection**: Interactive interface for all data sources
- **Feature Engineering**: Configurable feature generation
- **Model Training**: Train multiple models with different parameters
- **Anomaly Detection**: Real-time anomaly detection with visualization
- **Results Analysis**: Comprehensive model comparison and evaluation

## 🧪 Testing

Run the test suite to verify everything works correctly:
```bash
python tests/test_pipeline.py
```

The tests cover:
- Data collection (with mocked APIs)
- Feature engineering
- Model training and prediction
- Model evaluation
- End-to-end pipeline

## 📚 API Documentation

### Data Collectors
- `YahooFinanceCollector`: Collect stock data from Yahoo Finance
- `BinanceCollector`: Collect cryptocurrency data from Binance
- `CoinGeckoCollector`: Collect cryptocurrency data from CoinGecko
- `FXCollector`: Collect forex data from multiple sources

### Models
- `IsolationForestAnomalyDetector`: Tree-based anomaly detection
- `AutoencoderAnomalyDetector`: Neural network-based reconstruction
- `GNNAnomalyDetector`: Graph neural network for correlated assets

### Utilities
- `FinancialFeatureEngineer`: Comprehensive feature engineering
- `AnomalyDetectionEvaluator`: Model evaluation and comparison

## 🔍 Anomaly Detection Metrics

The system provides multiple evaluation metrics:
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Ranking Metrics**: ROC-AUC, PR-AUC
- **Anomaly-Specific**: Anomaly rate, threshold analysis
- **Visualization**: Time series plots, confusion matrices, score distributions

## 🚀 Advanced Features

### Ensemble Methods
Combine multiple models for improved detection:
```python
# Train ensemble of models
ensemble_results = {}
for model_name, model in models.items():
    predictions, scores, metadata = model.detect_anomalies(features_df)
    ensemble_results[model_name] = {'predictions': predictions, 'scores': scores}

# Combine results (example: majority voting)
combined_predictions = np.mean([r['predictions'] for r in ensemble_results.values()], axis=0)
```

### Real-time Detection
For streaming data analysis:
```python
# Process new data points
new_data = collector.get_latest_data("AAPL")
new_features = engineer.engineer_all_features(new_data)
new_features_df, _, _ = engineer.prepare_for_ml(new_features)

# Detect anomalies in real-time
predictions, scores, metadata = model.detect_anomalies(new_features_df)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

MIT License - Free for personal and commercial use.

---

**Note**: This system is for educational and research purposes. Always verify results and consider market conditions when making financial decisions.
