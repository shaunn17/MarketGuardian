# Financial Anomaly Detection Pro

A comprehensive AI-powered machine learning system for detecting unusual trading patterns and potential fraud in financial markets (equities, crypto, FX). This project provides a complete end-to-end solution with an enhanced web dashboard, AI-powered insights, and state-of-the-art anomaly detection techniques.

## 🚀 Key Features

### 📊 **Enhanced Dashboard**
- **Modern UI**: Beautiful, responsive interface with light/dark theme support
- **Interactive Pages**: Data Collection, Feature Engineering, Model Training, Anomaly Detection, Analytics
- **Real-time Visualization**: Interactive charts and graphs with Plotly
- **Comprehensive Settings**: Customizable system configuration

### 🤖 **AI-Powered Analysis**
- **AI Anomaly Insights**: Intelligent explanations of detected anomalies
- **Market Analysis**: AI-powered market condition assessment
- **Trading Recommendations**: AI-generated buy/sell/hold signals
- **Risk Assessment**: AI evaluation of portfolio risks
- **Demo Mode**: Test AI features without API keys

### 🔧 **Advanced ML Pipeline**
- **Multi-source Data Collection**: Yahoo Finance, Binance, CoinGecko, FX data
- **Advanced ML Models**: Isolation Forest, Autoencoder, Graph Neural Networks
- **Comprehensive Feature Engineering**: 50+ technical indicators and financial features
- **Model Evaluation**: Multiple metrics for unsupervised anomaly detection
- **Real-time Detection**: Support for streaming data analysis

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

### Option 1: Enhanced Dashboard (Recommended)
```bash
streamlit run dashboard/enhanced_app.py
```

### Option 2: Original Dashboard
```bash
streamlit run dashboard/app.py
```

### Option 3: Simple Example
```bash
python examples/simple_example.py
```

### Option 4: Complete Analysis
```bash
python examples/run_analysis.py
```

## 🎯 **Getting Started with the Enhanced Dashboard**

1. **Launch the dashboard**:
   ```bash
   streamlit run dashboard/enhanced_app.py
   ```

2. **Navigate through the pages**:
   - **📈 Data Collection**: Gather financial data from multiple sources
   - **🔧 Feature Engineering**: Create technical indicators and features
   - **🤖 Model Training**: Train Isolation Forest and Autoencoder models
   - **🔍 Anomaly Detection**: Detect and analyze anomalies
   - **📊 Analytics**: Comprehensive data analysis and visualization
   - **🧠 AI Analysis**: AI-powered insights and recommendations
   - **⚙️ Settings**: Configure the system

3. **Test AI features** (No API keys needed):
   - Go to "🧠 AI Analysis"
   - Enable "Demo Mode" to see AI features in action
   - Explore anomaly insights, market analysis, and trading recommendations

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
│   ├── model_evaluator.py
│   └── ai_anomaly_analyzer.py  # AI-powered analysis
├── dashboard/              # Streamlit dashboards
│   ├── app.py             # Original dashboard
│   ├── enhanced_app.py    # Enhanced dashboard with AI features
│   ├── components.py      # Reusable UI components
│   ├── realtime_dashboard.py  # Real-time monitoring
│   └── ai_components.py   # AI-specific components
├── examples/               # Example scripts
│   ├── simple_example.py
│   ├── run_analysis.py
│   ├── autoencoder_explanation.py
│   └── model_comparison_explanation.py
├── tests/                  # Unit tests
│   └── test_pipeline.py
├── requirements.txt        # Python dependencies
├── AI_FEATURES_README.md   # AI features documentation
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

## 📊 Enhanced Dashboard Features

### 🎨 **Modern UI/UX**
- **Light/Dark Theme**: Toggle between themes with persistent settings
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Interactive Charts**: Plotly-powered visualizations with zoom, pan, and hover
- **Real-time Updates**: Live data refresh and progress tracking
- **Custom Styling**: Modern CSS with professional appearance

### 📈 **Data Collection Page**
- **Multi-source Support**: Yahoo Finance, Binance, CoinGecko, FX APIs
- **Interactive Configuration**: Symbol selection, time periods, intervals
- **Real-time Status**: Live collection progress and error handling
- **Data Preview**: Immediate data validation and preview
- **Export Options**: Save collected data in multiple formats

### 🔧 **Feature Engineering Page**
- **Interactive Controls**: Select specific feature types to generate
- **Advanced Options**: Customizable parameters for technical indicators
- **Feature Preview**: Real-time preview of generated features
- **Progress Tracking**: Visual progress bars and status updates
- **Data Validation**: Automatic handling of missing values and outliers

### 🤖 **Model Training Page**
- **Model Selection**: Choose between Isolation Forest and Autoencoder
- **Parameter Tuning**: Interactive sliders and input fields
- **Training Progress**: Real-time training metrics and visualizations
- **Model Comparison**: Side-by-side performance comparison
- **Save/Load Models**: Persistent model storage and retrieval

### 🔍 **Anomaly Detection Page**
- **Interactive Detection**: Configure contamination rates and thresholds
- **Real-time Results**: Live anomaly detection with instant feedback
- **Detailed Analysis**: Individual anomaly information and scores
- **Visualization**: Interactive charts showing anomalies over time
- **Export Results**: Save detection results in CSV/JSON formats

### 📊 **Analytics Page**
- **Price Analysis**: Candlestick charts, price distributions, volatility metrics
- **Volume Analysis**: Volume patterns, correlations, and spikes
- **Technical Indicators**: Comprehensive technical analysis with charts
- **Summary Statistics**: Data quality metrics and correlation matrices
- **Interactive Tabs**: Organized analysis by category

### 🧠 **AI Analysis Page**
- **Demo Mode**: Test AI features without API keys
- **AI Provider Selection**: Easy switching between Demo and OpenAI
- **Anomaly Insights**: AI-powered explanations of detected anomalies
- **Market Analysis**: AI assessment of market conditions
- **Trading Recommendations**: AI-generated buy/sell/hold signals
- **Risk Assessment**: AI evaluation of portfolio risks

### ⚙️ **Settings Page**
- **Appearance Settings**: Theme, chart preferences, UI customization
- **Data Collection Settings**: API configurations, collection limits
- **AI Configuration**: OpenAI API keys, model selection, parameters
- **Analysis Settings**: Default parameters, feature selection, model configs
- **Reset Options**: Restore defaults and clear settings

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

## 🤖 AI Features

### 🎭 **Demo Mode (No API Keys Required)**
- **Sample Data**: Realistic financial anomalies for testing
- **AI Insights**: Simulated AI explanations and analysis
- **Trading Recommendations**: Mock buy/sell/hold signals
- **Risk Assessment**: Sample risk analysis and mitigation strategies
- **Perfect for Learning**: Understand AI capabilities without setup

### 🔑 **OpenAI Integration (API Key Required)**
- **Real AI Analysis**: GPT-4 powered anomaly explanations
- **Market Intelligence**: AI assessment of market conditions
- **Trading Signals**: AI-generated trading recommendations
- **Risk Evaluation**: Professional risk assessment
- **Free Tier Available**: $5 in free credits at platform.openai.com

### 🧠 **AI Analysis Capabilities**
- **Anomaly Insights**: Intelligent explanations of why anomalies occurred
- **Market Analysis**: AI-powered market condition assessment
- **Trading Recommendations**: Buy/sell/hold signals with confidence levels
- **Risk Assessment**: Portfolio risk evaluation and mitigation strategies
- **Contextual Analysis**: AI considers market conditions, volatility, and trends

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

### AI-Powered Analysis
```python
from utils.ai_anomaly_analyzer import AIAnomalyAnalyzer

# Initialize AI analyzer
ai_analyzer = AIAnomalyAnalyzer(openai_api_key="your-key-here")

# Get AI insights for anomalies
insights = ai_analyzer.analyze_anomalies(anomalies, market_context)
recommendations = ai_analyzer.generate_trading_recommendations(anomalies)
risk_assessment = ai_analyzer.assess_risk(anomalies, portfolio_data)
```

## 📦 Dependencies

### Core Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **torch**: PyTorch for neural networks
- **plotly**: Interactive visualizations
- **streamlit**: Web dashboard framework

### Data Collection
- **yfinance**: Yahoo Finance API
- **ccxt**: Cryptocurrency exchange APIs
- **requests**: HTTP requests for APIs

### AI Features
- **openai**: OpenAI GPT models
- **anthropic**: Claude models (optional)
- **requests**: API communication

### Installation
```bash
pip install -r requirements.txt
```

## 🎯 **Getting Started Guide**

### 1. **Quick Test (5 minutes)**
```bash
# Launch enhanced dashboard
streamlit run dashboard/enhanced_app.py

# Navigate to AI Analysis → Enable Demo Mode
# Explore all features without any setup
```

### 2. **Full Analysis (15 minutes)**
```bash
# 1. Data Collection: Collect some stock data
# 2. Feature Engineering: Generate technical indicators
# 3. Model Training: Train Isolation Forest and Autoencoder
# 4. Anomaly Detection: Detect anomalies in your data
# 5. Analytics: Explore comprehensive data analysis
# 6. AI Analysis: Get AI-powered insights (Demo Mode)
```

### 3. **Real AI Analysis (Optional)**
```bash
# Get free OpenAI API key at platform.openai.com
# Go to Settings → AI Configuration
# Enter your API key and select model
# Go to AI Analysis → Select OpenAI → Initialize
```

## 🔧 **Configuration Options**

### Dashboard Settings
- **Theme**: Light/Dark mode with persistent settings
- **Charts**: Interactive Plotly visualizations
- **Auto-refresh**: Real-time data updates
- **Export**: Multiple data formats (CSV, JSON, Excel)

### AI Configuration
- **Demo Mode**: No setup required, sample data
- **OpenAI**: GPT-4, GPT-3.5-turbo models
- **Parameters**: Analysis depth, response length, creativity
- **Free Tier**: $5 in free credits available

### Model Parameters
- **Isolation Forest**: Contamination rate, number of estimators
- **Autoencoder**: Encoding dimension, epochs, learning rate
- **Feature Engineering**: Technical indicator periods, normalization

## 🚀 **Performance Tips**

- **Data Collection**: Use appropriate time periods to avoid rate limits
- **Feature Engineering**: Select only needed features for faster processing
- **Model Training**: Start with default parameters, then tune as needed
- **AI Analysis**: Use Demo Mode for testing, OpenAI for production
- **Memory Usage**: Monitor data size for large datasets

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

## 🆘 **Support & Troubleshooting**

### Common Issues
- **Dashboard not loading**: Check if port 8501 is available
- **Data collection fails**: Verify internet connection and API limits
- **Model training errors**: Ensure data is properly formatted
- **AI features not working**: Check API keys and internet connection

### Getting Help
- **Demo Mode**: Use for testing without API keys
- **Settings Page**: Configure system parameters
- **Error Messages**: Check console for detailed error information
- **Documentation**: Refer to this README and inline help text
