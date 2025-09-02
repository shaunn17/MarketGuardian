"""
Streamlit Dashboard for Financial Anomaly Detection

This is the main dashboard application for the Financial Anomaly Detection system.
It provides an interactive interface for data collection, model training, and anomaly detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from data.collectors.yahoo_finance_collector import YahooFinanceCollector
from data.collectors.crypto_collector import BinanceCollector, CoinGeckoCollector
from data.collectors.fx_collector import FXCollector
from data.processors.feature_engineer import FinancialFeatureEngineer
from models.isolation_forest import IsolationForestAnomalyDetector
from models.autoencoder import AutoencoderAnomalyDetector
from models.gnn_anomaly import GNNAnomalyDetector
from utils.model_evaluator import AnomalyDetectionEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Financial Anomaly Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-highlight {
        background-color: #ffebee;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Financial Anomaly Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📈 Data Collection", "🔧 Feature Engineering", "🤖 Model Training", "🔍 Anomaly Detection", "📊 Results & Analysis"]
    )
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📈 Data Collection":
        show_data_collection_page()
    elif page == "🔧 Feature Engineering":
        show_feature_engineering_page()
    elif page == "🤖 Model Training":
        show_model_training_page()
    elif page == "🔍 Anomaly Detection":
        show_anomaly_detection_page()
    elif page == "📊 Results & Analysis":
        show_results_page()

def show_home_page():
    """Display the home page with system overview."""
    
    st.markdown("## Welcome to the Financial Anomaly Detection System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Purpose
        This system detects unusual trading patterns and potential fraud in financial markets using advanced machine learning techniques.
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 Features
        - Multi-source data collection (Stocks, Crypto, FX)
        - Advanced ML models (Isolation Forest, Autoencoder, GNN)
        - Interactive visualization and analysis
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Supported Markets
        - **Equities**: Yahoo Finance API
        - **Cryptocurrency**: Binance, CoinGecko APIs
        - **Forex**: Multiple free data sources
        """)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("## 🚀 Quick Start Guide")
    
    steps = [
        "1. **Data Collection**: Collect historical data from your preferred data source",
        "2. **Feature Engineering**: Generate technical indicators and features",
        "3. **Model Training**: Train anomaly detection models",
        "4. **Anomaly Detection**: Detect anomalies in your data",
        "5. **Analysis**: Review results and visualizations"
    ]
    
    for step in steps:
        st.markdown(step)
    
    # System status
    st.markdown("---")
    st.markdown("## 📈 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        data_status = "✅ Ready" if st.session_state.data is not None else "❌ No Data"
        st.metric("Data Status", data_status)
    
    with col2:
        features_status = "✅ Ready" if st.session_state.features is not None else "❌ No Features"
        st.metric("Features Status", features_status)
    
    with col3:
        models_count = len(st.session_state.models)
        st.metric("Trained Models", models_count)
    
    with col4:
        results_count = len(st.session_state.results)
        st.metric("Analysis Results", results_count)

def show_data_collection_page():
    """Display the data collection page."""
    
    st.markdown("## 📈 Data Collection")
    
    # Data source selection
    data_source = st.selectbox(
        "Select Data Source:",
        ["Yahoo Finance (Stocks)", "Binance (Cryptocurrency)", "CoinGecko (Cryptocurrency)", "FX Data"]
    )
    
    if data_source == "Yahoo Finance (Stocks)":
        show_yahoo_finance_collection()
    elif data_source == "Binance (Cryptocurrency)":
        show_binance_collection()
    elif data_source == "CoinGecko (Cryptocurrency)":
        show_coingecko_collection()
    elif data_source == "FX Data":
        show_fx_collection()

def show_yahoo_finance_collection():
    """Show Yahoo Finance data collection interface."""
    
    st.markdown("### Yahoo Finance Data Collection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbols_input = st.text_area(
            "Enter stock symbols (one per line):",
            value="AAPL\nMSFT\nGOOGL\nAMZN\nTSLA",
            height=100
        )
        
        period = st.selectbox(
            "Data Period:",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
        )
    
    with col2:
        interval = st.selectbox(
            "Data Interval:",
            ["1d", "1wk", "1mo"]
        )
        
        start_date = st.date_input(
            "Start Date (optional):",
            value=None
        )
        
        end_date = st.date_input(
            "End Date (optional):",
            value=None
        )
    
    if st.button("Collect Data", type="primary"):
        with st.spinner("Collecting data..."):
            try:
                collector = YahooFinanceCollector()
                symbols = [s.strip() for s in symbols_input.split('\n') if s.strip()]
                
                if start_date and end_date:
                    data = collector.get_multiple_stocks(
                        symbols=symbols,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        interval=interval
                    )
                else:
                    data = collector.get_multiple_stocks(
                        symbols=symbols,
                        period=period,
                        interval=interval
                    )
                
                if not data.empty:
                    st.session_state.data = data
                    st.success(f"Successfully collected data for {len(symbols)} symbols!")
                    
                    # Display data summary
                    st.markdown("### Data Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Records", len(data))
                    with col2:
                        st.metric("Symbols", data['Symbol'].nunique())
                    with col3:
                        st.metric("Date Range", f"{data['Date'].min().date()} to {data['Date'].max().date()}")
                    
                    # Display sample data
                    st.markdown("### Sample Data")
                    st.dataframe(data.head(10))
                    
                    # Plot price data
                    st.markdown("### Price Visualization")
                    fig = px.line(data, x='Date', y='Close', color='Symbol', title='Stock Prices Over Time')
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("No data collected. Please check your symbols and try again.")
                    
            except Exception as e:
                st.error(f"Error collecting data: {str(e)}")

def show_binance_collection():
    """Show Binance data collection interface."""
    
    st.markdown("### Binance Cryptocurrency Data Collection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbols_input = st.text_area(
            "Enter trading pairs (one per line):",
            value="BTC/USDT\nETH/USDT\nBNB/USDT\nADA/USDT\nSOL/USDT",
            height=100
        )
        
        timeframe = st.selectbox(
            "Timeframe:",
            ["1h", "4h", "1d", "1w"]
        )
    
    with col2:
        limit = st.slider(
            "Number of candles:",
            min_value=100,
            max_value=1000,
            value=500,
            step=100
        )
    
    if st.button("Collect Data", type="primary"):
        with st.spinner("Collecting data..."):
            try:
                collector = BinanceCollector()
                symbols = [s.strip() for s in symbols_input.split('\n') if s.strip()]
                
                data = collector.get_multiple_cryptos(
                    symbols=symbols,
                    timeframe=timeframe,
                    limit=limit
                )
                
                if not data.empty:
                    st.session_state.data = data
                    st.success(f"Successfully collected data for {len(symbols)} trading pairs!")
                    
                    # Display data summary
                    st.markdown("### Data Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Records", len(data))
                    with col2:
                        st.metric("Trading Pairs", data['Symbol'].nunique())
                    with col3:
                        st.metric("Date Range", f"{data['Date'].min().date()} to {data['Date'].max().date()}")
                    
                    # Display sample data
                    st.markdown("### Sample Data")
                    st.dataframe(data.head(10))
                    
                    # Plot price data
                    st.markdown("### Price Visualization")
                    fig = px.line(data, x='Date', y='Close', color='Symbol', title='Cryptocurrency Prices Over Time')
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("No data collected. Please check your trading pairs and try again.")
                    
            except Exception as e:
                st.error(f"Error collecting data: {str(e)}")

def show_coingecko_collection():
    """Show CoinGecko data collection interface."""
    
    st.markdown("### CoinGecko Cryptocurrency Data Collection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        coin_ids_input = st.text_area(
            "Enter coin IDs (one per line):",
            value="bitcoin\nethereum\nbinancecoin\ncardano\nsolana",
            height=100
        )
        
        days = st.selectbox(
            "Days of data:",
            [7, 14, 30, 90, 180, 365, "max"]
        )
    
    with col2:
        vs_currency = st.selectbox(
            "Target currency:",
            ["usd", "eur", "btc", "eth"]
        )
    
    if st.button("Collect Data", type="primary"):
        with st.spinner("Collecting data..."):
            try:
                collector = CoinGeckoCollector()
                coin_ids = [s.strip() for s in coin_ids_input.split('\n') if s.strip()]
                
                all_data = []
                for coin_id in coin_ids:
                    data = collector.get_crypto_data(coin_id, days=days, vs_currency=vs_currency)
                    if not data.empty:
                        all_data.append(data)
                
                if all_data:
                    data = pd.concat(all_data, ignore_index=True)
                    st.session_state.data = data
                    st.success(f"Successfully collected data for {len(coin_ids)} cryptocurrencies!")
                    
                    # Display data summary
                    st.markdown("### Data Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Records", len(data))
                    with col2:
                        st.metric("Cryptocurrencies", data['Symbol'].nunique())
                    with col3:
                        st.metric("Date Range", f"{data['Date'].min().date()} to {data['Date'].max().date()}")
                    
                    # Display sample data
                    st.markdown("### Sample Data")
                    st.dataframe(data.head(10))
                    
                    # Plot price data
                    st.markdown("### Price Visualization")
                    fig = px.line(data, x='Date', y='Close', color='Symbol', title='Cryptocurrency Prices Over Time')
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("No data collected. Please check your coin IDs and try again.")
                    
            except Exception as e:
                st.error(f"Error collecting data: {str(e)}")

def show_fx_collection():
    """Show FX data collection interface."""
    
    st.markdown("### Forex Data Collection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pairs_input = st.text_area(
            "Enter currency pairs (one per line):",
            value="USD/EUR\nUSD/GBP\nUSD/JPY\nUSD/CAD\nUSD/AUD",
            height=100
        )
        
        days = st.slider(
            "Days of data:",
            min_value=7,
            max_value=365,
            value=30,
            step=7
        )
    
    with col2:
        data_source = st.selectbox(
            "Data Source:",
            ["Alpha Vantage (Demo)", "ExchangeRate-API"]
        )
    
    if st.button("Collect Data", type="primary"):
        with st.spinner("Collecting data..."):
            try:
                collector = FXCollector()
                pairs = [s.strip() for s in pairs_input.split('\n') if s.strip()]
                
                if data_source == "Alpha Vantage (Demo)":
                    data = collector.get_multiple_fx_pairs(pairs, days=days)
                else:
                    base_currencies = list(set([pair.split('/')[0] for pair in pairs]))
                    target_currencies = list(set([pair.split('/')[1] for pair in pairs]))
                    data = collector.get_fx_data_from_exchangerate_api(
                        base_currency=base_currencies[0],
                        target_currencies=target_currencies,
                        days=days
                    )
                
                if not data.empty:
                    st.session_state.data = data
                    st.success(f"Successfully collected data for {len(pairs)} currency pairs!")
                    
                    # Display data summary
                    st.markdown("### Data Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Records", len(data))
                    with col2:
                        st.metric("Currency Pairs", data['Symbol'].nunique())
                    with col3:
                        st.metric("Date Range", f"{data['Date'].min().date()} to {data['Date'].max().date()}")
                    
                    # Display sample data
                    st.markdown("### Sample Data")
                    st.dataframe(data.head(10))
                    
                    # Plot price data
                    st.markdown("### Price Visualization")
                    fig = px.line(data, x='Date', y='Close', color='Symbol', title='Exchange Rates Over Time')
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("No data collected. Please check your currency pairs and try again.")
                    
            except Exception as e:
                st.error(f"Error collecting data: {str(e)}")

def show_feature_engineering_page():
    """Display the feature engineering page."""
    
    st.markdown("## 🔧 Feature Engineering")
    
    if st.session_state.data is None:
        st.warning("Please collect data first before engineering features.")
        return
    
    st.markdown("### Feature Engineering Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_time_features = st.checkbox("Include Time Features", value=True)
        include_correlation_features = st.checkbox("Include Correlation Features", value=False)
    
    with col2:
        feature_types = st.multiselect(
            "Feature Types to Include:",
            ["Price Features", "Volume Features", "Technical Indicators", "Returns Features", "Anomaly Features"],
            default=["Price Features", "Volume Features", "Technical Indicators", "Returns Features", "Anomaly Features"]
        )
    
    if st.button("Generate Features", type="primary"):
        with st.spinner("Generating features..."):
            try:
                engineer = FinancialFeatureEngineer()
                
                # Get symbols for correlation features
                symbols = st.session_state.data['Symbol'].unique().tolist() if include_correlation_features else None
                
                # Generate features
                features = engineer.engineer_all_features(
                    st.session_state.data,
                    include_time_features=include_time_features,
                    include_correlation_features=include_correlation_features,
                    symbols=symbols
                )
                
                st.session_state.features = features
                st.success("Features generated successfully!")
                
                # Display feature summary
                st.markdown("### Feature Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Features", len(features.columns))
                with col2:
                    st.metric("Original Features", len(st.session_state.data.columns))
                with col3:
                    st.metric("New Features", len(features.columns) - len(st.session_state.data.columns))
                
                # Display feature list
                st.markdown("### Generated Features")
                feature_list = [col for col in features.columns if col not in st.session_state.data.columns]
                st.write(feature_list)
                
                # Display sample data
                st.markdown("### Sample Data with Features")
                st.dataframe(features.head(10))
                
            except Exception as e:
                st.error(f"Error generating features: {str(e)}")

def show_model_training_page():
    """Display the model training page."""
    
    st.markdown("## 🤖 Model Training")
    
    if st.session_state.features is None:
        st.warning("Please generate features first before training models.")
        return
    
    st.markdown("### Model Selection")
    
    models_to_train = st.multiselect(
        "Select models to train:",
        ["Isolation Forest", "Autoencoder", "Graph Neural Network"],
        default=["Isolation Forest", "Autoencoder"]
    )
    
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models..."):
            try:
                # Prepare data for training
                features_df, _, feature_names = FinancialFeatureEngineer().prepare_for_ml(st.session_state.features)
                
                # Train selected models
                for model_name in models_to_train:
                    if model_name == "Isolation Forest":
                        model = IsolationForestAnomalyDetector(contamination=0.1)
                        model.fit(features_df)
                        st.session_state.models[model_name] = model
                        
                    elif model_name == "Autoencoder":
                        model = AutoencoderAnomalyDetector(
                            encoding_dim=32,
                            hidden_dims=[64, 32],
                            epochs=50
                        )
                        model.fit(features_df, verbose=False)
                        st.session_state.models[model_name] = model
                        
                    elif model_name == "Graph Neural Network":
                        model = GNNAnomalyDetector(
                            model_type='GCN',
                            hidden_dim=32,
                            output_dim=16,
                            epochs=50
                        )
                        model.fit(st.session_state.features, verbose=False)
                        st.session_state.models[model_name] = model
                
                st.success(f"Successfully trained {len(models_to_train)} models!")
                
                # Display model information
                st.markdown("### Trained Models")
                for model_name, model in st.session_state.models.items():
                    with st.expander(f"{model_name} - Model Info"):
                        model_info = model.get_model_info()
                        st.json(model_info)
                
            except Exception as e:
                st.error(f"Error training models: {str(e)}")

def show_anomaly_detection_page():
    """Display the anomaly detection page."""
    
    st.markdown("## 🔍 Anomaly Detection")
    
    if not st.session_state.models:
        st.warning("Please train models first before detecting anomalies.")
        return
    
    st.markdown("### Detection Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model = st.selectbox(
            "Select Model:",
            list(st.session_state.models.keys())
        )
        
        threshold = st.slider(
            "Anomaly Threshold (percentile):",
            min_value=80,
            max_value=99,
            value=95,
            step=1
        )
    
    with col2:
        show_individual_anomalies = st.checkbox("Show Individual Anomalies", value=True)
        show_reconstruction_errors = st.checkbox("Show Reconstruction Errors", value=True)
    
    if st.button("Detect Anomalies", type="primary"):
        with st.spinner("Detecting anomalies..."):
            try:
                model = st.session_state.models[selected_model]
                
                # Prepare data
                if selected_model == "Graph Neural Network":
                    # GNN needs the full features dataframe
                    predictions, scores, metadata = model.detect_anomalies(
                        st.session_state.features,
                        threshold=threshold/100
                    )
                else:
                    # Other models need the prepared features
                    features_df, _, _ = FinancialFeatureEngineer().prepare_for_ml(st.session_state.features)
                    predictions, scores, metadata = model.detect_anomalies(
                        features_df,
                        threshold=threshold/100
                    )
                
                # Store results
                st.session_state.results[selected_model] = {
                    'predictions': predictions,
                    'scores': scores,
                    'metadata': metadata
                }
                
                st.success("Anomaly detection completed!")
                
                # Display results summary
                st.markdown("### Detection Results")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", len(predictions))
                with col2:
                    st.metric("Anomalies Detected", metadata['n_anomalies'])
                with col3:
                    st.metric("Anomaly Rate", f"{metadata['anomaly_rate']:.2%}")
                with col4:
                    st.metric("Threshold Used", f"{threshold}th percentile")
                
                # Plot results
                st.markdown("### Anomaly Visualization")
                
                # Create time series plot with anomalies
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Price with Anomalies', 'Anomaly Scores'),
                    vertical_spacing=0.1
                )
                
                # Plot price data with anomalies
                for symbol in st.session_state.features['Symbol'].unique():
                    symbol_data = st.session_state.features[st.session_state.features['Symbol'] == symbol]
                    symbol_predictions = predictions[:len(symbol_data)]
                    
                    # Normal points
                    normal_mask = symbol_predictions == 1
                    if np.any(normal_mask):
                        fig.add_trace(
                            go.Scatter(
                                x=symbol_data[normal_mask]['Date'],
                                y=symbol_data[normal_mask]['Close'],
                                mode='lines',
                                name=f'{symbol} (Normal)',
                                line=dict(color='blue', width=1)
                            ),
                            row=1, col=1
                        )
                    
                    # Anomaly points
                    anomaly_mask = symbol_predictions == -1
                    if np.any(anomaly_mask):
                        fig.add_trace(
                            go.Scatter(
                                x=symbol_data[anomaly_mask]['Date'],
                                y=symbol_data[anomaly_mask]['Close'],
                                mode='markers',
                                name=f'{symbol} (Anomaly)',
                                marker=dict(color='red', size=8)
                            ),
                            row=1, col=1
                        )
                
                # Plot anomaly scores
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.features['Date'],
                        y=scores,
                        mode='lines',
                        name='Anomaly Score',
                        line=dict(color='green', width=1)
                    ),
                    row=2, col=1
                )
                
                # Add threshold line
                threshold_value = np.percentile(scores, threshold)
                fig.add_hline(
                    y=threshold_value,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold: {threshold_value:.3f}",
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=800,
                    title_text=f"Anomaly Detection Results - {selected_model}",
                    showlegend=True
                )
                
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Price", row=1, col=1)
                fig.update_yaxes(title_text="Anomaly Score", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display anomaly details
                if show_individual_anomalies:
                    st.markdown("### Anomaly Details")
                    anomaly_indices = np.where(predictions == -1)[0]
                    if len(anomaly_indices) > 0:
                        anomaly_data = st.session_state.features.iloc[anomaly_indices]
                        anomaly_data['Anomaly_Score'] = scores[anomaly_indices]
                        st.dataframe(anomaly_data[['Date', 'Symbol', 'Close', 'Volume', 'Anomaly_Score']])
                    else:
                        st.info("No anomalies detected.")
                
            except Exception as e:
                st.error(f"Error detecting anomalies: {str(e)}")

def show_results_page():
    """Display the results and analysis page."""
    
    st.markdown("## 📊 Results & Analysis")
    
    if not st.session_state.results:
        st.warning("No analysis results available. Please run anomaly detection first.")
        return
    
    st.markdown("### Model Comparison")
    
    # Create comparison table
    comparison_data = []
    for model_name, result in st.session_state.results.items():
        comparison_data.append({
            'Model': model_name,
            'Anomalies Detected': result['metadata']['n_anomalies'],
            'Anomaly Rate': f"{result['metadata']['anomaly_rate']:.2%}",
            'Threshold': result['metadata'].get('threshold_used', 'N/A')
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Plot comparison
    st.markdown("### Anomaly Rate Comparison")
    fig = px.bar(
        comparison_df,
        x='Model',
        y='Anomalies Detected',
        title='Anomalies Detected by Model',
        color='Anomalies Detected',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model-specific results
    st.markdown("### Detailed Results")
    
    for model_name, result in st.session_state.results.items():
        with st.expander(f"{model_name} - Detailed Results"):
            st.markdown(f"**Anomalies Detected:** {result['metadata']['n_anomalies']}")
            st.markdown(f"**Anomaly Rate:** {result['metadata']['anomaly_rate']:.2%}")
            st.markdown(f"**Threshold Used:** {result['metadata'].get('threshold_used', 'N/A')}")
            
            # Score distribution
            st.markdown("**Anomaly Score Distribution:**")
            fig = px.histogram(
                x=result['scores'],
                title=f'{model_name} - Anomaly Score Distribution',
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
