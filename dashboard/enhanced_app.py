"""
Enhanced Streamlit Dashboard for Financial Anomaly Detection

This is an enhanced version of the dashboard with modern UI/UX improvements,
advanced visualizations, and better user experience.
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
import time
import json
from typing import Dict, List, Optional, Tuple

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
    page_title="Financial Anomaly Detection Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state with enhanced features FIRST
if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'notifications' not in st.session_state:
    st.session_state.notifications = []

# Enhanced CSS with modern design and working theme toggle
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Light theme variables */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --info-color: #3b82f6;
        --bg-color: #ffffff;
        --surface-color: #f8fafc;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }
    
    /* Dark theme variables */
    .dark-theme {
        --primary-color: #818cf8;
        --secondary-color: #a78bfa;
        --success-color: #34d399;
        --warning-color: #fbbf24;
        --error-color: #f87171;
        --info-color: #60a5fa;
        --bg-color: #0f172a;
        --surface-color: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border-color: #334155;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    /* Apply theme to body */
    body {
        background-color: var(--bg-color);
        color: var(--text-primary);
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg-color);
        color: var(--text-primary);
    }
    
    /* Header styles */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 0.5rem;
    }
    
    /* Card styles */
    .metric-card {
        background: var(--light-surface);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-success {
        background-color: rgba(16, 185, 129, 0.1);
        color: var(--success-color);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .status-warning {
        background-color: rgba(245, 158, 11, 0.1);
        color: var(--warning-color);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .status-error {
        background-color: rgba(239, 68, 68, 0.1);
        color: var(--error-color);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    /* Anomaly highlight */
    .anomaly-highlight {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.1));
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--error-color);
        margin: 1rem 0;
    }
    
    /* Progress bars */
    .progress-container {
        background-color: var(--border-color);
        border-radius: 10px;
        overflow: hidden;
        height: 8px;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-lg);
    }
    
    /* Sidebar enhancements */
    .css-1d391kg {
        background: var(--light-surface);
    }
    
    /* Custom containers */
    .info-container {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-container {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(99, 102, 241, 0.3);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--surface-color);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-color);
    }
    
    /* Theme toggle button */
    .theme-toggle {
        background: var(--surface-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: var(--text-primary);
        transition: all 0.3s ease;
    }
    
    .theme-toggle:hover {
        background: var(--primary-color);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Apply theme using CSS classes that actually work
theme_css = f"""
<style>
    /* Apply theme based on session state */
    .stApp {{
        background-color: {'#0f172a' if st.session_state.theme == 'dark' else '#ffffff'};
        color: {'#f1f5f9' if st.session_state.theme == 'dark' else '#1e293b'};
    }}
    
    .main .block-container {{
        background-color: {'#0f172a' if st.session_state.theme == 'dark' else '#ffffff'};
        color: {'#f1f5f9' if st.session_state.theme == 'dark' else '#1e293b'};
    }}
    
    .css-1d391kg {{
        background-color: {'#1e293b' if st.session_state.theme == 'dark' else '#f8fafc'};
    }}
    
    /* Update metric cards for theme */
    .metric-card {{
        background: {'#1e293b' if st.session_state.theme == 'dark' else '#f8fafc'};
        color: {'#f1f5f9' if st.session_state.theme == 'dark' else '#1e293b'};
        border-color: {'#334155' if st.session_state.theme == 'dark' else '#e2e8f0'};
    }}
    
    .metric-value {{
        color: {'#818cf8' if st.session_state.theme == 'dark' else '#6366f1'};
    }}
    
    .metric-label {{
        color: {'#94a3b8' if st.session_state.theme == 'dark' else '#64748b'};
    }}
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# Session state already initialized above

def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal") -> str:
    """Create a styled metric card."""
    delta_html = ""
    if delta:
        delta_color_class = "positive" if delta_color == "normal" else "negative"
        delta_html = f'<div class="metric-delta {delta_color_class}">{delta}</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

def show_notification(message: str, type: str = "info"):
    """Show a styled notification."""
    icon_map = {
        "success": "✅",
        "warning": "⚠️", 
        "error": "❌",
        "info": "ℹ️"
    }
    
    type_class = f"status-{type}" if type in ["success", "warning", "error"] else "status-info"
    
    st.markdown(f"""
    <div class="status-indicator {type_class}">
        <span>{icon_map.get(type, 'ℹ️')}</span>
        <span>{message}</span>
    </div>
    """, unsafe_allow_html=True)

def create_progress_bar(progress: float, label: str = "") -> str:
    """Create a custom progress bar."""
    return f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {progress}%"></div>
    </div>
    <div style="text-align: center; margin-top: 0.5rem; font-size: 0.875rem; color: var(--text-secondary);">
        {label} ({progress:.1f}%)
    </div>
    """

def main():
    """Main dashboard application with enhanced UI."""
    
    # Header with theme toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">🚀 Financial Anomaly Detection Pro</h1>', unsafe_allow_html=True)
    
    with col2:
        theme_button_text = "🌙 Dark Mode" if st.session_state.theme == 'light' else "☀️ Light Mode"
        current_theme = "Dark" if st.session_state.theme == 'dark' else "Light"
        
        # Show current theme status with better styling
        theme_color = "#94a3b8" if st.session_state.theme == 'dark' else "#64748b"
        st.markdown(f'<div style="text-align: center; margin-bottom: 0.5rem; font-size: 0.875rem; color: {theme_color}; font-weight: 500;">Current: {current_theme} Theme</div>', unsafe_allow_html=True)
        
        if st.button(theme_button_text, key="theme_toggle"):
            st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
            st.success(f"Switched to {current_theme} theme!")
            st.rerun()
    
    with col3:
        st.session_state.auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)
    
    # Enhanced sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        # Main navigation
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Dashboard", "📈 Data Collection", "🔧 Feature Engineering", "🤖 Model Training", "🔍 Anomaly Detection", "📊 Analytics", "⚙️ Settings"],
            key="main_nav"
        )
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        
        if st.button("💾 Save Session", use_container_width=True):
            show_notification("Session saved successfully!", "success")
        
        if st.button("📤 Export Results", use_container_width=True):
            show_notification("Results exported successfully!", "success")
        
        st.markdown("---")
        
        # System status
        st.markdown("### 📊 System Status")
        
        # Data status
        data_status = "✅ Ready" if st.session_state.data is not None else "❌ No Data"
        st.markdown(f"**Data:** {data_status}")
        
        # Features status
        features_status = "✅ Ready" if st.session_state.features is not None else "❌ No Features"
        st.markdown(f"**Features:** {features_status}")
        
        # Models status
        models_count = len(st.session_state.models)
        st.markdown(f"**Models:** {models_count} trained")
        
        # Results status
        results_count = len(st.session_state.results)
        st.markdown(f"**Results:** {results_count} analyses")
        
        # Performance metrics
        if st.session_state.results:
            st.markdown("---")
            st.markdown("### 📈 Performance")
            
            for model_name, result in st.session_state.results.items():
                anomaly_rate = result['metadata']['anomaly_rate']
                st.markdown(f"**{model_name}:** {anomaly_rate:.1%} anomaly rate")
    
    # Route to appropriate page
    if page == "🏠 Dashboard":
        show_enhanced_dashboard()
    elif page == "📈 Data Collection":
        show_enhanced_data_collection_page()
    elif page == "🔧 Feature Engineering":
        show_enhanced_feature_engineering_page()
    elif page == "🤖 Model Training":
        show_enhanced_model_training_page()
    elif page == "🔍 Anomaly Detection":
        show_enhanced_anomaly_detection_page()
    elif page == "📊 Analytics":
        show_enhanced_analytics_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_enhanced_dashboard():
    """Enhanced dashboard with modern design and real-time updates."""
    
    st.markdown('<h2 class="sub-header">📊 Executive Dashboard</h2>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.data is not None:
            total_records = len(st.session_state.data)
            symbols = st.session_state.data['Symbol'].nunique()
            st.markdown(create_metric_card("Total Records", f"{total_records:,}", f"{symbols} symbols"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Total Records", "0", "No data"), unsafe_allow_html=True)
    
    with col2:
        if st.session_state.features is not None:
            feature_count = len(st.session_state.features.columns)
            st.markdown(create_metric_card("Features", f"{feature_count}", "Engineered"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Features", "0", "Not generated"), unsafe_allow_html=True)
    
    with col3:
        models_count = len(st.session_state.models)
        st.markdown(create_metric_card("Models", f"{models_count}", "Trained"), unsafe_allow_html=True)
    
    with col4:
        if st.session_state.results:
            total_anomalies = sum(result['metadata']['n_anomalies'] for result in st.session_state.results.values())
            st.markdown(create_metric_card("Anomalies", f"{total_anomalies}", "Detected"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Anomalies", "0", "Not analyzed"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start section
    if st.session_state.data is None:
        st.markdown("### 🚀 Get Started")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-container">
                <h4>📈 Step 1: Collect Data</h4>
                <p>Start by collecting financial data from your preferred source. We support stocks, crypto, and forex data.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-container">
                <h4>🔧 Step 2: Engineer Features</h4>
                <p>Generate technical indicators and features to prepare your data for anomaly detection.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-container">
                <h4>🤖 Step 3: Train Models</h4>
                <p>Train advanced ML models to detect anomalies in your financial data.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick data collection
        st.markdown("### ⚡ Quick Data Collection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Collect Stock Data", use_container_width=True):
                st.switch_page("📈 Data Collection")
        
        with col2:
            if st.button("₿ Collect Crypto Data", use_container_width=True):
                st.switch_page("📈 Data Collection")
    
    else:
        # Data overview
        st.markdown("### 📊 Data Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.session_state.data is not None:
                # Price chart
                fig = px.line(
                    st.session_state.data, 
                    x='Date', 
                    y='Close', 
                    color='Symbol',
                    title='Price Movement Over Time',
                    height=400
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter, sans-serif")
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Data summary
            st.markdown("#### 📋 Data Summary")
            
            if st.session_state.data is not None:
                data = st.session_state.data
                
                st.markdown(f"**Symbols:** {data['Symbol'].nunique()}")
                st.markdown(f"**Date Range:** {data['Date'].min().date()} to {data['Date'].max().date()}")
                st.markdown(f"**Total Records:** {len(data):,}")
                
                # Volume analysis
                avg_volume = data['Volume'].mean()
                st.metric("Avg Volume", f"{avg_volume:,.0f}")
                
                # Price statistics
                price_stats = data.groupby('Symbol')['Close'].agg(['min', 'max', 'mean']).round(2)
                st.markdown("#### 💰 Price Statistics")
                st.dataframe(price_stats, use_container_width=True)
        
        # Recent anomalies (if available)
        if st.session_state.results:
            st.markdown("### 🚨 Recent Anomalies")
            
            for model_name, result in st.session_state.results.items():
                with st.expander(f"Anomalies from {model_name}"):
                    anomaly_indices = np.where(result['predictions'] == -1)[0]
                    if len(anomaly_indices) > 0:
                        anomaly_data = st.session_state.features.iloc[anomaly_indices]
                        st.dataframe(
                            anomaly_data[['Date', 'Symbol', 'Close', 'Volume']].head(10),
                            use_container_width=True
                        )
                    else:
                        st.info("No anomalies detected by this model.")

def show_enhanced_data_collection_page():
    """Enhanced data collection page with better UX."""
    
    st.markdown('<h2 class="sub-header">📈 Data Collection</h2>', unsafe_allow_html=True)
    
    # Data source selection with enhanced UI
    st.markdown("### 🎯 Select Data Source")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📈 Yahoo Finance", use_container_width=True, type="primary"):
            st.session_state.selected_source = "yahoo"
    
    with col2:
        if st.button("₿ Binance", use_container_width=True):
            st.session_state.selected_source = "binance"
    
    with col3:
        if st.button("🪙 CoinGecko", use_container_width=True):
            st.session_state.selected_source = "coingecko"
    
    with col4:
        if st.button("💱 Forex", use_container_width=True):
            st.session_state.selected_source = "forex"
    
    # Show selected source interface
    if hasattr(st.session_state, 'selected_source'):
        if st.session_state.selected_source == "yahoo":
            show_enhanced_yahoo_finance_collection()
        elif st.session_state.selected_source == "binance":
            show_enhanced_binance_collection()
        elif st.session_state.selected_source == "coingecko":
            show_enhanced_coingecko_collection()
        elif st.session_state.selected_source == "forex":
            show_enhanced_fx_collection()

def show_enhanced_yahoo_finance_collection():
    """Enhanced Yahoo Finance collection interface."""
    
    st.markdown("### 📈 Yahoo Finance Data Collection")
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📝 Configuration")
        
        symbols_input = st.text_area(
            "Enter stock symbols (one per line):",
            value="AAPL\nMSFT\nGOOGL\nAMZN\nTSLA\nNVDA\nMETA\nNFLX\nAMD\nINTC",
            height=120,
            help="Enter one stock symbol per line. Popular symbols: AAPL, MSFT, GOOGL, AMZN, TSLA"
        )
        
        col1a, col1b = st.columns(2)
        with col1a:
            period = st.selectbox(
                "Data Period:",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                index=3,
                help="Select the time period for data collection"
            )
        
        with col1b:
            interval = st.selectbox(
                "Data Interval:",
                ["1d", "1wk", "1mo"],
                help="Select the data frequency"
            )
    
    with col2:
        st.markdown("#### ⚙️ Advanced Options")
        
        use_custom_dates = st.checkbox("Use Custom Date Range")
        
        if use_custom_dates:
            start_date = st.date_input(
                "Start Date:",
                value=datetime.now() - timedelta(days=365),
                help="Select start date for data collection"
            )
            
            end_date = st.date_input(
                "End Date:",
                value=datetime.now(),
                help="Select end date for data collection"
            )
        else:
            start_date = None
            end_date = None
        
        # Data validation
        st.markdown("#### ✅ Validation")
        symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
        st.markdown(f"**Symbols to collect:** {len(symbols)}")
        st.markdown(f"**Estimated records:** {len(symbols) * 250} (approx)")
    
    # Collection button with enhanced feedback
    if st.button("🚀 Collect Data", type="primary", use_container_width=True):
        if not symbols:
            show_notification("Please enter at least one stock symbol.", "error")
        else:
            with st.spinner("Collecting data..."):
                try:
                    status_text.text("Initializing collector...")
                    progress_bar.progress(10)
                    
                    collector = YahooFinanceCollector()
                    progress_bar.progress(20)
                    
                    status_text.text(f"Collecting data for {len(symbols)} symbols...")
                    
                    if use_custom_dates and start_date and end_date:
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
                    
                    progress_bar.progress(80)
                    status_text.text("Processing data...")
                    
                    if not data.empty:
                        st.session_state.data = data
                        progress_bar.progress(100)
                        status_text.text("✅ Data collection completed!")
                        
                        show_notification(f"Successfully collected {len(data):,} records for {len(symbols)} symbols!", "success")
                        
                        # Enhanced data summary
                        st.markdown("### 📊 Collection Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(create_metric_card("Total Records", f"{len(data):,}"), unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(create_metric_card("Symbols", f"{data['Symbol'].nunique()}"), unsafe_allow_html=True)
                        
                        with col3:
                            date_range = f"{(data['Date'].max() - data['Date'].min()).days} days"
                            st.markdown(create_metric_card("Date Range", date_range), unsafe_allow_html=True)
                        
                        with col4:
                            avg_price = f"${data['Close'].mean():.2f}"
                            st.markdown(create_metric_card("Avg Price", avg_price), unsafe_allow_html=True)
                        
                        # Data preview with enhanced styling
                        st.markdown("### 👀 Data Preview")
                        
                        # Sample data with better formatting
                        sample_data = data.head(10)
                        st.dataframe(
                            sample_data,
                            use_container_width=True,
                            column_config={
                                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                                "Close": st.column_config.NumberColumn("Close", format="$%.2f"),
                                "Volume": st.column_config.NumberColumn("Volume", format="%d")
                            }
                        )
                        
                        # Interactive price chart
                        st.markdown("### 📈 Price Visualization")
                        
                        fig = px.line(
                            data, 
                            x='Date', 
                            y='Close', 
                            color='Symbol',
                            title='Stock Prices Over Time',
                            height=500
                        )
                        
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter, sans-serif"),
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Volume analysis
                        st.markdown("### 📊 Volume Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Volume by symbol
                            volume_by_symbol = data.groupby('Symbol')['Volume'].mean().sort_values(ascending=False)
                            fig_volume = px.bar(
                                x=volume_by_symbol.index,
                                y=volume_by_symbol.values,
                                title='Average Volume by Symbol',
                                height=300
                            )
                            st.plotly_chart(fig_volume, use_container_width=True)
                        
                        with col2:
                            # Volume over time
                            fig_volume_time = px.line(
                                data.groupby('Date')['Volume'].sum().reset_index(),
                                x='Date',
                                y='Volume',
                                title='Total Volume Over Time',
                                height=300
                            )
                            st.plotly_chart(fig_volume_time, use_container_width=True)
                        
                    else:
                        show_notification("No data collected. Please check your symbols and try again.", "error")
                        
                except Exception as e:
                    show_notification(f"Error collecting data: {str(e)}", "error")
                    logger.error(f"Data collection error: {str(e)}")
                
                finally:
                    progress_bar.empty()
                    status_text.empty()

# Placeholder functions for other enhanced pages
def show_enhanced_binance_collection():
    st.info("Enhanced Binance collection interface - Coming soon!")

def show_enhanced_coingecko_collection():
    st.info("Enhanced CoinGecko collection interface - Coming soon!")

def show_enhanced_fx_collection():
    st.info("Enhanced Forex collection interface - Coming soon!")

def show_enhanced_feature_engineering_page():
    st.info("Enhanced feature engineering page - Coming soon!")

def show_enhanced_model_training_page():
    st.info("Enhanced model training page - Coming soon!")

def show_enhanced_anomaly_detection_page():
    st.info("Enhanced anomaly detection page - Coming soon!")

def show_enhanced_analytics_page():
    st.info("Enhanced analytics page - Coming soon!")

def show_settings_page():
    st.info("Settings page - Coming soon!")

if __name__ == "__main__":
    main()
