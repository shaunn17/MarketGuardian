"""
Real-time Monitoring Dashboard for Financial Anomaly Detection

This module provides real-time monitoring capabilities with live data updates,
alerts, and interactive monitoring tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
import queue
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

# Add project root to path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collectors.yahoo_finance_collector import YahooFinanceCollector
from data.collectors.crypto_collector import BinanceCollector
from models.isolation_forest import IsolationForestAnomalyDetector
from models.autoencoder import AutoencoderAnomalyDetector
from data.processors.feature_engineer import FinancialFeatureEngineer

logger = logging.getLogger(__name__)

class RealTimeMonitor:
    """Real-time monitoring system for financial anomaly detection."""
    
    def __init__(self):
        self.data_queue = queue.Queue()
        self.alert_queue = queue.Queue()
        self.is_monitoring = False
        self.monitoring_thread = None
        self.last_update = None
        self.metrics_history = []
        self.alert_history = []
        
    def start_monitoring(self, symbols: List[str], interval: int = 30):
        """Start real-time monitoring."""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(symbols, interval),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started monitoring {len(symbols)} symbols with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped monitoring")
    
    def _monitoring_loop(self, symbols: List[str], interval: int):
        """Main monitoring loop."""
        collector = YahooFinanceCollector()
        
        while self.is_monitoring:
            try:
                # Collect latest data
                current_data = []
                for symbol in symbols:
                    try:
                        data = collector.get_stock_data(symbol, period="1d", interval="1m")
                        if not data.empty:
                            latest = data.iloc[-1].copy()
                            latest['timestamp'] = datetime.now()
                            current_data.append(latest)
                    except Exception as e:
                        logger.error(f"Error collecting data for {symbol}: {e}")
                
                if current_data:
                    # Process data
                    df = pd.DataFrame(current_data)
                    metrics = self._calculate_metrics(df)
                    
                    # Check for alerts
                    alerts = self._check_alerts(metrics)
                    
                    # Update queues
                    self.data_queue.put({
                        'timestamp': datetime.now(),
                        'data': df,
                        'metrics': metrics
                    })
                    
                    if alerts:
                        self.alert_queue.put({
                            'timestamp': datetime.now(),
                            'alerts': alerts
                        })
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _calculate_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate real-time metrics."""
        metrics = {}
        
        if not data.empty:
            metrics['avg_price'] = data['Close'].mean()
            metrics['total_volume'] = data['Volume'].sum()
            metrics['price_volatility'] = data['Close'].std()
            metrics['price_change'] = data['Close'].pct_change().mean() * 100
            
            # Calculate moving averages
            if len(data) > 5:
                metrics['ma_5'] = data['Close'].rolling(5).mean().iloc[-1]
            if len(data) > 20:
                metrics['ma_20'] = data['Close'].rolling(20).mean().iloc[-1]
        
        return metrics
    
    def _check_alerts(self, metrics: Dict[str, float]) -> List[str]:
        """Check for alert conditions."""
        alerts = []
        
        # Price change alerts
        if 'price_change' in metrics:
            if abs(metrics['price_change']) > 5:  # 5% change threshold
                alerts.append(f"High price volatility: {metrics['price_change']:.2f}%")
        
        # Volume alerts
        if 'total_volume' in metrics:
            if metrics['total_volume'] > 10000000:  # High volume threshold
                alerts.append(f"High trading volume: {metrics['total_volume']:,.0f}")
        
        # Volatility alerts
        if 'price_volatility' in metrics:
            if metrics['price_volatility'] > 2.0:  # High volatility threshold
                alerts.append(f"High price volatility: {metrics['price_volatility']:.2f}")
        
        return alerts
    
    def get_latest_data(self) -> Optional[Dict]:
        """Get latest data from queue."""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_latest_alerts(self) -> Optional[Dict]:
        """Get latest alerts from queue."""
        try:
            return self.alert_queue.get_nowait()
        except queue.Empty:
            return None

def create_realtime_dashboard():
    """Create the real-time monitoring dashboard."""
    
    st.markdown("""
    <style>
        .realtime-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .metric-card-realtime {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
        }
        
        .alert-card {
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #ef4444;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online {
            background-color: #10b981;
            animation: pulse 2s infinite;
        }
        
        .status-offline {
            background-color: #ef4444;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="realtime-header">
        <h1>🚀 Real-time Financial Monitoring</h1>
        <p>Live anomaly detection and market monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for monitoring
    if 'monitor' not in st.session_state:
        st.session_state.monitor = RealTimeMonitor()
    
    if 'monitoring_data' not in st.session_state:
        st.session_state.monitoring_data = []
    
    if 'monitoring_alerts' not in st.session_state:
        st.session_state.monitoring_alerts = []
    
    # Control panel
    st.markdown("### 🎛️ Control Panel")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        symbols_input = st.text_area(
            "Symbols to Monitor",
            value="AAPL\nMSFT\nGOOGL\nAMZN\nTSLA",
            height=100,
            help="Enter one symbol per line"
        )
    
    with col2:
        refresh_interval = st.selectbox(
            "Refresh Interval",
            options=[10, 30, 60, 120],
            index=1,
            format_func=lambda x: f"{x} seconds"
        )
    
    with col3:
        st.markdown("#### Status")
        is_monitoring = st.session_state.monitor.is_monitoring
        status_color = "status-online" if is_monitoring else "status-offline"
        status_text = "🟢 Online" if is_monitoring else "🔴 Offline"
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 1rem 0;">
            <span class="status-indicator {status_color}"></span>
            <span>{status_text}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("#### Actions")
        
        if st.button("▶️ Start Monitoring", disabled=is_monitoring):
            symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
            if symbols:
                st.session_state.monitor.start_monitoring(symbols, refresh_interval)
                st.success(f"Started monitoring {len(symbols)} symbols")
                st.rerun()
            else:
                st.error("Please enter at least one symbol")
        
        if st.button("⏹️ Stop Monitoring", disabled=not is_monitoring):
            st.session_state.monitor.stop_monitoring()
            st.success("Stopped monitoring")
            st.rerun()
    
    # Real-time data processing
    if is_monitoring:
        # Get latest data
        latest_data = st.session_state.monitor.get_latest_data()
        if latest_data:
            st.session_state.monitoring_data.append(latest_data)
            # Keep only last 100 data points
            if len(st.session_state.monitoring_data) > 100:
                st.session_state.monitoring_data = st.session_state.monitoring_data[-100:]
        
        # Get latest alerts
        latest_alerts = st.session_state.monitor.get_latest_alerts()
        if latest_alerts:
            st.session_state.monitoring_alerts.append(latest_alerts)
            # Keep only last 50 alerts
            if len(st.session_state.monitoring_alerts) > 50:
                st.session_state.monitoring_alerts = st.session_state.monitoring_alerts[-50:]
    
    # Auto-refresh
    if is_monitoring:
        time.sleep(1)
        st.rerun()
    
    # Display real-time metrics
    if st.session_state.monitoring_data:
        st.markdown("### 📊 Real-time Metrics")
        
        latest_metrics = st.session_state.monitoring_data[-1]['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card-realtime">
                <h4>Average Price</h4>
                <h2>${latest_metrics.get('avg_price', 0):.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card-realtime">
                <h4>Total Volume</h4>
                <h2>{latest_metrics.get('total_volume', 0):,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card-realtime">
                <h4>Price Change</h4>
                <h2>{latest_metrics.get('price_change', 0):.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card-realtime">
                <h4>Volatility</h4>
                <h2>{latest_metrics.get('price_volatility', 0):.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Real-time charts
        st.markdown("### 📈 Real-time Charts")
        
        # Prepare data for charts
        timestamps = [d['timestamp'] for d in st.session_state.monitoring_data]
        avg_prices = [d['metrics'].get('avg_price', 0) for d in st.session_state.monitoring_data]
        volumes = [d['metrics'].get('total_volume', 0) for d in st.session_state.monitoring_data]
        price_changes = [d['metrics'].get('price_change', 0) for d in st.session_state.monitoring_data]
        
        # Price chart
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=timestamps,
            y=avg_prices,
            mode='lines+markers',
            name='Average Price',
            line=dict(color='#667eea', width=2),
            marker=dict(size=4)
        ))
        
        fig_price.update_layout(
            title='Real-time Average Price',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Volume and price change chart
        fig_combined = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Total Volume', 'Price Change %'),
            vertical_spacing=0.1
        )
        
        fig_combined.add_trace(
            go.Scatter(x=timestamps, y=volumes, mode='lines', name='Volume', line=dict(color='#10b981')),
            row=1, col=1
        )
        
        fig_combined.add_trace(
            go.Scatter(x=timestamps, y=price_changes, mode='lines', name='Price Change', line=dict(color='#f59e0b')),
            row=2, col=1
        )
        
        fig_combined.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        st.plotly_chart(fig_combined, use_container_width=True)
    
    # Alerts section
    if st.session_state.monitoring_alerts:
        st.markdown("### 🚨 Recent Alerts")
        
        # Show last 10 alerts
        recent_alerts = st.session_state.monitoring_alerts[-10:]
        
        for alert_data in reversed(recent_alerts):
            timestamp = alert_data['timestamp'].strftime('%H:%M:%S')
            for alert in alert_data['alerts']:
                st.markdown(f"""
                <div class="alert-card">
                    <strong>{timestamp}</strong> - {alert}
                </div>
                """, unsafe_allow_html=True)
    
    # Historical data table
    if st.session_state.monitoring_data:
        st.markdown("### 📋 Historical Data")
        
        # Create DataFrame from monitoring data
        history_data = []
        for data_point in st.session_state.monitoring_data:
            row = {
                'Timestamp': data_point['timestamp'],
                'Symbols': len(data_point['data']),
                'Avg Price': data_point['metrics'].get('avg_price', 0),
                'Total Volume': data_point['metrics'].get('total_volume', 0),
                'Price Change %': data_point['metrics'].get('price_change', 0),
                'Volatility': data_point['metrics'].get('price_volatility', 0)
            }
            history_data.append(row)
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df.tail(20), use_container_width=True)
    
    # Export functionality
    if st.session_state.monitoring_data:
        st.markdown("### 📤 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Monitoring Data"):
                # Create comprehensive export
                export_data = []
                for data_point in st.session_state.monitoring_data:
                    for _, row in data_point['data'].iterrows():
                        export_row = row.to_dict()
                        export_row['timestamp'] = data_point['timestamp']
                        export_row.update(data_point['metrics'])
                        export_data.append(export_row)
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"realtime_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📋 Export Alerts"):
                alert_data = []
                for alert_data_point in st.session_state.monitoring_alerts:
                    for alert in alert_data_point['alerts']:
                        alert_data.append({
                            'timestamp': alert_data_point['timestamp'],
                            'alert': alert
                        })
                
                if alert_data:
                    alert_df = pd.DataFrame(alert_data)
                    csv = alert_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Alerts CSV",
                        data=csv,
                        file_name=f"monitoring_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

def create_anomaly_monitoring():
    """Create real-time anomaly monitoring."""
    
    st.markdown("### 🔍 Real-time Anomaly Detection")
    
    if st.session_state.monitoring_data:
        # Use the latest data for anomaly detection
        latest_data = st.session_state.monitoring_data[-1]['data']
        
        if not latest_data.empty:
            # Prepare features
            engineer = FinancialFeatureEngineer()
            features = engineer.engineer_all_features(latest_data)
            features_df, _, _ = engineer.prepare_for_ml(features)
            
            # Quick anomaly detection with Isolation Forest
            if st.button("🔍 Detect Anomalies"):
                with st.spinner("Detecting anomalies..."):
                    detector = IsolationForestAnomalyDetector(contamination=0.1)
                    detector.fit(features_df)
                    predictions, scores, metadata = detector.detect_anomalies(features_df)
                    
                    # Display results
                    st.markdown("#### Anomaly Detection Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Records", len(predictions))
                    
                    with col2:
                        st.metric("Anomalies Detected", metadata['n_anomalies'])
                    
                    with col3:
                        st.metric("Anomaly Rate", f"{metadata['anomaly_rate']:.2%}")
                    
                    # Show anomalies
                    if metadata['n_anomalies'] > 0:
                        anomaly_indices = np.where(predictions == -1)[0]
                        anomaly_data = latest_data.iloc[anomaly_indices]
                        
                        st.markdown("#### Detected Anomalies")
                        st.dataframe(anomaly_data, use_container_width=True)
                        
                        # Anomaly visualization
                        fig = px.scatter(
                            latest_data,
                            x='Close',
                            y='Volume',
                            color=predictions,
                            color_discrete_map={1: 'blue', -1: 'red'},
                            title='Anomaly Detection Results',
                            labels={'color': 'Anomaly Status'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("No anomalies detected in the current data!")

if __name__ == "__main__":
    create_realtime_dashboard()
    create_anomaly_monitoring()
