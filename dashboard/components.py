"""
Enhanced UI Components for Financial Anomaly Detection Dashboard

This module contains reusable UI components with modern design and enhanced functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import time
from datetime import datetime, timedelta

class EnhancedComponents:
    """Collection of enhanced UI components."""
    
    @staticmethod
    def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal", icon: str = "📊") -> str:
        """Create a modern metric card with icon and delta."""
        delta_html = ""
        if delta:
            delta_color_class = "positive" if delta_color == "normal" else "negative"
            delta_html = f'<div class="metric-delta {delta_color_class}">{delta}</div>'
        
        return f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """
    
    @staticmethod
    def create_progress_ring(progress: float, size: int = 120, color: str = "#6366f1") -> str:
        """Create a circular progress indicator."""
        circumference = 2 * 3.14159 * (size // 2 - 10)
        stroke_dasharray = circumference
        stroke_dashoffset = circumference - (progress / 100) * circumference
        
        return f"""
        <div class="progress-ring-container">
            <svg width="{size}" height="{size}" class="progress-ring">
                <circle
                    cx="{size//2}"
                    cy="{size//2}"
                    r="{size//2 - 10}"
                    fill="none"
                    stroke="#e2e8f0"
                    stroke-width="8"
                />
                <circle
                    cx="{size//2}"
                    cy="{size//2}"
                    r="{size//2 - 10}"
                    fill="none"
                    stroke="{color}"
                    stroke-width="8"
                    stroke-linecap="round"
                    stroke-dasharray="{stroke_dasharray}"
                    stroke-dashoffset="{stroke_dashoffset}"
                    transform="rotate(-90 {size//2} {size//2})"
                />
            </svg>
            <div class="progress-ring-text">{progress:.0f}%</div>
        </div>
        """
    
    @staticmethod
    def create_status_badge(status: str, type: str = "info") -> str:
        """Create a status badge with appropriate styling."""
        type_classes = {
            "success": "status-success",
            "warning": "status-warning", 
            "error": "status-error",
            "info": "status-info"
        }
        
        icons = {
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "info": "ℹ️"
        }
        
        return f"""
        <div class="status-badge {type_classes.get(type, 'status-info')}">
            <span class="status-icon">{icons.get(type, 'ℹ️')}</span>
            <span class="status-text">{status}</span>
        </div>
        """
    
    @staticmethod
    def create_animated_counter(target: int, duration: int = 2000) -> str:
        """Create an animated counter component."""
        return f"""
        <div class="animated-counter" data-target="{target}" data-duration="{duration}">
            <span class="counter-value">0</span>
        </div>
        <script>
            function animateCounter(element) {{
                const target = parseInt(element.dataset.target);
                const duration = parseInt(element.dataset.duration);
                const start = performance.now();
                
                function updateCounter(currentTime) {{
                    const elapsed = currentTime - start;
                    const progress = Math.min(elapsed / duration, 1);
                    const current = Math.floor(progress * target);
                    
                    element.querySelector('.counter-value').textContent = current.toLocaleString();
                    
                    if (progress < 1) {{
                        requestAnimationFrame(updateCounter);
                    }}
                }}
                
                requestAnimationFrame(updateCounter);
            }}
            
            document.querySelectorAll('.animated-counter').forEach(animateCounter);
        </script>
        """

class AdvancedCharts:
    """Advanced chart components with enhanced interactivity."""
    
    @staticmethod
    def create_candlestick_chart(data: pd.DataFrame, title: str = "Price Chart") -> go.Figure:
        """Create an enhanced candlestick chart."""
        fig = go.Figure()
        
        for symbol in data['Symbol'].unique():
            symbol_data = data[data['Symbol'] == symbol].sort_values('Date')
            
            fig.add_trace(go.Candlestick(
                x=symbol_data['Date'],
                open=symbol_data['Open'],
                high=symbol_data['High'],
                low=symbol_data['Low'],
                close=symbol_data['Close'],
                name=symbol,
                increasing_line_color='#10b981',
                decreasing_line_color='#ef4444'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif"),
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_anomaly_heatmap(data: pd.DataFrame, predictions: np.ndarray, title: str = "Anomaly Heatmap") -> go.Figure:
        """Create a heatmap showing anomaly patterns."""
        # Create a pivot table for the heatmap
        data_with_predictions = data.copy()
        data_with_predictions['Anomaly'] = predictions == -1
        
        # Group by symbol and date to create heatmap data
        heatmap_data = data_with_predictions.groupby(['Symbol', 'Date']).agg({
            'Anomaly': 'sum',
            'Close': 'last'
        }).reset_index()
        
        # Create pivot table for heatmap
        pivot_data = heatmap_data.pivot(index='Symbol', columns='Date', values='Anomaly')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Reds',
            showscale=True,
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        
        return fig
    
    @staticmethod
    def create_correlation_matrix(data: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
        """Create an interactive correlation matrix."""
        # Calculate correlation matrix for numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            showscale=True,
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        
        return fig
    
    @staticmethod
    def create_3d_scatter(data: pd.DataFrame, x_col: str, y_col: str, z_col: str, 
                         color_col: str = None, title: str = "3D Scatter Plot") -> go.Figure:
        """Create a 3D scatter plot."""
        fig = go.Figure()
        
        if color_col:
            fig.add_trace(go.Scatter3d(
                x=data[x_col],
                y=data[y_col],
                z=data[z_col],
                mode='markers',
                marker=dict(
                    size=5,
                    color=data[color_col],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=data.index,
                hovertemplate=f'{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{z_col}: %{{z}}<extra></extra>'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=data[x_col],
                y=data[y_col],
                z=data[z_col],
                mode='markers',
                marker=dict(size=5, color='#6366f1'),
                text=data.index,
                hovertemplate=f'{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{z_col}: %{{z}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        
        return fig

class InteractiveWidgets:
    """Interactive widget components."""
    
    @staticmethod
    def create_parameter_tuner(model_type: str) -> Dict[str, Any]:
        """Create an interactive parameter tuning interface."""
        st.markdown("### 🎛️ Model Parameters")
        
        params = {}
        
        if model_type == "Isolation Forest":
            col1, col2 = st.columns(2)
            
            with col1:
                params['contamination'] = st.slider(
                    "Contamination Rate",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.1,
                    step=0.01,
                    help="Expected proportion of anomalies"
                )
                
                params['n_estimators'] = st.slider(
                    "Number of Estimators",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    help="Number of trees in the forest"
                )
            
            with col2:
                params['max_samples'] = st.selectbox(
                    "Max Samples",
                    options=['auto', 0.5, 0.7, 0.8, 0.9, 1.0],
                    index=0,
                    help="Number of samples to draw from X"
                )
                
                params['max_features'] = st.slider(
                    "Max Features",
                    min_value=0.1,
                    max_value=1.0,
                    value=1.0,
                    step=0.1,
                    help="Number of features to draw from X"
                )
        
        elif model_type == "Autoencoder":
            col1, col2 = st.columns(2)
            
            with col1:
                params['encoding_dim'] = st.slider(
                    "Encoding Dimension",
                    min_value=8,
                    max_value=128,
                    value=32,
                    step=8,
                    help="Size of the encoding layer"
                )
                
                params['hidden_dims'] = st.multiselect(
                    "Hidden Layer Dimensions",
                    options=[16, 32, 64, 128, 256],
                    default=[64, 32],
                    help="Dimensions of hidden layers"
                )
            
            with col2:
                params['learning_rate'] = st.selectbox(
                    "Learning Rate",
                    options=[0.0001, 0.001, 0.01, 0.1],
                    index=1,
                    help="Learning rate for optimizer"
                )
                
                params['epochs'] = st.slider(
                    "Epochs",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    help="Number of training epochs"
                )
        
        return params
    
    @staticmethod
    def create_data_filter(data: pd.DataFrame) -> pd.DataFrame:
        """Create an interactive data filtering interface."""
        st.markdown("### 🔍 Data Filters")
        
        col1, col2, col3 = st.columns(3)
        
        filtered_data = data.copy()
        
        with col1:
            # Symbol filter
            if 'Symbol' in data.columns:
                symbols = st.multiselect(
                    "Select Symbols",
                    options=data['Symbol'].unique(),
                    default=data['Symbol'].unique(),
                    help="Filter by specific symbols"
                )
                filtered_data = filtered_data[filtered_data['Symbol'].isin(symbols)]
        
        with col2:
            # Date range filter
            if 'Date' in data.columns:
                date_range = st.date_input(
                    "Date Range",
                    value=(data['Date'].min().date(), data['Date'].max().date()),
                    min_value=data['Date'].min().date(),
                    max_value=data['Date'].max().date(),
                    help="Filter by date range"
                )
                
                if len(date_range) == 2:
                    filtered_data = filtered_data[
                        (filtered_data['Date'].dt.date >= date_range[0]) &
                        (filtered_data['Date'].dt.date <= date_range[1])
                    ]
        
        with col3:
            # Price range filter
            if 'Close' in data.columns:
                price_range = st.slider(
                    "Price Range",
                    min_value=float(data['Close'].min()),
                    max_value=float(data['Close'].max()),
                    value=(float(data['Close'].min()), float(data['Close'].max())),
                    help="Filter by price range"
                )
                filtered_data = filtered_data[
                    (filtered_data['Close'] >= price_range[0]) &
                    (filtered_data['Close'] <= price_range[1])
                ]
        
        # Show filter summary
        st.info(f"Filtered data: {len(filtered_data):,} records (from {len(data):,} total)")
        
        return filtered_data
    
    @staticmethod
    def create_export_options(data: pd.DataFrame, results: Dict = None) -> None:
        """Create export options for data and results."""
        st.markdown("### 📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Data (CSV)", use_container_width=True):
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"financial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Export Data (Excel)", use_container_width=True):
                # Create Excel file with multiple sheets
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    data.to_excel(writer, sheet_name='Data', index=False)
                    if results:
                        for model_name, result in results.items():
                            result_df = pd.DataFrame({
                                'Date': data['Date'],
                                'Predictions': result['predictions'],
                                'Scores': result['scores']
                            })
                            result_df.to_excel(writer, sheet_name=f'{model_name}_Results', index=False)
                
                st.download_button(
                    label="Download Excel",
                    data=output.getvalue(),
                    file_name=f"financial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col3:
            if st.button("📋 Export Report (JSON)", use_container_width=True):
                report_data = {
                    'timestamp': datetime.now().isoformat(),
                    'data_summary': {
                        'total_records': len(data),
                        'symbols': data['Symbol'].nunique() if 'Symbol' in data.columns else 0,
                        'date_range': {
                            'start': data['Date'].min().isoformat() if 'Date' in data.columns else None,
                            'end': data['Date'].max().isoformat() if 'Date' in data.columns else None
                        }
                    },
                    'results': results
                }
                
                json_data = json.dumps(report_data, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

class RealTimeComponents:
    """Real-time monitoring components."""
    
    @staticmethod
    def create_live_metrics(metrics: Dict[str, float]) -> None:
        """Create live updating metrics display."""
        st.markdown("### 📊 Live Metrics")
        
        cols = st.columns(len(metrics))
        
        for i, (name, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(
                    label=name,
                    value=f"{value:.2f}",
                    delta=f"{np.random.uniform(-5, 5):.1f}%"  # Simulated real-time change
                )
    
    @staticmethod
    def create_auto_refresh_toggle() -> bool:
        """Create auto-refresh toggle with interval selection."""
        col1, col2 = st.columns([1, 2])
        
        with col1:
            auto_refresh = st.checkbox("🔄 Auto Refresh", value=False)
        
        with col2:
            if auto_refresh:
                refresh_interval = st.selectbox(
                    "Refresh Interval",
                    options=[5, 10, 30, 60],
                    index=1,
                    format_func=lambda x: f"{x} seconds"
                )
                
                # Auto-refresh logic
                if auto_refresh:
                    time.sleep(refresh_interval)
                    st.rerun()
        
        return auto_refresh
    
    @staticmethod
    def create_alert_system(thresholds: Dict[str, float], current_values: Dict[str, float]) -> List[str]:
        """Create an alert system for threshold monitoring."""
        alerts = []
        
        for metric, threshold in thresholds.items():
            if metric in current_values:
                value = current_values[metric]
                if value > threshold:
                    alerts.append(f"⚠️ {metric} exceeded threshold: {value:.2f} > {threshold:.2f}")
                elif value < -threshold:
                    alerts.append(f"⚠️ {metric} below threshold: {value:.2f} < {-threshold:.2f}")
        
        if alerts:
            st.markdown("### 🚨 Alerts")
            for alert in alerts:
                st.warning(alert)
        
        return alerts

# Additional CSS for enhanced components
ENHANCED_CSS = """
<style>
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .progress-ring-container {
        position: relative;
        display: inline-block;
    }
    
    .progress-ring {
        transform: rotate(-90deg);
    }
    
    .progress-ring-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 1.5rem;
        font-weight: bold;
        color: var(--primary-color);
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .animated-counter {
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary-color);
    }
    
    .parameter-tuner {
        background: var(--light-surface);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin: 1rem 0;
    }
    
    .filter-section {
        background: rgba(99, 102, 241, 0.05);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }
    
    .export-section {
        background: rgba(16, 185, 129, 0.05);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--success-color);
        margin: 1rem 0;
    }
    
    .alert-container {
        background: rgba(239, 68, 68, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--error-color);
        margin: 1rem 0;
    }
    
    .live-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
"""
