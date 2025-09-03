"""
AI-Powered Dashboard Components
Enhanced UI components with AI anomaly analysis capabilities
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ai_anomaly_analyzer import AIAnomalyAnalyzer

class AIAnomalyInsights:
    """AI-powered anomaly insights component"""
    
    def __init__(self):
        self.analyzer = None
        self._initialize_analyzer()
    
    def _initialize_analyzer(self):
        """Initialize the AI analyzer with user preferences"""
        # Check if OpenAI API key is available
        openai_key = st.session_state.get('openai_api_key', None)
        use_local_llm = st.session_state.get('use_local_llm', False)
        
        if openai_key or use_local_llm:
            self.analyzer = AIAnomalyAnalyzer(
                openai_api_key=openai_key,
                use_local_llm=use_local_llm
            )
    
    def render_ai_settings(self):
        """Render AI configuration settings"""
        st.subheader("🤖 AI Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.openai_api_key = st.text_input(
                "OpenAI API Key (Optional)",
                value=st.session_state.get('openai_api_key', ''),
                type="password",
                help="Enter your OpenAI API key for GPT-4 powered analysis"
            )
        
        with col2:
            st.session_state.use_local_llm = st.checkbox(
                "Use Local LLM (Ollama)",
                value=st.session_state.get('use_local_llm', False),
                help="Use local Ollama instance instead of OpenAI"
            )
        
        if st.button("🔄 Initialize AI Analyzer", type="primary"):
            self._initialize_analyzer()
            if self.analyzer:
                st.success("✅ AI Analyzer initialized successfully!")
            else:
                st.warning("⚠️ AI Analyzer not available. Using fallback analysis.")
    
    def render_anomaly_analysis(self, anomaly_data: Dict, market_context: Dict, historical_data: pd.DataFrame):
        """Render AI-powered anomaly analysis"""
        
        if not self.analyzer:
            st.warning("⚠️ AI Analyzer not initialized. Please configure AI settings first.")
            return
        
        with st.spinner("🤖 AI is analyzing the anomaly..."):
            try:
                # Get AI analysis
                analysis = self.analyzer.analyze_anomaly(anomaly_data, market_context, historical_data)
                
                # Display the analysis
                self._display_ai_analysis(analysis)
                
            except Exception as e:
                st.error(f"❌ AI analysis failed: {str(e)}")
                st.info("💡 Try configuring AI settings or check your connection.")
    
    def _display_ai_analysis(self, analysis: Dict):
        """Display the AI analysis results"""
        
        # Main analysis section
        st.subheader("🧠 AI Analysis")
        
        # Confidence and source
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence = analysis.get('confidence', 0)
            confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
            st.metric(
                "AI Confidence",
                f"{confidence:.1%}",
                help=f"Confidence level: {confidence:.1%}"
            )
        
        with col2:
            source = analysis.get('source', 'Unknown')
            st.metric(
                "Analysis Source",
                source,
                help=f"Powered by: {source}"
            )
        
        with col3:
            risk_level = analysis.get('insights', {}).get('risk_level', 'Unknown')
            risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
            st.metric(
                "Risk Level",
                risk_level,
                help=f"Risk assessment: {risk_level}"
            )
        
        # AI Explanation
        st.subheader("📝 AI Explanation")
        
        explanation = analysis.get('explanation', 'No explanation available.')
        
        # Create an expandable section for the full explanation
        with st.expander("🔍 View Full AI Analysis", expanded=True):
            st.markdown(explanation)
        
        # Key Insights
        insights = analysis.get('insights', {})
        if insights:
            st.subheader("💡 Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Risk Level:** {insights.get('risk_level', 'Unknown')}")
                st.info(f"**Similar Events:** {insights.get('similar_events_count', 0)} found")
            
            with col2:
                st.info(f"**Confidence:** {insights.get('confidence', 0):.1%}")
                st.info(f"**Analysis Time:** {analysis.get('timestamp', 'Unknown')}")
        
        # Recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            st.subheader("🎯 AI Recommendations")
            
            for i, rec in enumerate(recommendations, 1):
                st.success(f"**{i}.** {rec}")
        
        # Historical Context
        historical_context = insights.get('historical_context', {})
        if historical_context and not historical_context.get('error'):
            st.subheader("📊 Historical Context")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Price Volatility",
                    f"{historical_context.get('price_volatility', 0):.3f}",
                    help="Historical price volatility"
                )
            
            with col2:
                st.metric(
                    "Volume Volatility",
                    f"{historical_context.get('volume_volatility', 0):.0f}",
                    help="Historical volume volatility"
                )
            
            with col3:
                st.metric(
                    "Similar Movements",
                    historical_context.get('similar_price_movements', 0),
                    help="Similar price movements in history"
                )
    
    def render_batch_analysis(self, anomalies: List[Dict], market_context: Dict, historical_data: pd.DataFrame):
        """Render batch analysis for multiple anomalies"""
        
        if not self.analyzer:
            st.warning("⚠️ AI Analyzer not initialized. Please configure AI settings first.")
            return
        
        if not anomalies:
            st.info("ℹ️ No anomalies to analyze.")
            return
        
        st.subheader(f"🤖 AI Batch Analysis ({len(anomalies)} anomalies)")
        
        with st.spinner("🤖 AI is analyzing all anomalies..."):
            try:
                # Get batch analysis
                analyses = self.analyzer.batch_analyze_anomalies(anomalies, market_context, historical_data)
                
                # Display summary
                summary = self.analyzer.get_analysis_summary(analyses)
                self._display_batch_summary(summary)
                
                # Display individual analyses
                self._display_batch_analyses(analyses)
                
            except Exception as e:
                st.error(f"❌ Batch analysis failed: {str(e)}")
    
    def _display_batch_summary(self, summary: Dict):
        """Display batch analysis summary"""
        
        st.subheader("📊 Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Anomalies",
                summary.get('total_anomalies', 0)
            )
        
        with col2:
            st.metric(
                "High Risk",
                summary.get('high_risk_count', 0)
            )
        
        with col3:
            st.metric(
                "Risk Percentage",
                f"{summary.get('risk_percentage', 0):.1f}%"
            )
        
        with col4:
            st.metric(
                "Avg Confidence",
                f"{summary.get('average_confidence', 0):.1%}"
            )
        
        # Top recommendations
        top_recommendations = summary.get('top_recommendations', [])
        if top_recommendations:
            st.subheader("🎯 Top Recommendations")
            
            for rec, count in top_recommendations:
                st.info(f"**{rec}** (mentioned {count} times)")
    
    def _display_batch_analyses(self, analyses: List[Dict]):
        """Display individual analyses in batch"""
        
        st.subheader("🔍 Individual Analyses")
        
        for i, analysis in enumerate(analyses):
            with st.expander(f"Anomaly {i+1}: {analysis.get('metadata', {}).get('symbol', 'Unknown')}", expanded=False):
                self._display_ai_analysis(analysis)
    
    def render_ai_chat_interface(self):
        """Render AI chat interface for market questions"""
        
        st.subheader("💬 AI Market Assistant")
        
        # Initialize chat history
        if 'ai_chat_history' not in st.session_state:
            st.session_state.ai_chat_history = []
        
        # Display chat history
        for message in st.session_state.ai_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about market anomalies, trading strategies, or risk management..."):
            # Add user message to chat history
            st.session_state.ai_chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("🤖 AI is thinking..."):
                    try:
                        response = self._generate_chat_response(prompt)
                        st.markdown(response)
                        
                        # Add AI response to chat history
                        st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                        st.markdown(error_msg)
                        st.session_state.ai_chat_history.append({"role": "assistant", "content": error_msg})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.ai_chat_history = []
            st.rerun()
    
    def _generate_chat_response(self, prompt: str) -> str:
        """Generate AI response for chat interface"""
        
        if not self.analyzer:
            return "⚠️ AI Analyzer not initialized. Please configure AI settings first."
        
        # Simple prompt for chat
        chat_prompt = f"""
        You are a financial analyst AI assistant. Answer this question about financial markets, 
        anomaly detection, or trading strategies: {prompt}
        
        Provide a helpful, accurate, and concise response. If you need more context about 
        specific anomalies or data, let the user know.
        """
        
        try:
            if self.analyzer.openai_client:
                response = self.analyzer.openai_client.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a financial analyst AI assistant."},
                        {"role": "user", "content": chat_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                return response.choices[0].message.content
            else:
                return "💡 AI chat requires OpenAI API key. Please configure it in the AI settings."
                
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

class AIEnhancedVisualizations:
    """AI-enhanced visualization components"""
    
    @staticmethod
    def render_ai_anomaly_timeline(anomalies: List[Dict], analyses: List[Dict]):
        """Render AI-enhanced anomaly timeline"""
        
        if not anomalies or not analyses:
            st.info("ℹ️ No anomalies or analyses to display.")
            return
        
        st.subheader("📈 AI-Enhanced Anomaly Timeline")
        
        # Create timeline data
        timeline_data = []
        for anomaly, analysis in zip(anomalies, analyses):
            timeline_data.append({
                'timestamp': anomaly.get('timestamp', datetime.now()),
                'symbol': anomaly.get('symbol', 'Unknown'),
                'score': anomaly.get('score', 0),
                'risk_level': analysis.get('insights', {}).get('risk_level', 'Unknown'),
                'confidence': analysis.get('confidence', 0),
                'explanation': analysis.get('explanation', '')[:100] + '...' if len(analysis.get('explanation', '')) > 100 else analysis.get('explanation', '')
            })
        
        # Create timeline chart
        fig = go.Figure()
        
        # Color mapping for risk levels
        risk_colors = {
            'High': 'red',
            'Medium': 'orange',
            'Low': 'yellow',
            'Very Low': 'green'
        }
        
        for data in timeline_data:
            fig.add_trace(go.Scatter(
                x=[data['timestamp']],
                y=[data['score']],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=risk_colors.get(data['risk_level'], 'gray'),
                    opacity=0.7
                ),
                text=[data['symbol']],
                textposition="top center",
                hovertemplate=f"""
                <b>{data['symbol']}</b><br>
                Score: {data['score']:.3f}<br>
                Risk: {data['risk_level']}<br>
                Confidence: {data['confidence']:.1%}<br>
                <extra></extra>
                """,
                name=data['risk_level']
            ))
        
        fig.update_layout(
            title="AI-Enhanced Anomaly Timeline",
            xaxis_title="Time",
            yaxis_title="Anomaly Score",
            hovermode='closest',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed timeline
        st.subheader("📋 Detailed Timeline")
        
        for i, data in enumerate(timeline_data):
            with st.expander(f"{data['symbol']} - {data['timestamp'].strftime('%Y-%m-%d %H:%M')}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Score", f"{data['score']:.3f}")
                
                with col2:
                    st.metric("Risk Level", data['risk_level'])
                
                with col3:
                    st.metric("Confidence", f"{data['confidence']:.1%}")
                
                st.markdown(f"**AI Explanation:** {data['explanation']}")
    
    @staticmethod
    def render_risk_distribution_chart(analyses: List[Dict]):
        """Render risk distribution chart"""
        
        if not analyses:
            st.info("ℹ️ No analyses to display.")
            return
        
        st.subheader("📊 Risk Distribution Analysis")
        
        # Count risk levels
        risk_counts = {}
        for analysis in analyses:
            risk_level = analysis.get('insights', {}).get('risk_level', 'Unknown')
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        # Create pie chart
        fig = px.pie(
            values=list(risk_counts.values()),
            names=list(risk_counts.keys()),
            title="Risk Level Distribution",
            color_discrete_map={
                'High': 'red',
                'Medium': 'orange',
                'Low': 'yellow',
                'Very Low': 'green',
                'Unknown': 'gray'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "High Risk Anomalies",
                risk_counts.get('High', 0),
                help="Anomalies requiring immediate attention"
            )
        
        with col2:
            st.metric(
                "Medium Risk Anomalies",
                risk_counts.get('Medium', 0),
                help="Anomalies requiring monitoring"
            )
        
        with col3:
            st.metric(
                "Low Risk Anomalies",
                risk_counts.get('Low', 0) + risk_counts.get('Very Low', 0),
                help="Anomalies with minimal risk"
            )
