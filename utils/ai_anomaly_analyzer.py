"""
AI-Powered Anomaly Analyzer
Provides intelligent explanations and insights for detected financial anomalies
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class AIAnomalyAnalyzer:
    """
    AI-powered analyzer that provides intelligent explanations for financial anomalies
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, use_local_llm: bool = False):
        """
        Initialize the AI anomaly analyzer
        
        Args:
            openai_api_key: OpenAI API key for GPT-4 access
            use_local_llm: Whether to use local LLM (Ollama) instead of OpenAI
        """
        self.openai_api_key = openai_api_key
        self.use_local_llm = use_local_llm
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE and openai_api_key:
            openai.api_key = openai_api_key
            self.openai_client = openai
        else:
            self.openai_client = None
            
        # Local LLM endpoint (Ollama)
        self.local_llm_url = "http://localhost:11434/api/generate"
        
    def analyze_anomaly(self, 
                       anomaly_data: Dict, 
                       market_context: Dict,
                       historical_data: pd.DataFrame) -> Dict:
        """
        Analyze a single anomaly and provide AI-powered insights
        
        Args:
            anomaly_data: Dictionary containing anomaly information
            market_context: Market context and conditions
            historical_data: Historical data for pattern analysis
            
        Returns:
            Dictionary with AI analysis, explanation, and recommendations
        """
        try:
            # Prepare context for AI analysis
            analysis_context = self._prepare_analysis_context(
                anomaly_data, market_context, historical_data
            )
            
            # Get AI explanation
            if self.use_local_llm:
                ai_explanation = self._get_local_llm_explanation(analysis_context)
            elif self.openai_client:
                ai_explanation = self._get_openai_explanation(analysis_context)
            else:
                ai_explanation = self._get_fallback_explanation(analysis_context)
            
            # Enhance with additional analysis
            enhanced_analysis = self._enhance_analysis(ai_explanation, analysis_context)
            
            return enhanced_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing anomaly: {e}")
            return self._get_error_analysis(str(e))
    
    def _prepare_analysis_context(self, 
                                anomaly_data: Dict, 
                                market_context: Dict,
                                historical_data: pd.DataFrame) -> Dict:
        """Prepare comprehensive context for AI analysis"""
        
        # Extract key anomaly metrics
        anomaly_metrics = {
            'score': anomaly_data.get('score', 0),
            'timestamp': anomaly_data.get('timestamp', datetime.now()),
            'symbol': anomaly_data.get('symbol', 'Unknown'),
            'price_change': anomaly_data.get('price_change', 0),
            'volume_ratio': anomaly_data.get('volume_ratio', 1),
            'technical_indicators': anomaly_data.get('technical_indicators', {}),
            'features': anomaly_data.get('features', {})
        }
        
        # Calculate additional context
        context = {
            'anomaly_metrics': anomaly_metrics,
            'market_context': market_context,
            'historical_patterns': self._analyze_historical_patterns(historical_data, anomaly_metrics),
            'risk_assessment': self._assess_risk_level(anomaly_metrics),
            'similar_events': self._find_similar_events(historical_data, anomaly_metrics)
        }
        
        return context
    
    def _get_openai_explanation(self, context: Dict) -> Dict:
        """Get AI explanation using OpenAI GPT-4"""
        
        prompt = self._create_analysis_prompt(context)
        
        try:
            response = self.openai_client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst AI specializing in anomaly detection and market analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content
            
            return {
                'explanation': ai_response,
                'confidence': 0.9,
                'source': 'OpenAI GPT-4',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return self._get_fallback_explanation(context)
    
    def _get_local_llm_explanation(self, context: Dict) -> Dict:
        """Get AI explanation using local LLM (Ollama)"""
        
        prompt = self._create_analysis_prompt(context)
        
        try:
            payload = {
                "model": "llama2:13b",  # or whatever model you have
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.local_llm_url, json=payload, timeout=30)
            response.raise_for_status()
            
            ai_response = response.json().get('response', '')
            
            return {
                'explanation': ai_response,
                'confidence': 0.8,
                'source': 'Local LLM (Ollama)',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Local LLM error: {e}")
            return self._get_fallback_explanation(context)
    
    def _get_fallback_explanation(self, context: Dict) -> Dict:
        """Fallback explanation when AI services are unavailable"""
        
        anomaly_metrics = context['anomaly_metrics']
        risk_level = context['risk_assessment']
        
        # Create rule-based explanation
        explanation_parts = []
        
        if anomaly_metrics['score'] > 0.8:
            explanation_parts.append("This is a high-confidence anomaly with significant deviation from normal patterns.")
        
        if anomaly_metrics['volume_ratio'] > 2:
            explanation_parts.append(f"Trading volume is {anomaly_metrics['volume_ratio']:.1f}x normal, suggesting unusual market activity.")
        
        if abs(anomaly_metrics['price_change']) > 0.05:
            explanation_parts.append(f"Price change of {anomaly_metrics['price_change']:.2%} is statistically significant.")
        
        if risk_level == 'High':
            explanation_parts.append("High risk level detected - monitor closely for potential market impact.")
        
        explanation = " ".join(explanation_parts) if explanation_parts else "Anomaly detected with standard deviation from normal patterns."
        
        return {
            'explanation': explanation,
            'confidence': 0.6,
            'source': 'Rule-based Analysis',
            'timestamp': datetime.now()
        }
    
    def _create_analysis_prompt(self, context: Dict) -> str:
        """Create a comprehensive prompt for AI analysis"""
        
        anomaly_metrics = context['anomaly_metrics']
        market_context = context['market_context']
        historical_patterns = context['historical_patterns']
        risk_assessment = context['risk_assessment']
        
        prompt = f"""
        Analyze this financial anomaly and provide a comprehensive explanation:
        
        ANOMALY DETAILS:
        - Symbol: {anomaly_metrics['symbol']}
        - Anomaly Score: {anomaly_metrics['score']:.3f}
        - Price Change: {anomaly_metrics['price_change']:.2%}
        - Volume Ratio: {anomaly_metrics['volume_ratio']:.1f}x
        - Timestamp: {anomaly_metrics['timestamp']}
        
        TECHNICAL INDICATORS:
        {json.dumps(anomaly_metrics['technical_indicators'], indent=2)}
        
        MARKET CONTEXT:
        {json.dumps(market_context, indent=2)}
        
        HISTORICAL PATTERNS:
        {json.dumps(historical_patterns, indent=2)}
        
        RISK ASSESSMENT: {risk_assessment}
        
        Please provide:
        1. A clear explanation of why this is anomalous
        2. Potential causes or triggers
        3. Historical context and similar events
        4. Risk assessment and implications
        5. Trading recommendations or actions to consider
        6. Confidence level in your analysis
        
        Format your response as a structured analysis with clear sections.
        """
        
        return prompt
    
    def _analyze_historical_patterns(self, historical_data: pd.DataFrame, anomaly_metrics: Dict) -> Dict:
        """Analyze historical patterns to provide context"""
        
        try:
            if historical_data.empty:
                return {'status': 'No historical data available'}
            
            # Calculate basic statistics
            price_volatility = historical_data['close'].pct_change().std()
            volume_volatility = historical_data['volume'].std()
            
            # Find similar price movements
            price_changes = historical_data['close'].pct_change()
            similar_movements = price_changes[abs(price_changes - anomaly_metrics['price_change']) < 0.01]
            
            return {
                'price_volatility': float(price_volatility),
                'volume_volatility': float(volume_volatility),
                'similar_price_movements': len(similar_movements),
                'data_points': len(historical_data),
                'time_range': f"{historical_data.index[0]} to {historical_data.index[-1]}"
            }
            
        except Exception as e:
            return {'error': f'Pattern analysis failed: {str(e)}'}
    
    def _assess_risk_level(self, anomaly_metrics: Dict) -> str:
        """Assess the risk level of the anomaly"""
        
        score = anomaly_metrics['score']
        price_change = abs(anomaly_metrics['price_change'])
        volume_ratio = anomaly_metrics['volume_ratio']
        
        risk_score = 0
        
        # Score-based risk
        if score > 0.8:
            risk_score += 3
        elif score > 0.6:
            risk_score += 2
        elif score > 0.4:
            risk_score += 1
        
        # Price change risk
        if price_change > 0.1:
            risk_score += 3
        elif price_change > 0.05:
            risk_score += 2
        elif price_change > 0.02:
            risk_score += 1
        
        # Volume risk
        if volume_ratio > 5:
            risk_score += 2
        elif volume_ratio > 3:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 6:
            return 'High'
        elif risk_score >= 4:
            return 'Medium'
        elif risk_score >= 2:
            return 'Low'
        else:
            return 'Very Low'
    
    def _find_similar_events(self, historical_data: pd.DataFrame, anomaly_metrics: Dict) -> List[Dict]:
        """Find similar historical events"""
        
        try:
            if historical_data.empty:
                return []
            
            similar_events = []
            
            # Look for similar price movements
            price_changes = historical_data['close'].pct_change()
            volume_ratios = historical_data['volume'] / historical_data['volume'].rolling(20).mean()
            
            for i, (price_change, volume_ratio) in enumerate(zip(price_changes, volume_ratios)):
                if (abs(price_change - anomaly_metrics['price_change']) < 0.01 and
                    abs(volume_ratio - anomaly_metrics['volume_ratio']) < 0.5):
                    
                    similar_events.append({
                        'date': historical_data.index[i],
                        'price_change': float(price_change),
                        'volume_ratio': float(volume_ratio),
                        'similarity_score': 1 - abs(price_change - anomaly_metrics['price_change'])
                    })
            
            # Sort by similarity and return top 5
            similar_events.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_events[:5]
            
        except Exception as e:
            return []
    
    def _enhance_analysis(self, ai_explanation: Dict, context: Dict) -> Dict:
        """Enhance AI explanation with additional analysis"""
        
        enhanced = ai_explanation.copy()
        
        # Add structured insights
        enhanced['insights'] = {
            'risk_level': context['risk_assessment'],
            'confidence': ai_explanation['confidence'],
            'similar_events_count': len(context['similar_events']),
            'historical_context': context['historical_patterns']
        }
        
        # Add recommendations
        enhanced['recommendations'] = self._generate_recommendations(context)
        
        # Add metadata
        enhanced['metadata'] = {
            'analysis_timestamp': datetime.now(),
            'anomaly_timestamp': context['anomaly_metrics']['timestamp'],
            'symbol': context['anomaly_metrics']['symbol'],
            'analysis_version': '1.0'
        }
        
        return enhanced
    
    def _generate_recommendations(self, context: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        risk_level = context['risk_assessment']
        anomaly_metrics = context['anomaly_metrics']
        
        if risk_level == 'High':
            recommendations.extend([
                "Monitor closely for potential market impact",
                "Consider reducing position size or adding stop-loss",
                "Watch for news or events that might explain the anomaly"
            ])
        elif risk_level == 'Medium':
            recommendations.extend([
                "Continue monitoring the situation",
                "Consider taking partial profits if position is profitable",
                "Be prepared for increased volatility"
            ])
        else:
            recommendations.extend([
                "Standard monitoring recommended",
                "Consider this as a potential trading opportunity",
                "Watch for follow-through patterns"
            ])
        
        # Add specific recommendations based on metrics
        if anomaly_metrics['volume_ratio'] > 3:
            recommendations.append("High volume suggests significant market interest - investigate further")
        
        if abs(anomaly_metrics['price_change']) > 0.05:
            recommendations.append("Large price movement detected - consider risk management")
        
        return recommendations
    
    def _get_error_analysis(self, error_message: str) -> Dict:
        """Return error analysis when something goes wrong"""
        
        return {
            'explanation': f"Analysis failed due to error: {error_message}",
            'confidence': 0.0,
            'source': 'Error',
            'timestamp': datetime.now(),
            'insights': {'risk_level': 'Unknown', 'confidence': 0.0},
            'recommendations': ['Manual analysis recommended', 'Check data quality'],
            'metadata': {'analysis_timestamp': datetime.now(), 'error': True}
        }
    
    def batch_analyze_anomalies(self, 
                               anomalies: List[Dict], 
                               market_context: Dict,
                               historical_data: pd.DataFrame) -> List[Dict]:
        """Analyze multiple anomalies in batch"""
        
        results = []
        
        for anomaly in anomalies:
            try:
                analysis = self.analyze_anomaly(anomaly, market_context, historical_data)
                results.append(analysis)
            except Exception as e:
                self.logger.error(f"Error analyzing anomaly {anomaly.get('symbol', 'Unknown')}: {e}")
                results.append(self._get_error_analysis(str(e)))
        
        return results
    
    def get_analysis_summary(self, analyses: List[Dict]) -> Dict:
        """Get summary of multiple anomaly analyses"""
        
        if not analyses:
            return {'summary': 'No analyses available'}
        
        # Calculate aggregate metrics
        total_anomalies = len(analyses)
        high_risk_count = sum(1 for a in analyses if a.get('insights', {}).get('risk_level') == 'High')
        avg_confidence = np.mean([a.get('confidence', 0) for a in analyses])
        
        # Get top recommendations
        all_recommendations = []
        for analysis in analyses:
            all_recommendations.extend(analysis.get('recommendations', []))
        
        # Count recommendation frequency
        rec_counts = {}
        for rec in all_recommendations:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        top_recommendations = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_anomalies': total_anomalies,
            'high_risk_count': high_risk_count,
            'risk_percentage': (high_risk_count / total_anomalies) * 100 if total_anomalies > 0 else 0,
            'average_confidence': float(avg_confidence),
            'top_recommendations': top_recommendations,
            'analysis_timestamp': datetime.now()
        }
