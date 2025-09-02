"""
Isolation Forest Anomaly Detection Model

This module implements an Isolation Forest-based anomaly detection system
for financial time series data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple, Dict, Any, Optional, List
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class IsolationForestAnomalyDetector:
    """Isolation Forest-based anomaly detector for financial data."""
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies in the dataset
            n_estimators: Number of base estimators in the ensemble
            max_samples: Number of samples to draw from X to train each base estimator
            max_features: Number of features to draw from X to train each base estimator
            bootstrap: Whether samples are drawn with replacement
            random_state: Random state for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'IsolationForestAnomalyDetector':
        """
        Fit the Isolation Forest model.
        
        Args:
            X: Feature matrix
            y: Target variable (ignored for unsupervised learning)
            
        Returns:
            Self
        """
        logger.info("Fitting Isolation Forest model...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        logger.info(f"Isolation Forest model fitted with {len(self.feature_names)} features")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies in the data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (-1 for anomalies, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute the decision function of the samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of decision function values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing decision function")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Compute decision function
        scores = self.model.decision_function(X_scaled)
        
        return scores
    
    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores (higher values indicate more anomalous).
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of anomaly scores
        """
        # Get decision function values
        scores = self.decision_function(X)
        
        # Convert to anomaly scores (higher = more anomalous)
        anomaly_scores = -scores
        
        return anomaly_scores
    
    def detect_anomalies(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Detect anomalies in the data.
        
        Args:
            X: Feature matrix
            threshold: Custom threshold for anomaly detection (optional)
            
        Returns:
            Tuple of (predictions, anomaly_scores, metadata)
        """
        # Get predictions and scores
        predictions = self.predict(X)
        scores = self.anomaly_scores(X)
        
        # If custom threshold is provided, override predictions
        if threshold is not None:
            predictions = np.where(scores > threshold, -1, 1)
        
        # Calculate metadata
        n_anomalies = np.sum(predictions == -1)
        n_normal = np.sum(predictions == 1)
        anomaly_rate = n_anomalies / len(predictions)
        
        metadata = {
            'n_anomalies': n_anomalies,
            'n_normal': n_normal,
            'anomaly_rate': anomaly_rate,
            'threshold_used': threshold if threshold is not None else 'model_default',
            'contamination': self.contamination
        }
        
        logger.info(f"Detected {n_anomalies} anomalies ({anomaly_rate:.2%} of data)")
        
        return predictions, scores, metadata
    
    def evaluate_model(
        self,
        X: pd.DataFrame,
        y_true: Optional[pd.Series] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model performance.
        
        Args:
            X: Feature matrix
            y_true: True labels (if available)
            threshold: Custom threshold for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions, scores, metadata = self.detect_anomalies(X, threshold)
        
        evaluation_results = {
            'metadata': metadata,
            'predictions': predictions,
            'anomaly_scores': scores
        }
        
        # If true labels are available, compute classification metrics
        if y_true is not None:
            # Convert true labels to binary (assuming 1 = anomaly, 0 = normal)
            y_true_binary = (y_true == 1).astype(int)
            y_pred_binary = (predictions == -1).astype(int)
            
            # Classification report
            report = classification_report(
                y_true_binary, y_pred_binary,
                target_names=['Normal', 'Anomaly'],
                output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            
            evaluation_results.update({
                'classification_report': report,
                'confusion_matrix': cm,
                'accuracy': report['accuracy'],
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1_score': report['1']['f1-score']
            })
        
        return evaluation_results
    
    def plot_anomalies(
        self,
        X: pd.DataFrame,
        dates: Optional[pd.Series] = None,
        prices: Optional[pd.Series] = None,
        threshold: Optional[float] = None,
        figsize: Tuple[int, int] = (15, 8)
    ) -> plt.Figure:
        """
        Plot anomalies on time series data.
        
        Args:
            X: Feature matrix
            dates: Date series for x-axis
            prices: Price series for y-axis
            threshold: Custom threshold for anomaly detection
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        predictions, scores, _ = self.detect_anomalies(X, threshold)
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Price series with anomalies
        if dates is not None and prices is not None:
            axes[0].plot(dates, prices, 'b-', alpha=0.7, label='Price')
            
            # Highlight anomalies
            anomaly_mask = predictions == -1
            if np.any(anomaly_mask):
                axes[0].scatter(
                    dates[anomaly_mask], prices[anomaly_mask],
                    color='red', s=50, alpha=0.8, label='Anomalies', zorder=5
                )
            
            axes[0].set_ylabel('Price')
            axes[0].set_title('Price Series with Detected Anomalies')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Anomaly scores
        if dates is not None:
            axes[1].plot(dates, scores, 'g-', alpha=0.7, label='Anomaly Score')
            
            # Add threshold line if provided
            if threshold is not None:
                axes[1].axhline(y=threshold, color='red', linestyle='--', alpha=0.8, label=f'Threshold: {threshold:.3f}')
            
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Anomaly Score')
            axes[1].set_title('Anomaly Scores Over Time')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot feature importance based on model's feature usage.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted or self.feature_names is None:
            raise ValueError("Model must be fitted before plotting feature importance")
        
        # Get feature importance (average depth across all trees)
        feature_importance = np.zeros(len(self.feature_names))
        
        for tree in self.model.estimators_:
            # Get feature usage in each tree
            tree_importance = np.zeros(len(self.feature_names))
            for i, feature_idx in enumerate(tree.feature_importances_):
                if i < len(self.feature_names):
                    tree_importance[i] = feature_idx
            
            feature_importance += tree_importance
        
        # Average across all trees
        feature_importance /= len(self.model.estimators_)
        
        # Create DataFrame for easier plotting
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(importance_df['feature'], importance_df['importance'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Isolation Forest Feature Importance')
        ax.grid(True, alpha=0.3)
        
        # Color bars by importance
        colors = plt.cm.viridis(importance_df['importance'] / importance_df['importance'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        return fig
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'IsolationForestAnomalyDetector':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.contamination = model_data['contamination']
        self.n_estimators = model_data['n_estimators']
        self.max_samples = model_data['max_samples']
        self.max_features = model_data['max_features']
        self.bootstrap = model_data['bootstrap']
        self.random_state = model_data['random_state']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'Isolation Forest',
            'is_fitted': self.is_fitted,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate normal data
    X_normal = np.random.randn(n_samples, n_features)
    
    # Generate some anomalies
    X_anomalies = np.random.randn(50, n_features) * 3 + 5
    X = np.vstack([X_normal, X_anomalies])
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Create true labels for evaluation
    y_true = np.hstack([np.zeros(n_samples), np.ones(50)])
    
    # Create and fit model
    detector = IsolationForestAnomalyDetector(contamination=0.05)
    detector.fit(X_df)
    
    # Detect anomalies
    predictions, scores, metadata = detector.detect_anomalies(X_df)
    
    print(f"Detected {metadata['n_anomalies']} anomalies")
    print(f"Anomaly rate: {metadata['anomaly_rate']:.2%}")
    
    # Evaluate model
    evaluation = detector.evaluate_model(X_df, y_true)
    print(f"Accuracy: {evaluation['accuracy']:.3f}")
    print(f"Precision: {evaluation['precision']:.3f}")
    print(f"Recall: {evaluation['recall']:.3f}")
    print(f"F1-Score: {evaluation['f1_score']:.3f}")
    
    # Get model info
    model_info = detector.get_model_info()
    print(f"Model info: {model_info}")
