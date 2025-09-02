"""
Model Evaluation Framework

This module provides comprehensive evaluation tools for anomaly detection models
including metrics, visualization, and comparison utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, silhouette_score
)
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AnomalyDetectionEvaluator:
    """Comprehensive evaluator for anomaly detection models."""
    
    def __init__(self):
        self.results = {}
        self.model_comparisons = {}
    
    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anomaly_scores: np.ndarray,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single anomaly detection model.
        
        Args:
            model_name: Name of the model
            y_true: True labels (1 for anomaly, 0 for normal)
            y_pred: Predicted labels (-1 for anomaly, 1 for normal)
            anomaly_scores: Anomaly scores (higher = more anomalous)
            metadata: Additional metadata about the model
            
        Returns:
            Dictionary with evaluation results
        """
        # Convert predictions to binary format
        y_pred_binary = (y_pred == -1).astype(int)
        
        # Basic metrics
        n_anomalies = np.sum(y_true == 1)
        n_normal = np.sum(y_true == 0)
        n_pred_anomalies = np.sum(y_pred_binary == 1)
        n_pred_normal = np.sum(y_pred_binary == 0)
        
        # Classification metrics
        accuracy = np.mean(y_true == y_pred_binary)
        precision = np.sum((y_true == 1) & (y_pred_binary == 1)) / max(n_pred_anomalies, 1)
        recall = np.sum((y_true == 1) & (y_pred_binary == 1)) / max(n_anomalies, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-8)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        
        # ROC AUC (if we have anomaly scores)
        roc_auc = None
        if len(np.unique(anomaly_scores)) > 1:
            try:
                roc_auc = roc_auc_score(y_true, anomaly_scores)
            except ValueError:
                roc_auc = None
        
        # Precision-Recall AUC
        pr_auc = None
        if len(np.unique(anomaly_scores)) > 1:
            try:
                pr_auc = average_precision_score(y_true, anomaly_scores)
            except ValueError:
                pr_auc = None
        
        # Store results
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'basic_metrics': {
                'n_anomalies_true': int(n_anomalies),
                'n_normal_true': int(n_normal),
                'n_anomalies_pred': int(n_pred_anomalies),
                'n_normal_pred': int(n_pred_normal),
                'anomaly_rate_true': float(n_anomalies / len(y_true)),
                'anomaly_rate_pred': float(n_pred_anomalies / len(y_pred))
            },
            'classification_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'specificity': float(np.sum((y_true == 0) & (y_pred_binary == 0)) / max(n_normal, 1))
            },
            'ranking_metrics': {
                'roc_auc': float(roc_auc) if roc_auc is not None else None,
                'pr_auc': float(pr_auc) if pr_auc is not None else None
            },
            'confusion_matrix': cm.tolist(),
            'metadata': metadata or {}
        }
        
        self.results[model_name] = results
        logger.info(f"Evaluation completed for {model_name}")
        
        return results
    
    def compare_models(self, model_names: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            model_names: List of model names to compare (if None, compare all)
            
        Returns:
            DataFrame with comparison results
        """
        if model_names is None:
            model_names = list(self.results.keys())
        
        comparison_data = []
        
        for model_name in model_names:
            if model_name not in self.results:
                logger.warning(f"Model {model_name} not found in results")
                continue
            
            result = self.results[model_name]
            
            comparison_data.append({
                'Model': model_name,
                'Accuracy': result['classification_metrics']['accuracy'],
                'Precision': result['classification_metrics']['precision'],
                'Recall': result['classification_metrics']['recall'],
                'F1-Score': result['classification_metrics']['f1_score'],
                'Specificity': result['classification_metrics']['specificity'],
                'ROC-AUC': result['ranking_metrics']['roc_auc'],
                'PR-AUC': result['ranking_metrics']['pr_auc'],
                'True Anomaly Rate': result['basic_metrics']['anomaly_rate_true'],
                'Predicted Anomaly Rate': result['basic_metrics']['anomaly_rate_pred']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        self.model_comparisons = comparison_df
        
        return comparison_df
    
    def plot_confusion_matrices(self, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        Plot confusion matrices for all models.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_models = len(self.results)
        if n_models == 0:
            raise ValueError("No results to plot")
        
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, result) in enumerate(self.results.items()):
            cm = np.array(result['confusion_matrix'])
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                ax=axes[i]
            )
            axes[i].set_title(f'{model_name}\nConfusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot ROC curves for all models.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for model_name, result in self.results.items():
            if result['ranking_metrics']['roc_auc'] is not None:
                # Note: This is a simplified version. In practice, you'd need
                # the actual anomaly scores to plot the ROC curve
                roc_auc = result['ranking_metrics']['roc_auc']
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                ax.text(0.6, 0.2, f'{model_name}: AUC = {roc_auc:.3f}')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_comparison(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot metrics comparison for all models.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.model_comparisons.empty:
            comparison_df = self.model_comparisons
        else:
            comparison_df = self.compare_models()
        
        # Select metrics to plot
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        n_metrics = len(available_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics):
            if i < len(axes):
                bars = axes[i].bar(comparison_df['Model'], comparison_df[metric])
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, comparison_df[metric]):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_anomaly_rates(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot true vs predicted anomaly rates.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.model_comparisons.empty:
            comparison_df = self.model_comparisons
        else:
            comparison_df = self.compare_models()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(comparison_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, comparison_df['True Anomaly Rate'], width, label='True', alpha=0.8)
        bars2 = ax.bar(x + width/2, comparison_df['Predicted Anomaly Rate'], width, label='Predicted', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Anomaly Rate')
        ax.set_title('True vs Predicted Anomaly Rates')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, output_file: str = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_file: Path to save the report (optional)
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ANOMALY DETECTION MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Model comparison summary
        if not self.model_comparisons.empty:
            report_lines.append("MODEL COMPARISON SUMMARY")
            report_lines.append("-" * 40)
            report_lines.append(self.model_comparisons.to_string(index=False))
            report_lines.append("")
        
        # Detailed results for each model
        for model_name, result in self.results.items():
            report_lines.append(f"DETAILED RESULTS: {model_name}")
            report_lines.append("-" * 40)
            
            # Basic metrics
            report_lines.append("Basic Metrics:")
            for key, value in result['basic_metrics'].items():
                report_lines.append(f"  {key}: {value}")
            report_lines.append("")
            
            # Classification metrics
            report_lines.append("Classification Metrics:")
            for key, value in result['classification_metrics'].items():
                report_lines.append(f"  {key}: {value:.4f}")
            report_lines.append("")
            
            # Ranking metrics
            report_lines.append("Ranking Metrics:")
            for key, value in result['ranking_metrics'].items():
                if value is not None:
                    report_lines.append(f"  {key}: {value:.4f}")
                else:
                    report_lines.append(f"  {key}: N/A")
            report_lines.append("")
            
            # Confusion matrix
            report_lines.append("Confusion Matrix:")
            cm = np.array(result['confusion_matrix'])
            report_lines.append(f"  Normal -> Normal: {cm[0, 0]}")
            report_lines.append(f"  Normal -> Anomaly: {cm[0, 1]}")
            report_lines.append(f"  Anomaly -> Normal: {cm[1, 0]}")
            report_lines.append(f"  Anomaly -> Anomaly: {cm[1, 1]}")
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")
        
        return report
    
    def save_results(self, filepath: str) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            filepath: Path to save the results
        """
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """
        Load evaluation results from JSON file.
        
        Args:
            filepath: Path to load the results from
        """
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        logger.info(f"Results loaded from {filepath}")
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, float]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, metric_value)
        """
        if not self.model_comparisons.empty:
            comparison_df = self.model_comparisons
        else:
            comparison_df = self.compare_models()
        
        if metric not in comparison_df.columns:
            raise ValueError(f"Metric {metric} not found in comparison results")
        
        best_idx = comparison_df[metric].idxmax()
        best_model = comparison_df.loc[best_idx, 'Model']
        best_value = comparison_df.loc[best_idx, metric]
        
        return best_model, best_value


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate true labels (5% anomalies)
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    # Generate predictions for different models
    models_data = {
        'Isolation Forest': {
            'y_pred': np.random.choice([-1, 1], size=n_samples, p=[0.1, 0.9]),
            'scores': np.random.rand(n_samples)
        },
        'Autoencoder': {
            'y_pred': np.random.choice([-1, 1], size=n_samples, p=[0.08, 0.92]),
            'scores': np.random.rand(n_samples)
        },
        'GNN': {
            'y_pred': np.random.choice([-1, 1], size=n_samples, p=[0.12, 0.88]),
            'scores': np.random.rand(n_samples)
        }
    }
    
    # Create evaluator
    evaluator = AnomalyDetectionEvaluator()
    
    # Evaluate each model
    for model_name, data in models_data.items():
        result = evaluator.evaluate_model(
            model_name=model_name,
            y_true=y_true,
            y_pred=data['y_pred'],
            anomaly_scores=data['scores']
        )
        print(f"Evaluated {model_name}: F1-Score = {result['classification_metrics']['f1_score']:.3f}")
    
    # Compare models
    comparison = evaluator.compare_models()
    print("\nModel Comparison:")
    print(comparison[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']])
    
    # Get best model
    best_model, best_f1 = evaluator.get_best_model('F1-Score')
    print(f"\nBest model: {best_model} (F1-Score: {best_f1:.3f})")
    
    # Generate report
    report = evaluator.generate_report()
    print("\nReport generated successfully!")
