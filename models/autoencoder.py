"""
Autoencoder-based Anomaly Detection Model

This module implements an Autoencoder-based anomaly detection system
for financial time series data using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple, Dict, Any, Optional, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """Autoencoder neural network for anomaly detection."""
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int,
        hidden_dims: List[int] = None,
        activation: str = 'relu',
        dropout_rate: float = 0.2
    ):
        """
        Initialize Autoencoder.
        
        Args:
            input_dim: Input dimension
            encoding_dim: Encoding dimension (bottleneck)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            dropout_rate: Dropout rate
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims or [input_dim // 2, input_dim // 4]
        self.dropout_rate = dropout_rate
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Bottleneck
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = encoding_dim
        
        # Reverse hidden dimensions
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """Forward pass through the autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, encoded):
        """Decode latent representation to output."""
        return self.decoder(encoded)


class AutoencoderAnomalyDetector:
    """Autoencoder-based anomaly detector for financial data."""
    
    def __init__(
        self,
        encoding_dim: int = 32,
        hidden_dims: List[int] = None,
        activation: str = 'relu',
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        device: str = 'auto'
    ):
        """
        Initialize Autoencoder anomaly detector.
        
        Args:
            encoding_dim: Encoding dimension (bottleneck)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            dropout_rate: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.model = None
        self.training_history = {'train_loss': [], 'val_loss': []}
    
    def _create_model(self, input_dim: int) -> Autoencoder:
        """Create the autoencoder model."""
        return Autoencoder(
            input_dim=input_dim,
            encoding_dim=self.encoding_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        ).to(self.device)
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> 'AutoencoderAnomalyDetector':
        """
        Fit the Autoencoder model.
        
        Args:
            X: Feature matrix
            y: Target variable (ignored for unsupervised learning)
            validation_split: Fraction of data to use for validation
            verbose: Whether to print training progress
            
        Returns:
            Self
        """
        logger.info("Fitting Autoencoder model...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Split data for validation
        n_samples = len(X_tensor)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        # Shuffle indices
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_train = X_tensor[train_indices]
        X_val = X_tensor[val_indices]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, X_train)  # Autoencoder uses input as target
        val_dataset = TensorDataset(X_val, X_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Create model
        self.model = self._create_model(X.shape[1])
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            # Average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        self.is_fitted = True
        logger.info(f"Autoencoder model fitted with {len(self.feature_names)} features")
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
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Get reconstructions
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
        
        # Calculate reconstruction errors
        reconstruction_errors = torch.mean((X_tensor - reconstructions) ** 2, dim=1)
        reconstruction_errors = reconstruction_errors.cpu().numpy()
        
        # Use threshold to classify anomalies
        threshold = np.percentile(reconstruction_errors, 95)  # Top 5% as anomalies
        predictions = np.where(reconstruction_errors > threshold, -1, 1)
        
        return predictions
    
    def reconstruction_errors(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate reconstruction errors for the data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of reconstruction errors
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing reconstruction errors")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Get reconstructions
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
        
        # Calculate reconstruction errors
        reconstruction_errors = torch.mean((X_tensor - reconstructions) ** 2, dim=1)
        
        return reconstruction_errors.cpu().numpy()
    
    def detect_anomalies(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
        percentile: float = 95
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Detect anomalies in the data.
        
        Args:
            X: Feature matrix
            threshold: Custom threshold for anomaly detection
            percentile: Percentile to use for threshold if not provided
            
        Returns:
            Tuple of (predictions, reconstruction_errors, metadata)
        """
        # Get reconstruction errors
        reconstruction_errors = self.reconstruction_errors(X)
        
        # Determine threshold
        if threshold is None:
            threshold = np.percentile(reconstruction_errors, percentile)
        
        # Make predictions
        predictions = np.where(reconstruction_errors > threshold, -1, 1)
        
        # Calculate metadata
        n_anomalies = np.sum(predictions == -1)
        n_normal = np.sum(predictions == 1)
        anomaly_rate = n_anomalies / len(predictions)
        
        metadata = {
            'n_anomalies': n_anomalies,
            'n_normal': n_normal,
            'anomaly_rate': anomaly_rate,
            'threshold_used': threshold,
            'percentile_used': percentile
        }
        
        logger.info(f"Detected {n_anomalies} anomalies ({anomaly_rate:.2%} of data)")
        
        return predictions, reconstruction_errors, metadata
    
    def evaluate_model(
        self,
        X: pd.DataFrame,
        y_true: Optional[pd.Series] = None,
        threshold: Optional[float] = None,
        percentile: float = 95
    ) -> Dict[str, Any]:
        """
        Evaluate the model performance.
        
        Args:
            X: Feature matrix
            y_true: True labels (if available)
            threshold: Custom threshold for evaluation
            percentile: Percentile to use for threshold if not provided
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions, reconstruction_errors, metadata = self.detect_anomalies(X, threshold, percentile)
        
        evaluation_results = {
            'metadata': metadata,
            'predictions': predictions,
            'reconstruction_errors': reconstruction_errors
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
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
        """
        Plot training history.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot losses
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot learning rate (if available)
        if hasattr(self, 'learning_rates'):
            ax2.plot(epochs, self.learning_rates, 'g-', label='Learning Rate')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_anomalies(
        self,
        X: pd.DataFrame,
        dates: Optional[pd.Series] = None,
        prices: Optional[pd.Series] = None,
        threshold: Optional[float] = None,
        percentile: float = 95,
        figsize: Tuple[int, int] = (15, 8)
    ) -> plt.Figure:
        """
        Plot anomalies on time series data.
        
        Args:
            X: Feature matrix
            dates: Date series for x-axis
            prices: Price series for y-axis
            threshold: Custom threshold for anomaly detection
            percentile: Percentile to use for threshold if not provided
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        predictions, reconstruction_errors, _ = self.detect_anomalies(X, threshold, percentile)
        
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
        
        # Plot 2: Reconstruction errors
        if dates is not None:
            axes[1].plot(dates, reconstruction_errors, 'g-', alpha=0.7, label='Reconstruction Error')
            
            # Add threshold line
            if threshold is None:
                threshold = np.percentile(reconstruction_errors, percentile)
            axes[1].axhline(y=threshold, color='red', linestyle='--', alpha=0.8, label=f'Threshold: {threshold:.3f}')
            
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Reconstruction Error')
            axes[1].set_title('Reconstruction Errors Over Time')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
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
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'encoding_dim': self.encoding_dim,
                'hidden_dims': self.hidden_dims,
                'activation': self.activation,
                'dropout_rate': self.dropout_rate
            },
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }
        
        torch.save(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'AutoencoderAnomalyDetector':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self
        """
        model_data = torch.load(filepath, map_location=self.device)
        
        # Restore configuration
        self.encoding_dim = model_data['model_config']['encoding_dim']
        self.hidden_dims = model_data['model_config']['hidden_dims']
        self.activation = model_data['model_config']['activation']
        self.dropout_rate = model_data['model_config']['dropout_rate']
        self.learning_rate = model_data['learning_rate']
        self.batch_size = model_data['batch_size']
        self.epochs = model_data['epochs']
        
        # Restore scaler and feature names
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data['training_history']
        
        # Recreate and load model
        self.model = self._create_model(model_data['model_config']['input_dim'])
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.eval()
        
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
            'model_type': 'Autoencoder',
            'is_fitted': self.is_fitted,
            'encoding_dim': self.encoding_dim,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': str(self.device),
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
    detector = AutoencoderAnomalyDetector(
        encoding_dim=5,
        hidden_dims=[8, 6],
        epochs=50,
        batch_size=32
    )
    detector.fit(X_df, verbose=True)
    
    # Detect anomalies
    predictions, reconstruction_errors, metadata = detector.detect_anomalies(X_df)
    
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
