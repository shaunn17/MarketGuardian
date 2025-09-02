"""
Graph Neural Network Anomaly Detection Model

This module implements a Graph Neural Network-based anomaly detection system
for correlated financial assets using PyTorch Geometric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple, Dict, Any, Optional, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class GCNAnomalyDetector(nn.Module):
    """Graph Convolutional Network for anomaly detection."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize GCN anomaly detector.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super(GCNAnomalyDetector, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Reconstruction layers
        self.reconstruction = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Anomaly scoring layer
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass through the GCN."""
        # GCN layers
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Global pooling if batch is provided
        if batch is not None:
            h = global_mean_pool(h, batch)
        
        # Reconstruction
        x_reconstructed = self.reconstruction(h)
        
        # Anomaly score
        anomaly_score = self.anomaly_scorer(h)
        
        return h, x_reconstructed, anomaly_score


class GATAnomalyDetector(nn.Module):
    """Graph Attention Network for anomaly detection."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2
    ):
        """
        Initialize GAT anomaly detector.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super(GATAnomalyDetector, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout))
        
        # Reconstruction layers
        self.reconstruction = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Anomaly scoring layer
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass through the GAT."""
        # GAT layers
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Global pooling if batch is provided
        if batch is not None:
            h = global_mean_pool(h, batch)
        
        # Reconstruction
        x_reconstructed = self.reconstruction(h)
        
        # Anomaly score
        anomaly_score = self.anomaly_scorer(h)
        
        return h, x_reconstructed, anomaly_score


class GNNAnomalyDetector:
    """Graph Neural Network-based anomaly detector for financial data."""
    
    def __init__(
        self,
        model_type: str = 'GCN',
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        device: str = 'auto'
    ):
        """
        Initialize GNN anomaly detector.
        
        Args:
            model_type: Type of GNN ('GCN' or 'GAT')
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GNN layers
            heads: Number of attention heads (for GAT)
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
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
        self.graph_data = None
        self.training_history = {'train_loss': [], 'val_loss': []}
    
    def _create_graph_from_correlation(
        self,
        data: pd.DataFrame,
        correlation_threshold: float = 0.3,
        symbols: List[str] = None
    ) -> Data:
        """
        Create graph from correlation matrix.
        
        Args:
            data: DataFrame with time series data
            correlation_threshold: Minimum correlation for edge creation
            symbols: List of symbols to include
            
        Returns:
            PyTorch Geometric Data object
        """
        if symbols is None:
            symbols = data['Symbol'].unique()
        
        # Filter data for specified symbols
        filtered_data = data[data['Symbol'].isin(symbols)].copy()
        
        # Pivot to get price data for each symbol
        price_pivot = filtered_data.pivot(index='Date', columns='Symbol', values='Close')
        price_pivot = price_pivot.ffill().bfill()
        
        # Calculate correlation matrix
        correlation_matrix = price_pivot.corr()
        
        # Create adjacency matrix based on correlation threshold
        adjacency_matrix = (abs(correlation_matrix) > correlation_threshold).astype(int)
        np.fill_diagonal(adjacency_matrix.values, 0)  # Remove self-loops
        
        # Convert to edge list
        edge_list = []
        for i in range(len(adjacency_matrix)):
            for j in range(len(adjacency_matrix)):
                if adjacency_matrix.iloc[i, j] == 1:
                    edge_list.append([i, j])
        
        if not edge_list:
            # If no edges, create a fully connected graph
            n_nodes = len(symbols)
            edge_list = []
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    edge_list.append([i, j])
                    edge_list.append([j, i])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create node features (use latest data for each symbol)
        node_features = []
        for symbol in symbols:
            symbol_data = filtered_data[filtered_data['Symbol'] == symbol].iloc[-1]
            # Use price and volume features
            features = [
                symbol_data['Close'],
                symbol_data['Volume'],
                symbol_data.get('Return_1d', 0),
                symbol_data.get('Volatility_20d', 0)
            ]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        graph_data = Data(x=x, edge_index=edge_index)
        
        return graph_data
    
    def _create_model(self, input_dim: int) -> nn.Module:
        """Create the GNN model."""
        if self.model_type == 'GCN':
            return GCNAnomalyDetector(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
        elif self.model_type == 'GAT':
            return GATAnomalyDetector(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers,
                heads=self.heads,
                dropout=self.dropout
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(
        self,
        data: pd.DataFrame,
        symbols: List[str] = None,
        correlation_threshold: float = 0.3,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> 'GNNAnomalyDetector':
        """
        Fit the GNN model.
        
        Args:
            data: DataFrame with time series data
            symbols: List of symbols to include
            correlation_threshold: Minimum correlation for edge creation
            validation_split: Fraction of data to use for validation
            verbose: Whether to print training progress
            
        Returns:
            Self
        """
        logger.info("Fitting GNN model...")
        
        if symbols is None:
            symbols = data['Symbol'].unique().tolist()
        
        # Store feature names
        self.feature_names = symbols
        
        # Create graph data
        self.graph_data = self._create_graph_from_correlation(
            data, correlation_threshold, symbols
        )
        
        # Scale node features
        x_scaled = self.scaler.fit_transform(self.graph_data.x.numpy())
        self.graph_data.x = torch.tensor(x_scaled, dtype=torch.float)
        
        # Create model
        self.model = self._create_model(self.graph_data.x.shape[1])
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass
            _, x_reconstructed, _ = self.model(
                self.graph_data.x.to(self.device),
                self.graph_data.edge_index.to(self.device)
            )
            
            # Reconstruction loss
            train_loss = criterion(x_reconstructed, self.graph_data.x.to(self.device))
            
            # Backward pass
            train_loss.backward()
            optimizer.step()
            
            # Validation (use same data for simplicity)
            self.model.eval()
            with torch.no_grad():
                _, x_reconstructed_val, _ = self.model(
                    self.graph_data.x.to(self.device),
                    self.graph_data.edge_index.to(self.device)
                )
                val_loss = criterion(x_reconstructed_val, self.graph_data.x.to(self.device))
            
            # Store history
            self.training_history['train_loss'].append(train_loss.item())
            self.training_history['val_loss'].append(val_loss.item())
            
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
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
        
        self.is_fitted = True
        logger.info(f"GNN model fitted with {len(self.feature_names)} nodes")
        return self
    
    def predict(self, data: pd.DataFrame, symbols: List[str] = None) -> np.ndarray:
        """
        Predict anomalies in the data.
        
        Args:
            data: DataFrame with time series data
            symbols: List of symbols to include
            
        Returns:
            Array of predictions (-1 for anomalies, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if symbols is None:
            symbols = self.feature_names
        
        # Create graph data for prediction
        graph_data = self._create_graph_from_correlation(data, symbols=symbols)
        
        # Scale node features
        x_scaled = self.scaler.transform(graph_data.x.numpy())
        graph_data.x = torch.tensor(x_scaled, dtype=torch.float)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            _, x_reconstructed, anomaly_scores = self.model(
                graph_data.x.to(self.device),
                graph_data.edge_index.to(self.device)
            )
        
        # Calculate reconstruction errors
        reconstruction_errors = torch.mean((graph_data.x.to(self.device) - x_reconstructed) ** 2, dim=1)
        reconstruction_errors = reconstruction_errors.cpu().numpy()
        
        # Use threshold to classify anomalies
        threshold = np.percentile(reconstruction_errors, 95)  # Top 5% as anomalies
        predictions = np.where(reconstruction_errors > threshold, -1, 1)
        
        return predictions
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        symbols: List[str] = None,
        threshold: Optional[float] = None,
        percentile: float = 95
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Detect anomalies in the data.
        
        Args:
            data: DataFrame with time series data
            symbols: List of symbols to include
            threshold: Custom threshold for anomaly detection
            percentile: Percentile to use for threshold if not provided
            
        Returns:
            Tuple of (predictions, reconstruction_errors, metadata)
        """
        if symbols is None:
            symbols = self.feature_names
        
        # Create graph data for prediction
        graph_data = self._create_graph_from_correlation(data, symbols=symbols)
        
        # Scale node features
        x_scaled = self.scaler.transform(graph_data.x.numpy())
        graph_data.x = torch.tensor(x_scaled, dtype=torch.float)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            _, x_reconstructed, anomaly_scores = self.model(
                graph_data.x.to(self.device),
                graph_data.edge_index.to(self.device)
            )
        
        # Calculate reconstruction errors
        reconstruction_errors = torch.mean((graph_data.x.to(self.device) - x_reconstructed) ** 2, dim=1)
        reconstruction_errors = reconstruction_errors.cpu().numpy()
        
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
            'percentile_used': percentile,
            'symbols': symbols
        }
        
        logger.info(f"Detected {n_anomalies} anomalies ({anomaly_rate:.2%} of data)")
        
        return predictions, reconstruction_errors, metadata
    
    def plot_graph_structure(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot the graph structure.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted or self.graph_data is None:
            raise ValueError("Model must be fitted before plotting graph structure")
        
        # Convert to NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for i, symbol in enumerate(self.feature_names):
            G.add_node(i, label=symbol)
        
        # Add edges
        edge_list = self.graph_data.edge_index.t().numpy()
        for edge in edge_list:
            G.add_edge(edge[0], edge[1])
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
        
        # Draw labels
        labels = {i: symbol for i, symbol in enumerate(self.feature_names)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f'Graph Structure ({self.model_type})')
        ax.axis('off')
        
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
                'model_type': self.model_type,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_layers': self.num_layers,
                'heads': self.heads,
                'dropout': self.dropout
            },
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'graph_data': self.graph_data,
            'training_history': self.training_history,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }
        
        torch.save(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'GNNAnomalyDetector':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self
        """
        model_data = torch.load(filepath, map_location=self.device)
        
        # Restore configuration
        self.model_type = model_data['model_config']['model_type']
        self.hidden_dim = model_data['model_config']['hidden_dim']
        self.output_dim = model_data['model_config']['output_dim']
        self.num_layers = model_data['model_config']['num_layers']
        self.heads = model_data['model_config']['heads']
        self.dropout = model_data['model_config']['dropout']
        self.learning_rate = model_data['learning_rate']
        self.batch_size = model_data['batch_size']
        self.epochs = model_data['epochs']
        
        # Restore scaler and feature names
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.graph_data = model_data['graph_data']
        self.training_history = model_data['training_history']
        
        # Recreate and load model
        self.model = self._create_model(self.graph_data.x.shape[1])
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
            'model_type': f'GNN ({self.model_type})',
            'is_fitted': self.is_fitted,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'heads': self.heads,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': str(self.device),
            'n_nodes': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Generate sample time series data
    data_list = []
    for symbol in symbols:
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        prices = 100 + np.random.randn(n_samples).cumsum()
        volumes = np.random.randint(1000000, 10000000, n_samples)
        
        symbol_data = pd.DataFrame({
            'Date': dates,
            'Symbol': symbol,
            'Open': prices,
            'High': prices + np.random.rand(n_samples) * 2,
            'Low': prices - np.random.rand(n_samples) * 2,
            'Close': prices,
            'Volume': volumes,
            'Return_1d': np.random.randn(n_samples) * 0.02,
            'Volatility_20d': np.random.rand(n_samples) * 0.3
        })
        data_list.append(symbol_data)
    
    data = pd.concat(data_list, ignore_index=True)
    
    # Create and fit model
    detector = GNNAnomalyDetector(
        model_type='GCN',
        hidden_dim=32,
        output_dim=16,
        epochs=50
    )
    detector.fit(data, symbols=symbols, verbose=True)
    
    # Detect anomalies
    predictions, reconstruction_errors, metadata = detector.detect_anomalies(data, symbols=symbols)
    
    print(f"Detected {metadata['n_anomalies']} anomalies")
    print(f"Anomaly rate: {metadata['anomaly_rate']:.2%}")
    
    # Get model info
    model_info = detector.get_model_info()
    print(f"Model info: {model_info}")
