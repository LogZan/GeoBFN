import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import os
import argparse
from tqdm import tqdm


class EnergyDataset(Dataset):
    """Custom Dataset for energy regression"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EnergyRegressor(nn.Module):
    """Deep neural network for energy regression"""
    def __init__(self, input_dim=1, hidden_dims=[128, 64, 32], dropout_rate=0.2):
        super(EnergyRegressor, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def load_and_preprocess_data(data_path):
    """Load and preprocess the data"""
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Extract features and target
    X = df['dxtb_energy'].values.reshape(-1, 1)
    y = df['properties'].values
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Data range - X: [{X.min():.3f}, {X.max():.3f}], y: [{y.min():.3f}, {y.max():.3f}]")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Standardize target (optional, can improve training stability)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
            y_train, y_test, scaler_X, scaler_y)


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """Train the model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_batches += 1
        
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def evaluate_model(model, X_test, y_test, scaler_y, device):
    """Evaluate the model"""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy().flatten()
    
    # Inverse transform predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nEvaluation Results:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    return y_pred, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def plot_results(y_true, y_pred, train_losses, val_losses, save_dir):
    """Plot training curves and predictions"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Predictions vs True values
    plt.subplot(1, 3, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.grid(True)
    
    # Plot 3: Residuals
    plt.subplot(1, 3, 3)
    residuals = y_pred - y_true
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train energy regression model')
    parser.add_argument('--data_path', type=str, 
                       default='../dataset/competition_round2/dxtb_energies.csv',
                       help='Path to the data file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128, 64, 32],
                       help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    (X_train, X_test, y_train_scaled, y_test_scaled,
     y_train, y_test, scaler_X, scaler_y) = load_and_preprocess_data(args.data_path)
    
    # Create datasets and data loaders
    train_dataset = EnergyDataset(X_train, y_train_scaled)
    test_dataset = EnergyDataset(X_test, y_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = EnergyRegressor(
        input_dim=1,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout
    ).to(device)
    
    print(f"Model architecture: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, args.epochs, args.lr, device
    )
    
    # Evaluate model
    y_pred, metrics = evaluate_model(model, X_test, y_test, scaler_y, device)
    
    # Save model and scalers
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'energy_regressor.pth'))
    joblib.dump(scaler_X, os.path.join(args.save_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(args.save_dir, 'scaler_y.pkl'))
    
    # Save training history and metrics
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metrics': metrics,
        'model_config': {
            'hidden_dims': args.hidden_dims,
            'dropout_rate': args.dropout,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs
        }
    }
    joblib.dump(training_history, os.path.join(args.save_dir, 'training_history.pkl'))
    
    # Plot results
    plot_results(y_test, y_pred, train_losses, val_losses, args.save_dir)
    
    print(f"\nModel and results saved to {args.save_dir}")


if __name__ == "__main__":
    main()