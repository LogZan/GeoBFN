import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import argparse
import os
from typing import Union, List


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


class EnergyPredictor:
    """Wrapper class for energy prediction"""
    
    def __init__(self, model_path: str, scaler_X_path: str, scaler_y_path: str, 
                 hidden_dims: List[int] = [128, 64, 32], dropout_rate: float = 0.2):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the saved model weights
            scaler_X_path: Path to the feature scaler
            scaler_y_path: Path to the target scaler
            hidden_dims: Hidden layer dimensions (must match training config)
            dropout_rate: Dropout rate (must match training config)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load scalers
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        
        # Initialize and load model
        self.model = EnergyRegressor(
            input_dim=1,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def predict_single(self, dxtb_energy: float) -> float:
        """
        Predict properties for a single dxtb_energy value
        
        Args:
            dxtb_energy: Single energy value
            
        Returns:
            Predicted properties value
        """
        # Prepare input
        X = np.array([[dxtb_energy]])
        X_scaled = self.scaler_X.transform(X)
        
        # Make prediction
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_pred_scaled = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)[0, 0]
        
        return y_pred
    
    def predict_batch(self, dxtb_energies: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Predict properties for multiple dxtb_energy values
        
        Args:
            dxtb_energies: Array or list of energy values
            
        Returns:
            Array of predicted properties values
        """
        # Prepare input
        if isinstance(dxtb_energies, list):
            dxtb_energies = np.array(dxtb_energies)
        
        X = dxtb_energies.reshape(-1, 1)
        X_scaled = self.scaler_X.transform(X)
        
        # Make predictions
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_pred_scaled = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        return y_pred
    
    def predict_from_csv(self, input_csv: str, output_csv: str = None, 
                        energy_column: str = 'dxtb_energy'):
        """
        Predict properties from a CSV file
        
        Args:
            input_csv: Path to input CSV file
            output_csv: Path to save predictions (optional)
            energy_column: Name of the energy column in CSV
            
        Returns:
            DataFrame with predictions
        """
        # Load data
        df = pd.read_csv(input_csv)
        
        if energy_column not in df.columns:
            raise ValueError(f"Column '{energy_column}' not found in CSV")
        
        # Make predictions
        energies = df[energy_column].values
        predictions = self.predict_batch(energies)
        
        # Add predictions to dataframe
        df['predicted_properties'] = predictions
        
        # Save if output path provided
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"Predictions saved to {output_csv}")
        
        return df


def load_model_config(model_dir: str):
    """Load model configuration from training history"""
    history_path = os.path.join(model_dir, 'training_history.pkl')
    if os.path.exists(history_path):
        history = joblib.load(history_path)
        return history['model_config']
    else:
        # Default configuration
        return {
            'hidden_dims': [128, 64, 32],
            'dropout_rate': 0.2
        }


def main():
    parser = argparse.ArgumentParser(description='Energy regression inference')
    parser.add_argument('--model_dir', type=str, default='./models',
                       help='Directory containing saved model and scalers')
    parser.add_argument('--input', type=str, help='Input CSV file or single energy value')
    parser.add_argument('--output', type=str, help='Output CSV file (for CSV input)')
    parser.add_argument('--energy_column', type=str, default='dxtb_energy',
                       help='Name of energy column in CSV')
    parser.add_argument('--single_value', type=float, help='Single energy value to predict')
    
    args = parser.parse_args()
    
    # Paths to model components
    model_path = os.path.join(args.model_dir, 'energy_regressor.pth')
    scaler_X_path = os.path.join(args.model_dir, 'scaler_X.pkl')
    scaler_y_path = os.path.join(args.model_dir, 'scaler_y.pkl')
    
    # Check if files exist
    for path in [model_path, scaler_X_path, scaler_y_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
    
    # Load model configuration
    config = load_model_config(args.model_dir)
    
    # Initialize predictor
    predictor = EnergyPredictor(
        model_path=model_path,
        scaler_X_path=scaler_X_path,
        scaler_y_path=scaler_y_path,
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate']
    )
    
    # Make predictions
    if args.single_value is not None:
        # Single value prediction
        prediction = predictor.predict_single(args.single_value)
        print(f"Input energy: {args.single_value}")
        print(f"Predicted properties: {prediction:.6f}")
        
    elif args.input:
        if args.input.endswith('.csv'):
            # CSV file prediction
            df_with_predictions = predictor.predict_from_csv(
                args.input, args.output, args.energy_column
            )
            print(f"Predictions completed for {len(df_with_predictions)} samples")
            print(f"Prediction statistics:")
            print(f"  Mean: {df_with_predictions['predicted_properties'].mean():.6f}")
            print(f"  Std:  {df_with_predictions['predicted_properties'].std():.6f}")
            print(f"  Min:  {df_with_predictions['predicted_properties'].min():.6f}")
            print(f"  Max:  {df_with_predictions['predicted_properties'].max():.6f}")
        else:
            # Try to parse as single value
            try:
                energy_value = float(args.input)
                prediction = predictor.predict_single(energy_value)
                print(f"Input energy: {energy_value}")
                print(f"Predicted properties: {prediction:.6f}")
            except ValueError:
                print("Error: Input must be a CSV file or a numeric value")
    else:
        print("Error: Please provide either --single_value or --input argument")


if __name__ == "__main__":
    main()