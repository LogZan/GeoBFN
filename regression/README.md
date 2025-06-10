# Energy Regression Model

This package provides a deep learning solution for predicting molecular properties from dxtb energy values.

## Overview

The model takes `dxtb_energy` as input and predicts the corresponding `properties` value using a deep neural network. The implementation includes:

- **Deep Neural Network**: Multi-layer perceptron with configurable architecture
- **Data Preprocessing**: Standardization of both features and targets
- **Training Pipeline**: Complete training with validation and early stopping
- **Inference Engine**: Support for single predictions and batch processing

## Files

- `train.py`: Training script for the regression model
- `inference.py`: Inference script for making predictions
- `requirements.txt`: Required Python packages
- `README.md`: This documentation

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a new model using the training data:

```bash
python train.py --data_path ../dataset/competition_round2/dxtb_energies.csv \
                --epochs 200 \
                --batch_size 64 \
                --lr 0.001 \
                --hidden_dims 128 64 32 \
                --dropout 0.2 \
                --save_dir ./models
```

**Training Parameters:**
- `--data_path`: Path to the CSV file containing training data
- `--epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size for training (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--hidden_dims`: Hidden layer dimensions (default: 128 64 32)
- `--dropout`: Dropout rate (default: 0.2)
- `--save_dir`: Directory to save trained model and scalers (default: ./models)

### Inference

#### Single Value Prediction

Predict properties for a single energy value:

```bash
python inference.py --model_dir ./models --single_value -35.5
```

#### Batch Prediction from CSV

Predict properties for multiple values from a CSV file:

```bash
python inference.py --model_dir ./models \
                   --input test_data.csv \
                   --output predictions.csv \
                   --energy_column dxtb_energy
```

**Inference Parameters:**
- `--model_dir`: Directory containing trained model files (default: ./models)
- `--input`: Input CSV file or single energy value
- `--output`: Output CSV file for batch predictions (optional)
- `--energy_column`: Name of energy column in CSV (default: dxtb_energy)
- `--single_value`: Single energy value for prediction

## Model Architecture

The model uses a feedforward neural network with:
- Input layer: 1 neuron (dxtb_energy)
- Hidden layers: Configurable (default: 128 → 64 → 32 neurons)
- Activation: ReLU
- Regularization: Dropout
- Output layer: 1 neuron (properties)

## Data Preprocessing

- **Feature Scaling**: StandardScaler applied to dxtb_energy values
- **Target Scaling**: StandardScaler applied to properties values
- **Train/Test Split**: 80%/20% split with random state 42

## Training Features

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with weight decay (1e-5)
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Based on validation loss
- **Metrics**: MSE, RMSE, MAE, R² Score

## Output Files

After training, the following files are saved in the model directory:

- `energy_regressor.pth`: Trained model weights
- `scaler_X.pkl`: Feature scaler (for dxtb_energy)
- `scaler_y.pkl`: Target scaler (for properties)
- `training_history.pkl`: Training history and configuration
- `training_results.png`: Visualization of training results

## Example Usage in Python

```python
from inference import EnergyPredictor

# Initialize predictor
predictor = EnergyPredictor(
    model_path='./models/energy_regressor.pth',
    scaler_X_path='./models/scaler_X.pkl',
    scaler_y_path='./models/scaler_y.pkl'
)

# Single prediction
result = predictor.predict_single(-35.5)
print(f"Predicted properties: {result}")

# Batch prediction
energies = [-35.5, -42.1, -28.3]
results = predictor.predict_batch(energies)
print(f"Predicted properties: {results}")

# CSV prediction
df_with_predictions = predictor.predict_from_csv('input.csv', 'output.csv')
```

## Performance Monitoring

The training script provides:
- Real-time loss monitoring
- Validation metrics (MSE, RMSE, MAE, R²)
- Training curves visualization
- Residual analysis plots

## Data Format

Expected CSV format:
```csv
index,properties,dxtb_energy
0,35.551597595214844,-42.14299493815474
1,15.802806854248047,-34.38929348447287
...
```

For inference-only CSV files, only the `dxtb_energy` column is required.

## Tips for Better Performance

1. **Data Quality**: Ensure clean, consistent data
2. **Hyperparameter Tuning**: Experiment with different architectures
3. **Regularization**: Adjust dropout rate to prevent overfitting
4. **Learning Rate**: Use learning rate scheduling for better convergence
5. **Validation**: Monitor validation metrics to detect overfitting

## Troubleshooting

- **CUDA Issues**: The model automatically detects and uses GPU if available
- **Memory Issues**: Reduce batch size if encountering out-of-memory errors
- **Poor Performance**: Try different network architectures or learning rates
- **File Not Found**: Ensure all required model files are in the specified directory