#!/usr/bin/env python3
"""
Example script demonstrating the usage of the energy regression model
"""

import os
import sys
import pandas as pd
import numpy as np

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import EnergyPredictor


def create_sample_data():
    """Create sample test data for demonstration"""
    # Create some sample energy values based on the data range we observed
    sample_energies = [
        -42.14, -34.39, -22.31, -49.17, -51.19, -23.86,
        -29.07, -44.51, -26.34, -23.69, -46.69, -42.78
    ]
    
    # Create a DataFrame
    df = pd.DataFrame({
        'index': range(len(sample_energies)),
        'dxtb_energy': sample_energies
    })
    
    # Save to CSV
    df.to_csv('sample_test_data.csv', index=False)
    print("Created sample_test_data.csv with sample energy values")
    return df


def demo_single_prediction(predictor):
    """Demonstrate single value prediction"""
    print("\n=== Single Value Prediction Demo ===")
    
    test_energy = -35.5
    prediction = predictor.predict_single(test_energy)
    
    print(f"Input dxtb_energy: {test_energy}")
    print(f"Predicted properties: {prediction:.6f}")


def demo_batch_prediction(predictor):
    """Demonstrate batch prediction"""
    print("\n=== Batch Prediction Demo ===")
    
    test_energies = [-42.1, -34.4, -22.3, -49.2, -51.2]
    predictions = predictor.predict_batch(test_energies)
    
    print("Batch prediction results:")
    for energy, pred in zip(test_energies, predictions):
        print(f"  Energy: {energy:6.1f} → Properties: {pred:8.3f}")


def demo_csv_prediction(predictor):
    """Demonstrate CSV file prediction"""
    print("\n=== CSV Prediction Demo ===")
    
    # Create sample data if it doesn't exist
    if not os.path.exists('sample_test_data.csv'):
        create_sample_data()
    
    # Make predictions
    df_with_predictions = predictor.predict_from_csv(
        'sample_test_data.csv', 
        'sample_predictions.csv'
    )
    
    print("CSV prediction results:")
    print(df_with_predictions.head())
    
    print(f"\nPrediction statistics:")
    stats = df_with_predictions['predicted_properties'].describe()
    for stat, value in stats.items():
        print(f"  {stat}: {value:.3f}")


def main():
    """Main demonstration function"""
    print("Energy Regression Model Demo")
    print("=" * 40)
    
    # Check if model exists
    model_dir = './models'
    model_files = [
        'energy_regressor.pth',
        'scaler_X.pkl', 
        'scaler_y.pkl'
    ]
    
    missing_files = []
    for file in model_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print("ERROR: The following model files are missing:")
        for file in missing_files:
            print(f"  - {os.path.join(model_dir, file)}")
        print("\nPlease train the model first by running:")
        print("python train.py --data_path ../dataset/competition_round2/dxtb_energies.csv")
        return
    
    # Initialize predictor
    try:
        predictor = EnergyPredictor(
            model_path=os.path.join(model_dir, 'energy_regressor.pth'),
            scaler_X_path=os.path.join(model_dir, 'scaler_X.pkl'),
            scaler_y_path=os.path.join(model_dir, 'scaler_y.pkl')
        )
        
        # Run demonstrations
        demo_single_prediction(predictor)
        demo_batch_prediction(predictor)
        demo_csv_prediction(predictor)
        
        print("\n=== Demo Complete ===")
        print("Files created:")
        print("  - sample_test_data.csv (test input)")
        print("  - sample_predictions.csv (predictions output)")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model has been trained and saved properly.")


if __name__ == "__main__":
    main()