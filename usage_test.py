#!/usr/bin/env python3
"""
Example usage of ElDensiago package with proper package data handling.
"""

from guess import MlDensity, get_package_data_path

def main():
    print("=== ElDensiago Package Data Example ===\n")
    
    # List available models
    print("Available pretrained models:")
    models = MlDensity.list_available_models()
    if models:
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
    else:
        print("  No models found!")
    
    print("\n" + "="*50 + "\n")
    
    # Example of accessing package data
    print("Example of accessing package data:")
    try:
        # Get path to a specific model directory
        model_path = get_package_data_path("pretrained_models/nmc_schnet")
        print(f"Model path: {model_path}")
        
        # Get path to arguments file
        args_path = get_package_data_path("pretrained_models/nmc_schnet/arguments.json")
        print(f"Arguments file: {args_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example of creating a predictor
    print("Creating density predictor:")
    try:
        predictor = MlDensity(
            model='nmc_schnet',
            device='cpu',
            grid_step=0.5,
            probe_count=10,
            force_pbc=True
        )
        print("✓ Predictor created successfully!")
        print(f"  Model: {predictor.model_name}")
        print(f"  Device: {predictor.device}")
        print(f"  Grid step: {predictor.grid_step}")
        
    except Exception as e:
        print(f"✗ Error creating predictor: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example of running a prediction (if example file exists)
    print("Running prediction example:")
    try:
        import os
        example_file = "example/batio3.xyz"
        if os.path.exists(example_file):
            print(f"Predicting density for {example_file}...")
            predictor.predict(example_file)
            print("✓ Prediction completed!")
        else:
            print(f"Example file not found: {example_file}")
            print("Please ensure the example file exists.")
            
    except Exception as e:
        print(f"✗ Error during prediction: {e}")

if __name__ == "__main__":
    main() 