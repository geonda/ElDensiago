#!/usr/bin/env python3
"""
Example usage of the ChargeDensityPredictor class.

This script demonstrates how to use the ChargeDensityPredictor class to make
charge density predictions using trained Charge3Net models with the full datamodule.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import the predictor
sys.path.append(str(Path(__file__).parent / "src"))

from guess.charge3net_core.src.predictor import ChargeDensityPredictor, load_predictor
from guess.charge3net_core.src.simple_predictor import SimpleChargeDensityPredictor, load_simple_predictor
from convert_chgcar_to_pkl import ChgcarToPklConverter
from convert_pkl_to_chgcar import PklToChgcarConverter


def main():
    """Example usage of the ChargeDensityPredictor class."""
    
    print("=== ChargeDensityPredictor Example ===")
    
    # Example 1: Basic predictor usage with simple predictor
    print("\n--- Example 1: Basic Predictor Usage (Simple) ---")
    try:
        # Load the simple predictor
        predictor = load_simple_predictor(
            checkpoint_path="./models/charge3net_mp.pt",
            config_path="configs/charge3net/test_chgcar_inputs.yaml",
            device='cpu'
        )
        
        # Get model information
        model_info = predictor.get_model_info()
        print(f"✓ Model loaded successfully:")
        print(f"  - Device: {model_info['device']}")
        print(f"  - Model type: {model_info['model_type']}")
        print(f"  - Parameters: {model_info['num_parameters']:,}")
        
        # Test model loading
        if predictor.test_model_loading():
            print("✓ Model test passed")
        else:
            print("✗ Model test failed")
        
    except Exception as e:
        print(f"✗ Error loading simple predictor: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 2: Full predictor usage with datamodule
    print("\n--- Example 2: Full Predictor Usage (with Datamodule) ---")
    try:
        # Load the full predictor
        predictor = load_predictor(
            checkpoint_path="./models/charge3net_mp.pt",
            config_path="configs/charge3net/test_chgcar_inputs.yaml",
            device='cpu',
            max_predict_batch_probes=10000
        )
        
        # Get model information
        model_info = predictor.get_model_info()
        print(f"✓ Full predictor loaded successfully:")
        print(f"  - Device: {model_info['device']}")
        print(f"  - Model type: {model_info['model_type']}")
        print(f"  - Parameters: {model_info['num_parameters']:,}")
        
        # Example: Make a single prediction using the full datamodule
        print("\nMaking single prediction with full datamodule...")
        result = predictor.predict_single(
            density_file="./charge3net_inputs_test/si.npy",
            atoms_file="./charge3net_inputs_test/si_atoms.pkl",
            output_file="./predictions/si_prediction_full.npy",
            return_cube=True
        )
        
        print(f"✓ Prediction completed:")
        print(f"  - Filename: {result['filename']}")
        print(f"  - Grid shape: {result['grid_shape']}")
        print(f"  - Prediction time: {result['prediction_time']:.3f}s")
        print(f"  - Prediction cube shape: {result['prediction_cube'].shape}")
        
    except Exception as e:
        print(f"✗ Error with full predictor: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 3: Batch prediction with datamodule
    print("\n--- Example 3: Batch Prediction with Datamodule ---")
    try:
        # Load the full predictor
        predictor = load_predictor(
            checkpoint_path="./models/charge3net_mp.pt",
            config_path="configs/charge3net/test_chgcar_inputs.yaml",
            device='cpu'
        )
        
        # Make batch predictions using the full datamodule
        print("Making batch predictions...")
        results = predictor.predict_batch(
            input_dir="./charge3net_inputs_test/",
            output_dir="./predictions/batch_predictions/",
            save_cubes=True
        )
        
        print(f"✓ Batch prediction completed:")
        print(f"  - Number of materials: {results['num_materials']}")
        print(f"  - Output directory: {results['output_dir']}")
        
    except Exception as e:
        print(f"✗ Error with batch prediction: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 4: Complete pipeline (CHGCAR → Prediction → CHGCAR)
    print("\n--- Example 4: Complete Pipeline ---")
    try:
        # Step 1: Convert CHGCAR to pickle format
        print("Step 1: Converting CHGCAR to pickle format...")
        chgcar_to_pkl = ChgcarToPklConverter()
        chgcar_to_pkl.convert_file(
            chgcar_file="./charge3net_inputs_test/si.CHGCAR",
            output_file="si_pipeline",
            overwrite=True,
            spin=False
        )
        
        # Step 2: Load predictor and make prediction
        print("Step 2: Loading predictor...")
        predictor = load_predictor(
            checkpoint_path="./models/charge3net_mp.pt",
            config_path="configs/charge3net/test_chgcar_inputs.yaml"
        )
        
        print("✓ Predictor loaded successfully")
        
        # Step 3: Make prediction using full datamodule
        print("Step 3: Making prediction with full datamodule...")
        result = predictor.predict_single(
            density_file="./si_pipeline.npy",
            atoms_file="./si_pipeline_atoms.pkl",
            output_file="./predictions/si_pipeline_prediction.npy",
            return_cube=True
        )
        
        print(f"✓ Prediction completed in {result['prediction_time']:.3f}s")
        
        # Step 4: Convert prediction back to CHGCAR
        print("Step 4: Converting prediction to CHGCAR format...")
        pkl_to_chgcar = PklToChgcarConverter()
        pkl_to_chgcar.convert_and_save(
            density_file="./predictions/si_pipeline_prediction.npy",
            atoms_file="./si_pipeline_atoms.pkl",
            output_file="./predictions/Si_pipeline_prediction.CHGCAR"
        )
        
        print("✓ Complete pipeline finished successfully!")
        
    except Exception as e:
        print(f"✗ Error in complete pipeline: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 5: Using different models
    print("\n--- Example 5: Using Different Models ---")
    model_paths = [
        "./models/charge3net_mp.pt",
        "./models/charge3net_nmc.pt", 
        "./models/charge3net_qm9.pt"
    ]
    
    for model_path in model_paths:
        if Path(model_path).exists():
            try:
                print(f"Testing model: {Path(model_path).name}")
                model_predictor = load_simple_predictor(
                    checkpoint_path=model_path,
                    config_path="configs/charge3net/test_chgcar_inputs.yaml"
                )
                
                model_info = model_predictor.get_model_info()
                print(f"  ✓ Loaded successfully on {model_info['device']}")
                
            except Exception as e:
                print(f"  ✗ Failed to load: {e}")
        else:
            print(f"Model not found: {model_path}")


def demonstrate_advanced_usage():
    """Demonstrate more advanced usage patterns."""
    
    print("\n=== Advanced Usage Examples ===")
    
    try:
        # Load predictor with custom settings
        predictor = load_predictor(
            checkpoint_path="./models/charge3net_mp.pt",
            config_path="configs/charge3net/test_chgcar_inputs.yaml",
            device="cpu",  # Force CPU usage
            max_predict_batch_probes=5000  # Smaller batch size for memory-constrained systems
        )
        
        print("✓ Predictor loaded with custom settings")
        
        # Example of getting model information
        model_info = predictor.get_model_info()
        print(f"Model details:")
        print(f"  - Checkpoint: {model_info['checkpoint_path']}")
        print(f"  - Device: {model_info['device']}")
        print(f"  - Trainable parameters: {model_info['trainable_parameters']:,}")
        
        # Example of setting up datamodule manually
        print("\nSetting up datamodule manually...")
        predictor.setup_datamodule("./charge3net_inputs_test/")
        print("✓ Datamodule setup completed")
        
    except Exception as e:
        print(f"✗ Error in advanced usage: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    demonstrate_advanced_usage()
    
    print("\n=== Example Completed ===")
    print("Check the ./predictions/ directory for output files.")
