#!/usr/bin/env python3
"""
Real Data Loading Example for Charge3Net Core Components

This script demonstrates real data loading and prediction using the actual datamodule
and model architecture with full graph construction.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from guess.charge3net_core.src.real_predictor import load_real_predictor
from convert_chgcar_to_pkl import ChgcarToPklConverter
from convert_pkl_to_chgcar import PklToChgcarConverter


def test_real_single_prediction():
    """Test real single prediction with actual data loading."""
    print("=== Real Single Prediction Test ===")
    
    try:
        # Load the real predictor
        print("Loading real predictor...")
        predictor = load_real_predictor(
            checkpoint_path="./models/charge3net_mp.pt",
            config_path="./configs/charge3net/test_chgcar_inputs.yaml",
            device='cpu'
        )
        
        model_info = predictor.get_model_info()
        print(f"✓ Real predictor loaded successfully:")
        print(f"  - Device: {model_info['device']}")
        print(f"  - Model type: {model_info['model_type']}")
        print(f"  - Parameters: {model_info['num_parameters']:,}")
        
        # Make real prediction
        print("\nMaking real prediction with actual data loading...")
        result = predictor.predict_single_real(
            density_file="./data/sample_inputs/si.npy",
            atoms_file="./data/sample_inputs/si_atoms.pkl",
            output_file="./predictions/si_real_prediction.npy",
            return_cube=True
        )
        
        print(f"✓ Real prediction completed:")
        print(f"  - Filename: {result['filename']}")
        print(f"  - Grid shape: {result['grid_shape']}")
        print(f"  - Prediction time: {result['prediction_time']:.3f}s")
        print(f"  - Note: {result['note']}")
        
        if 'prediction_cube' in result:
            print(f"  - Prediction cube shape: {result['prediction_cube'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Real single prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_batch_prediction():
    """Test real batch prediction with actual data loading."""
    print("\n=== Real Batch Prediction Test ===")
    
    try:
        # Load the real predictor
        print("Loading real predictor for batch processing...")
        predictor = load_real_predictor(
            checkpoint_path="./models/charge3net_mp.pt",
            config_path="./configs/charge3net/test_chgcar_inputs.yaml",
            device='cpu'
        )
        
        # Make real batch predictions
        print("Making real batch predictions...")
        results = predictor.predict_batch_real(
            input_dir="./data/sample_inputs/",
            output_dir="./predictions/real_batch_predictions/",
            save_cubes=True
        )
        
        print(f"✓ Real batch prediction completed:")
        print(f"  - Number of materials: {results['num_materials']}")
        print(f"  - Output directory: {results['output_dir']}")
        print(f"  - Note: {results['note']}")
        
        # Show individual results
        for i, pred in enumerate(results['predictions']):
            if 'error' in pred:
                print(f"  - Material {i+1}: {pred['filename']} - ERROR: {pred['error']}")
            else:
                print(f"  - Material {i+1}: {pred['filename']} - ✓ Success")
        
        return True
        
    except Exception as e:
        print(f"✗ Real batch prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_pipeline():
    """Test complete real pipeline from CHGCAR to prediction."""
    print("\n=== Real Pipeline Test ===")
    print("CHGCAR → Pickle → Real Prediction → CHGCAR")
    
    try:
        # Step 1: Convert CHGCAR to pickle format
        print("Step 1: Converting CHGCAR to pickle format...")
        chgcar_to_pkl = ChgcarToPklConverter()
        chgcar_to_pkl.convert_file(
            chgcar_file="./data/sample_inputs/si.CHGCAR",
            output_file="./data/sample_inputs/si_real_pipeline",
            overwrite=True,
            spin=False
        )
        print("✓ CHGCAR to pickle conversion completed")
        
        # Step 2: Load real predictor
        print("\nStep 2: Loading real predictor...")
        predictor = load_real_predictor(
            checkpoint_path="./models/charge3net_mp.pt",
            config_path="./configs/charge3net/test_chgcar_inputs.yaml"
        )
        print("✓ Real predictor loaded successfully")
        
        # Step 3: Make real prediction
        print("\nStep 3: Making real prediction with actual data loading...")
        result = predictor.predict_single_real(
            density_file="./data/sample_inputs/si_real_pipeline.npy",
            atoms_file="./data/sample_inputs/si_real_pipeline_atoms.pkl",
            output_file="./predictions/si_real_pipeline_prediction.npy",
            return_cube=True
        )
        
        print(f"✓ Real prediction completed in {result['prediction_time']:.3f}s")
        print(f"  - Note: {result['note']}")
        
        # Step 4: Convert prediction back to CHGCAR
        print("\nStep 4: Converting real prediction to CHGCAR format...")
        pkl_to_chgcar = PklToChgcarConverter()
        pkl_to_chgcar.convert_and_save(
            density_file="./predictions/si_real_pipeline_prediction.npy",
            atoms_file="./data/sample_inputs/si_real_pipeline_atoms.pkl",
            output_file="./predictions/Si_real_pipeline_prediction.CHGCAR"
        )
        print("✓ Real prediction converted to CHGCAR format")
        
        # Summary
        print("\n" + "=" * 50)
        print("🎉 Real pipeline completed successfully!")
        print("\nOutput files:")
        print(f"  - Pickle files: ./data/sample_inputs/si_real_pipeline.npy/.pkl")
        print(f"  - Prediction: ./predictions/si_real_pipeline_prediction.npy")
        print(f"  - CHGCAR: ./predictions/Si_real_pipeline_prediction.CHGCAR")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Real pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading_details():
    """Test and show details about data loading process."""
    print("\n=== Data Loading Details Test ===")
    
    try:
        # Load predictor
        predictor = load_real_predictor(
            checkpoint_path="./models/charge3net_mp.pt",
            config_path="./configs/charge3net/test_chgcar_inputs.yaml"
        )
        
        # Setup datamodule
        print("Setting up datamodule...")
        predictor.setup_datamodule("./data/sample_inputs/")
        
        # Show datamodule details
        print("✓ Datamodule setup completed")
        print(f"  - Test set size: {len(predictor.datamodule.test_set)}")
        
        # Try to get first item to show data structure
        if len(predictor.datamodule.test_set) > 0:
            try:
                first_item = predictor.datamodule.test_set[0]
                print(f"  - First item keys: {list(first_item.keys())}")
                
                # Show some key information
                if 'grid_shape' in first_item:
                    print(f"  - Grid shape: {first_item['grid_shape']}")
                if 'num_probes' in first_item:
                    print(f"  - Number of probes: {first_item['num_probes']}")
                if 'num_nodes' in first_item:
                    print(f"  - Number of nodes: {first_item['num_nodes']}")
                
            except Exception as e:
                print(f"  - Could not access first item: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading details test failed: {e}")
        return False


def main():
    """Run all real data loading tests."""
    print("Charge3Net Core Components - Real Data Loading Test")
    print("=" * 70)
    
    # Test real single prediction
    single_success = test_real_single_prediction()
    
    # Test real batch prediction
    batch_success = test_real_batch_prediction()
    
    # Test real pipeline
    pipeline_success = test_real_pipeline()
    
    # Test data loading details
    details_success = test_data_loading_details()
    
    # Summary
    print("\n" + "=" * 70)
    print("Real Data Loading Test Summary:")
    print(f"  Single Prediction: {'✓ PASS' if single_success else '✗ FAIL'}")
    print(f"  Batch Prediction:  {'✓ PASS' if batch_success else '✗ FAIL'}")
    print(f"  Real Pipeline:     {'✓ PASS' if pipeline_success else '✗ FAIL'}")
    print(f"  Data Details:      {'✓ PASS' if details_success else '✗ FAIL'}")
    
    if all([single_success, batch_success, pipeline_success, details_success]):
        print("\n🎉 All real data loading tests passed!")
        print("The real predictor is working with actual data loading and graph construction.")
    else:
        print("\n⚠️  Some real data loading tests failed.")
        print("Check the error messages above for details.")


if __name__ == "__main__":
    main()
