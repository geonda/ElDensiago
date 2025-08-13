#!/usr/bin/env python3
"""
Full Pipeline Example for Charge3Net Core Components

This script demonstrates a complete pipeline from CHGCAR input to prediction output.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from guess.charge3net_core.src.predictor import load_predictor
from convert_chgcar_to_pkl import ChgcarToPklConverter
from convert_pkl_to_chgcar import PklToChgcarConverter


def run_full_pipeline():
    """Run the complete pipeline: CHGCAR → Prediction → CHGCAR."""
    print("=== Charge3Net Full Pipeline ===")
    print("CHGCAR → Pickle → Prediction → CHGCAR")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("./predictions")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Convert CHGCAR to pickle format
        print("Step 1: Converting CHGCAR to pickle format...")
        chgcar_to_pkl = ChgcarToPklConverter()
        chgcar_to_pkl.convert_file(
            chgcar_file="./data/sample_inputs/si.CHGCAR",
            output_file="./data/sample_inputs/si_pipeline",
            overwrite=True,
            spin=False
        )
        print("✓ CHGCAR to pickle conversion completed")
        
        # Step 2: Load predictor
        print("\nStep 2: Loading predictor...")
        predictor = load_predictor(
            checkpoint_path="./models/charge3net_mp.pt",
            config_path="./configs/charge3net/test_chgcar_inputs.yaml",
            device='cpu'
        )
        print("✓ Predictor loaded successfully")
        
        # Step 3: Make prediction using full datamodule
        print("\nStep 3: Making prediction with full datamodule...")
        result = predictor.predict_single(
            density_file="./data/sample_inputs/si_pipeline.npy",
            atoms_file="./data/sample_inputs/si_pipeline_atoms.pkl",
            output_file="./predictions/si_pipeline_prediction.npy",
            return_cube=True
        )
        
        print(f"✓ Prediction completed:")
        print(f"  - Filename: {result['filename']}")
        print(f"  - Grid shape: {result['grid_shape']}")
        print(f"  - Prediction time: {result['prediction_time']:.3f}s")
        if 'note' in result:
            print(f"  - Note: {result['note']}")
        
        # Step 4: Convert prediction back to CHGCAR
        print("\nStep 4: Converting prediction to CHGCAR format...")
        pkl_to_chgcar = PklToChgcarConverter()
        pkl_to_chgcar.convert_and_save(
            density_file="./predictions/si_pipeline_prediction.npy",
            atoms_file="./data/sample_inputs/si_pipeline_atoms.pkl",
            output_file="./predictions/Si_pipeline_prediction.CHGCAR"
        )
        print("✓ Prediction converted to CHGCAR format")
        
        # Step 5: Summary
        print("\n" + "=" * 50)
        print("🎉 Full pipeline completed successfully!")
        print("\nOutput files:")
        print(f"  - Pickle files: ./data/sample_inputs/si_pipeline.npy/.pkl")
        print(f"  - Prediction: ./predictions/si_pipeline_prediction.npy")
        print(f"  - CHGCAR: ./predictions/Si_pipeline_prediction.CHGCAR")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_batch_pipeline():
    """Run batch prediction pipeline."""
    print("\n=== Batch Prediction Pipeline ===")
    
    try:
        # Load predictor
        print("Loading predictor for batch processing...")
        predictor = load_predictor(
            checkpoint_path="./models/charge3net_mp.pt",
            config_path="./configs/charge3net/test_chgcar_inputs.yaml",
            device='cpu'
        )
        
        # Setup datamodule and run batch prediction
        print("Running batch prediction...")
        results = predictor.predict_batch(
            input_dir="./data/sample_inputs/",
            output_dir="./predictions/batch_predictions/",
            save_cubes=True
        )
        
        print(f"✓ Batch prediction completed:")
        print(f"  - Number of materials: {results['num_materials']}")
        print(f"  - Output directory: {results['output_dir']}")
        if 'note' in results:
            print(f"  - Note: {results['note']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch pipeline failed: {e}")
        return False


def main():
    """Run the full pipeline."""
    print("Charge3Net Core Components - Full Pipeline")
    print("=" * 60)
    
    # Run full pipeline
    pipeline_success = run_full_pipeline()
    
    # Run batch pipeline
    batch_success = run_batch_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Summary:")
    print(f"  Full Pipeline: {'✓ PASS' if pipeline_success else '✗ FAIL'}")
    print(f"  Batch Pipeline: {'✓ PASS' if batch_success else '✗ FAIL'}")
    
    if pipeline_success and batch_success:
        print("\n🎉 All pipelines completed successfully!")
        print("The core components are ready for production use.")
    else:
        print("\n⚠️  Some pipelines failed. Check the error messages above.")


if __name__ == "__main__":
    main()
