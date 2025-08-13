#!/usr/bin/env python3
"""
Basic Usage Example for Charge3Net Core Components

This script demonstrates the basic usage of the Charge3Net predictor classes
for model loading and testing.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from guess.charge3net_core.src.simple_predictor import load_simple_predictor
from guess.charge3net_core.src.predictor import load_predictor


def test_basic_predictor():
    """Test the basic predictor functionality."""
    print("=== Testing Basic Predictor ===")
    
    try:
        # Load the simple predictor
        predictor = load_simple_predictor(
            checkpoint_path="./models/charge3net_mp.pt",
            config_path="./configs/charge3net/test_chgcar_inputs.yaml",
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
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_full_predictor():
    """Test the full predictor functionality."""
    print("\n=== Testing Full Predictor ===")
    
    try:
        # Load the full predictor
        predictor = load_predictor(
            checkpoint_path="./models/charge3net_mp.pt",
            config_path="./configs/charge3net/test_chgcar_inputs.yaml",
            device='cpu'
        )
        
        # Get model information
        model_info = predictor.get_model_info()
        print(f"✓ Full predictor loaded successfully:")
        print(f"  - Device: {model_info['device']}")
        print(f"  - Model type: {model_info['model_type']}")
        print(f"  - Parameters: {model_info['num_parameters']:,}")
        
        # Test datamodule setup
        print("\nTesting datamodule setup...")
        predictor.setup_datamodule("./data/sample_inputs/")
        print("✓ Datamodule setup completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_file_conversion():
    """Test file conversion functionality."""
    print("\n=== Testing File Conversion ===")
    
    try:
        from convert_chgcar_to_pkl import ChgcarToPklConverter
        from convert_pkl_to_chgcar import PklToChgcarConverter
        
        # Test CHGCAR to pickle conversion
        print("Testing CHGCAR to pickle conversion...")
        converter = ChgcarToPklConverter()
        converter.convert_file(
            chgcar_file="./data/sample_inputs/si.CHGCAR",
            output_file="./data/sample_inputs/si_test",
            overwrite=True
        )
        print("✓ CHGCAR to pickle conversion successful")
        
        # Test pickle to CHGCAR conversion
        print("Testing pickle to CHGCAR conversion...")
        pkl_converter = PklToChgcarConverter()
        pkl_converter.convert_and_save(
            density_file="./data/sample_inputs/si_test.npy",
            atoms_file="./data/sample_inputs/si_test_atoms.pkl",
            output_file="./data/sample_inputs/Si_test_converted.CHGCAR"
        )
        print("✓ Pickle to CHGCAR conversion successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("Charge3Net Core Components - Basic Usage Test")
    print("=" * 50)
    
    # Test basic predictor
    basic_success = test_basic_predictor()
    
    # Test full predictor
    full_success = test_full_predictor()
    
    # Test file conversion
    conversion_success = test_file_conversion()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Basic Predictor: {'✓ PASS' if basic_success else '✗ FAIL'}")
    print(f"  Full Predictor:  {'✓ PASS' if full_success else '✗ FAIL'}")
    print(f"  File Conversion: {'✓ PASS' if conversion_success else '✗ FAIL'}")
    
    if all([basic_success, full_success, conversion_success]):
        print("\n🎉 All tests passed! The core components are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    main()
