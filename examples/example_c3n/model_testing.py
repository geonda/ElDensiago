#!/usr/bin/env python3
"""
Model Testing Example for Charge3Net Core Components

This script tests different models and configurations to ensure they work correctly.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from guess.charge3net_core.src.simple_predictor import load_simple_predictor
from guess.charge3net_core.src.predictor import load_predictor


def test_model_loading():
    """Test loading different models."""
    print("=== Model Loading Tests ===")
    
    model_paths = [
        "./models/charge3net_mp.pt",
        "./models/charge3net_nmc.pt", 
        "./models/charge3net_qm9.pt"
    ]
    
    config_path = "./configs/charge3net/test_chgcar_inputs.yaml"
    
    results = {}
    
    for model_path in model_paths:
        model_name = Path(model_path).name
        print(f"\nTesting model: {model_name}")
        
        if Path(model_path).exists():
            try:
                # Test simple predictor
                predictor = load_simple_predictor(
                    checkpoint_path=model_path,
                    config_path=config_path,
                    device='cpu'
                )
                
                model_info = predictor.get_model_info()
                print(f"  ✓ Loaded successfully on {model_info['device']}")
                print(f"  - Parameters: {model_info['num_parameters']:,}")
                
                # Test model functionality
                if predictor.test_model_loading():
                    print("  ✓ Model test passed")
                    results[model_name] = "PASS"
                else:
                    print("  ✗ Model test failed")
                    results[model_name] = "FAIL"
                    
            except Exception as e:
                print(f"  ✗ Failed to load: {e}")
                results[model_name] = "FAIL"
        else:
            print(f"  ✗ Model file not found")
            results[model_name] = "NOT_FOUND"
    
    return results


def test_configurations():
    """Test different configuration options."""
    print("\n=== Configuration Tests ===")
    
    configs = [
        "./configs/charge3net/test_chgcar_inputs.yaml",
        "./configs/charge3net/model/e3_density.yaml"
    ]
    
    results = {}
    
    for config_path in configs:
        config_name = Path(config_path).name
        print(f"\nTesting config: {config_name}")
        
        if Path(config_path).exists():
            try:
                # Test with simple predictor
                predictor = load_simple_predictor(
                    checkpoint_path="./models/charge3net_mp.pt",
                    config_path=config_path,
                    device='cpu'
                )
                
                model_info = predictor.get_model_info()
                print(f"  ✓ Configuration loaded successfully")
                print(f"  - Model type: {model_info['model_type']}")
                
                results[config_name] = "PASS"
                
            except Exception as e:
                print(f"  ✗ Failed to load config: {e}")
                results[config_name] = "FAIL"
        else:
            print(f"  ✗ Config file not found")
            results[config_name] = "NOT_FOUND"
    
    return results


def test_devices():
    """Test different device configurations."""
    print("\n=== Device Tests ===")
    
    devices = ['cpu']
    
    # Add MPS if available
    import torch
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices.append('mps')
    
    # Add CUDA if available
    if torch.cuda.is_available():
        devices.append('cuda')
    
    results = {}
    
    for device in devices:
        print(f"\nTesting device: {device}")
        
        try:
            predictor = load_simple_predictor(
                checkpoint_path="./models/charge3net_mp.pt",
                config_path="./configs/charge3net/test_chgcar_inputs.yaml",
                device=device
            )
            
            model_info = predictor.get_model_info()
            print(f"  ✓ Model loaded on {model_info['device']}")
            
            if predictor.test_model_loading():
                print("  ✓ Model test passed")
                results[device] = "PASS"
            else:
                print("  ✗ Model test failed")
                results[device] = "FAIL"
                
        except Exception as e:
            print(f"  ✗ Failed on {device}: {e}")
            results[device] = "FAIL"
    
    return results


def test_datamodule_integration():
    """Test datamodule integration."""
    print("\n=== Datamodule Integration Tests ===")
    
    try:
        # Load full predictor
        predictor = load_predictor(
            checkpoint_path="./models/charge3net_mp.pt",
            config_path="./configs/charge3net/test_chgcar_inputs.yaml",
            device='cpu'
        )
        
        print("✓ Full predictor loaded successfully")
        
        # Test datamodule setup
        print("Testing datamodule setup...")
        predictor.setup_datamodule("./data/sample_inputs/")
        print("✓ Datamodule setup completed")
        
        # Test datamodule properties
        test_set = predictor.datamodule.test_set
        print(f"✓ Test set contains {len(test_set)} items")
        
        return "PASS"
        
    except Exception as e:
        print(f"✗ Datamodule integration failed: {e}")
        return "FAIL"


def main():
    """Run all tests."""
    print("Charge3Net Core Components - Model Testing")
    print("=" * 60)
    
    # Test model loading
    model_results = test_model_loading()
    
    # Test configurations
    config_results = test_configurations()
    
    # Test devices
    device_results = test_devices()
    
    # Test datamodule integration
    datamodule_result = test_datamodule_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    
    print("\nModel Loading:")
    for model, result in model_results.items():
        print(f"  {model}: {'✓ PASS' if result == 'PASS' else '✗ FAIL' if result == 'FAIL' else '⚠️ NOT_FOUND'}")
    
    print("\nConfigurations:")
    for config, result in config_results.items():
        print(f"  {config}: {'✓ PASS' if result == 'PASS' else '✗ FAIL' if result == 'FAIL' else '⚠️ NOT_FOUND'}")
    
    print("\nDevices:")
    for device, result in device_results.items():
        print(f"  {device}: {'✓ PASS' if result == 'PASS' else '✗ FAIL'}")
    
    print(f"\nDatamodule Integration: {'✓ PASS' if datamodule_result == 'PASS' else '✗ FAIL'}")
    
    # Overall result
    all_passed = (
        all(r == 'PASS' for r in model_results.values()) and
        all(r == 'PASS' for r in config_results.values()) and
        all(r == 'PASS' for r in device_results.values()) and
        datamodule_result == 'PASS'
    )
    
    if all_passed:
        print("\n🎉 All tests passed! The models are ready for use.")
    else:
        print("\n⚠️  Some tests failed. Check the results above.")


if __name__ == "__main__":
    main()
