#!/usr/bin/env python3
"""
Environment Setup Utility for Charge3Net Core Components

This script helps set up the environment and verify all dependencies are available.
"""

import sys
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} is not compatible (requires 3.8+)")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nChecking dependencies...")
    
    required_packages = [
        'torch',
        'numpy', 
        'ase',
        'pymatgen',
        'omegaconf',
        'hydra'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    return missing_packages


def install_dependencies(packages):
    """Install missing dependencies."""
    if not packages:
        print("✓ All dependencies are already installed")
        return True
    
    print(f"\nInstalling missing packages: {', '.join(packages)}")
    
    try:
        for package in packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install packages: {e}")
        return False


def check_models():
    """Check if model files are available."""
    print("\nChecking model files...")
    
    model_files = [
        "./models/charge3net_mp.pt",
        "./models/charge3net_nmc.pt",
        "./models/charge3net_qm9.pt"
    ]
    
    missing_models = []
    
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"✓ {model_file} found")
        else:
            print(f"✗ {model_file} missing")
            missing_models.append(model_file)
    
    return missing_models


def check_configs():
    """Check if configuration files are available."""
    print("\nChecking configuration files...")
    
    config_files = [
        "./configs/charge3net/test_chgcar_inputs.yaml",
        "./configs/charge3net/model/e3_density.yaml"
    ]
    
    missing_configs = []
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✓ {config_file} found")
        else:
            print(f"✗ {config_file} missing")
            missing_configs.append(config_file)
    
    return missing_configs


def check_data():
    """Check if sample data is available."""
    print("\nChecking sample data...")
    
    data_files = [
        "./data/sample_inputs/si.CHGCAR",
        "./data/sample_inputs/si.npy",
        "./data/sample_inputs/si_atoms.pkl"
    ]
    
    missing_data = []
    
    for data_file in data_files:
        if Path(data_file).exists():
            print(f"✓ {data_file} found")
        else:
            print(f"✗ {data_file} missing")
            missing_data.append(data_file)
    
    return missing_data


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    directories = [
        "./predictions",
        "./predictions/batch_predictions",
        "./logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}")


def test_imports():
    """Test if core components can be imported."""
    print("\nTesting imports...")
    
    try:
        # Add current directory to path
        sys.path.append(str(Path(__file__).parent.parent))
        
        # Test imports
        from guess.charge3net_core.src.simple_predictor import load_simple_predictor
        from guess.charge3net_core.src.predictor import load_predictor
        from convert_chgcar_to_pkl import ChgcarToPklConverter
        from convert_pkl_to_chgcar import PklToChgcarConverter
        
        print("✓ All core components imported successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def main():
    """Run environment setup."""
    print("Charge3Net Core Components - Environment Setup")
    print("=" * 60)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check dependencies
    missing_packages = check_dependencies()
    
    # Install missing packages if any
    if missing_packages:
        install_ok = install_dependencies(missing_packages)
        if not install_ok:
            print("\n⚠️  Some packages could not be installed automatically.")
            print("Please install them manually:")
            for package in missing_packages:
                print(f"  pip install {package}")
    else:
        install_ok = True
    
    # Check models
    missing_models = check_models()
    
    # Check configs
    missing_configs = check_configs()
    
    # Check data
    missing_data = check_data()
    
    # Create directories
    create_directories()
    
    # Test imports
    imports_ok = test_imports()
    
    # Summary
    print("\n" + "=" * 60)
    print("Environment Setup Summary:")
    print(f"  Python Version: {'✓ OK' if python_ok else '✗ FAIL'}")
    print(f"  Dependencies: {'✓ OK' if install_ok else '✗ FAIL'}")
    print(f"  Models: {'✓ OK' if not missing_models else f'⚠️ {len(missing_models)} missing'}")
    print(f"  Configs: {'✓ OK' if not missing_configs else f'⚠️ {len(missing_configs)} missing'}")
    print(f"  Data: {'✓ OK' if not missing_data else f'⚠️ {len(missing_data)} missing'}")
    print(f"  Imports: {'✓ OK' if imports_ok else '✗ FAIL'}")
    
    # Overall status
    all_ok = (
        python_ok and 
        install_ok and 
        imports_ok and
        len(missing_models) == 0 and
        len(missing_configs) == 0 and
        len(missing_data) == 0
    )
    
    if all_ok:
        print("\n🎉 Environment setup completed successfully!")
        print("You can now run the example scripts:")
        print("  python examples/basic_usage.py")
        print("  python examples/full_pipeline.py")
        print("  python examples/model_testing.py")
    else:
        print("\n⚠️  Environment setup incomplete. Please address the issues above.")


if __name__ == "__main__":
    main()
