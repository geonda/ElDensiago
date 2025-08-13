# ElDensiago

Frok from DeepDFT. Concise version for DFT acceleration.

## Description

ElDensiago is a Python package that uses deep learning models (trained by DeepDFT team) to predict electronic density distributions for atomic structures.

## Features

- **Multiple Models**: Support for SchNet and PaiNN architectures
- **Pretrained Models**: Ready-to-use models for NMC, QM9, and ethylene carbonate systems
- **Flexible Input**: Accepts various atomic structure formats (XYZ, CIF, etc.)
- **Periodic Systems**: Full support for materials with periodic boundary conditions
- **GPU Acceleration**: CUDA support for faster computation
- **Multiple Output Formats**: CHGCAR and cube file outputs

## Installation

### From Source

```bash
git clone https://github.com/yourusername/ElDensiago.git
cd ElDensiago
pip install -e .
```

### Dependencies

The package requires:
- Python >= 3.7
- PyTorch
- ASE (Atomic Simulation Environment)
- NumPy
- tqdm
- pymatgen
- lz4
- asap3

## Quick Start

### Python API

```python
from guess import MlDensity

# List available models
models = MlDensity.list_available_models()
print(f"Available models: {models}")

# Create a density predictor
predictor = MlDensity(
    model='nmc_schnet',      # Choose pretrained model
    device='cpu',            # or 'cuda' for GPU
    grid_step=0.5,           # Grid spacing
    probe_count=10,          # Probe points per batch
    force_pbc=True           # Force periodic boundary conditions
)

# Predict density for a structure
predictor.predict('path/to/structure.xyz')
```

### Package Data Access

The package includes pretrained models that are automatically accessible:

```python
from guess import get_package_data_path

# Get path to a specific model
model_path = get_package_data_path("pretrained_models/nmc_schnet")
print(f"Model directory: {model_path}")

# Get path to model arguments
args_path = get_package_data_path("pretrained_models/nmc_schnet/arguments.json")
```
## Output

The prediction generates:
- `CHGCAR`: VASP charge density format
- `prediction.cube`: Gaussian cube format
- `predict.log`: Detailed log file

## License

MIT License - see LICENSE file for details.

### for charge3net_core

# Charge3Net Core Components

This folder contains all the essential components for successful testing and predicting charge density using Charge3Net models.

## 📁 Folder Structure

```
charge3net_core/
├── README.md                 # This file
├── examples/                 # Example scripts and usage
├── models/                   # Pre-trained model files
├── configs/                  # Configuration files
├── data/                     # Sample data and utilities
├── utils/                    # Utility scripts
└── docs/                     # Documentation
```

## 🚀 Quick Start

### 1. Basic Usage

```python
from guess.charge3net_core.src.simple_predictor import load_simple_predictor

# Load a model
predictor = load_simple_predictor(
    checkpoint_path="./models/charge3net_mp.pt",
    config_path="./configs/charge3net/test_chgcar_inputs.yaml"
)

# Test model loading
if predictor.test_model_loading():
    print("✓ Model loaded successfully!")
```

### 2. Full Predictor Usage (with Datamodule)

```python
from guess.charge3net_core.src.predictor import load_predictor

# Load the full predictor
predictor = load_predictor(
    checkpoint_path="./models/charge3net_mp.pt",
    config_path="./configs/charge3net/test_chgcar_inputs.yaml"
)

# Setup datamodule
predictor.setup_datamodule("./data/sample_inputs/")

# Make predictions
result = predictor.predict_single(
    density_file="./data/sample_inputs/si.npy",
    atoms_file="./data/sample_inputs/si_atoms.pkl",
    output_file="./predictions/si_prediction.npy"
)
```

### 3. Real Data Loading (Production Ready)

```python
from guess.charge3net_core.src.real_predictor import load_real_predictor

# Load the real predictor with actual data loading
predictor = load_real_predictor(
    checkpoint_path="./models/charge3net_mp.pt",
    config_path="./configs/charge3net/test_chgcar_inputs.yaml"
)

# Make real predictions with actual graph construction
result = predictor.predict_single_real(
    density_file="./data/sample_inputs/si.npy",
    atoms_file="./data/sample_inputs/si_atoms.pkl",
    output_file="./predictions/si_real_prediction.npy"
)

# Batch processing with real data loading
results = predictor.predict_batch_real(
    input_dir="./data/sample_inputs/",
    output_dir="./predictions/real_batch_predictions/"
)
```

### 4. File Conversion

```python
from convert_chgcar_to_pkl import ChgcarToPklConverter
from convert_pkl_to_chgcar import PklToChgcarConverter

# Convert CHGCAR to pickle format
converter = ChgcarToPklConverter()
converter.convert_file(
    chgcar_file="./data/sample_inputs/si.CHGCAR",
    output_file="si",
    overwrite=True
)

# Convert back to CHGCAR
pkl_converter = PklToChgcarConverter()
pkl_converter.convert_and_save(
    density_file="./predictions/si_prediction.npy",
    atoms_file="./data/sample_inputs/si_atoms.pkl",
    output_file="./predictions/Si_prediction.CHGCAR"
)
```

## 📋 Requirements

- Python 3.8+
- PyTorch
- NumPy
- ASE (Atomic Simulation Environment)
- pymatgen
- omegaconf
- hydra

## 🔧 Installation

```bash
pip install torch numpy ase pymatgen omegaconf hydra
```

## 📖 Documentation

See the `docs/` folder for detailed documentation on:
- Model architecture
- Data preparation
- Configuration options
- Troubleshooting

## 🧪 Testing

Run the example scripts in the `examples/` folder to test the functionality:

```bash
python examples/basic_usage.py
python examples/full_pipeline.py
python examples/model_testing.py
python examples/real_data_loading.py
```

## 📁 Key Files

### Core Components
- `src/predictor.py` - Full-featured predictor with datamodule integration
- `src/simple_predictor.py` - Basic predictor for model loading and testing
- `src/real_predictor.py` - Real predictor with actual data loading and graph construction
- `convert_chgcar_to_pkl.py` - CHGCAR to pickle converter
- `convert_pkl_to_chgcar.py` - Pickle to CHGCAR converter

### Configuration
- `configs/test_chgcar_inputs.yaml` - Test configuration
- `configs/model/e3_density.yaml` - E3 model configuration

### Data
- `data/sample_inputs/` - Sample input files
- `data/filelist.txt` - File list for batch processing
- `data/split.json` - Data split configuration

## 🎯 Use Cases

1. **Single Material Prediction**: Predict charge density for individual materials
2. **Batch Processing**: Process multiple materials efficiently
3. **Real Data Loading**: Use actual graph construction and data loading
4. **File Format Conversion**: Convert between CHGCAR and pickle formats
5. **Model Testing**: Verify model loading and basic functionality
6. **Integration**: Use as building blocks for larger workflows

## 🔍 Troubleshooting

Common issues and solutions:

1. **Model Loading Errors**: Check checkpoint path and configuration
2. **Data Format Issues**: Ensure proper file structure and format
3. **Memory Issues**: Reduce batch size or number of probes
4. **Import Errors**: Verify Python path and dependencies

## 📞 Support

For issues and questions, check the documentation in `docs/` or refer to the main Charge3Net repository.
