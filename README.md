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

