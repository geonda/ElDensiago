# Charge3Net Core Components - API Reference

This document provides detailed API reference for the Charge3Net core components.

## Predictor Classes

### SimpleChargeDensityPredictor

A simplified predictor class for basic model loading and testing.

#### Constructor

```python
SimpleChargeDensityPredictor(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: Optional[str] = None
)
```

**Parameters:**
- `checkpoint_path` (str): Path to the trained model checkpoint
- `config_path` (str, optional): Path to the model configuration file
- `device` (str, optional): Device to run on ('cpu', 'mps', 'cuda'). Auto-detects if None

#### Methods

##### `get_model_info() -> Dict[str, Any]`

Returns information about the loaded model.

**Returns:**
- Dictionary containing model information (device, type, parameters, etc.)

##### `test_model_loading() -> bool`

Tests if the model can be loaded and run a simple forward pass.

**Returns:**
- True if test passes, False otherwise

#### Convenience Function

```python
load_simple_predictor(checkpoint_path: str, **kwargs) -> SimpleChargeDensityPredictor
```

### ChargeDensityPredictor

A full-featured predictor class with datamodule integration.

#### Constructor

```python
ChargeDensityPredictor(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
    max_predict_batch_probes: int = 10000
)
```

**Parameters:**
- `checkpoint_path` (str): Path to the trained model checkpoint
- `config_path` (str, optional): Path to the model configuration file
- `device` (str, optional): Device to run on ('cpu', 'mps', 'cuda'). Auto-detects if None
- `max_predict_batch_probes` (int): Maximum number of probes per batch for prediction

#### Methods

##### `setup_datamodule(input_dir: str)`

Setup the datamodule for predictions.

**Parameters:**
- `input_dir` (str): Directory containing the input data (filelist.txt, split.json, probe_counts.csv)

##### `predict_single(density_file: str, atoms_file: str, output_file: Optional[str] = None, return_cube: bool = True) -> Dict[str, Any]`

Make a prediction for a single material using the full datamodule.

**Parameters:**
- `density_file` (str): Path to the input density .npy file
- `atoms_file` (str): Path to the input atoms .pkl file
- `output_file` (str, optional): Path to save the prediction
- `return_cube` (bool): Whether to return the prediction as a numpy array

**Returns:**
- Dictionary containing prediction results

##### `predict_batch(input_dir: str, output_dir: Optional[str] = None, save_cubes: bool = True) -> Dict[str, Any]`

Make predictions for a batch of materials using the full datamodule.

**Parameters:**
- `input_dir` (str): Directory containing the input data
- `output_dir` (str, optional): Directory to save predictions
- `save_cubes` (bool): Whether to save prediction cubes

**Returns:**
- Dictionary containing batch prediction results

##### `get_model_info() -> Dict[str, Any]`

Returns information about the loaded model.

**Returns:**
- Dictionary containing model information

#### Convenience Function

```python
load_predictor(checkpoint_path: str, **kwargs) -> ChargeDensityPredictor
```

## Converter Classes

### ChgcarToPklConverter

Convert CHGCAR files to pickle format for machine learning models.

#### Constructor

```python
ChgcarToPklConverter()
```

#### Methods

##### `convert(chgcar_file: str, output_density_file: str, output_atoms_file: str, filelist_file: Optional[str] = None, overwrite: bool = False, spin: bool = False)`

Convert a CHGCAR file to pickle format.

**Parameters:**
- `chgcar_file` (str): Path to the input CHGCAR file
- `output_density_file` (str): Path to the output density .npy file
- `output_atoms_file` (str): Path to the output atoms .pkl file
- `filelist_file` (str, optional): Path to the filelist.txt file
- `overwrite` (bool): Whether to overwrite existing files
- `spin` (bool): Whether to handle spin-polarized data

##### `convert_file(chgcar_file: str, output_file: str, overwrite: bool = True, spin: bool = False)`

Convenience method to convert a CHGCAR file with automatic naming.

**Parameters:**
- `chgcar_file` (str): Path to the input CHGCAR file
- `output_file` (str): Base name for output files (without extension)
- `overwrite` (bool): Whether to overwrite existing files
- `spin` (bool): Whether to handle spin-polarized data

### PklToChgcarConverter

Convert pickle format files back to CHGCAR format.

#### Constructor

```python
PklToChgcarConverter()
```

#### Methods

##### `convert(density_file: str, atoms_file: str, aug_chgcar_file: Optional[str] = None) -> VaspChargeDensity`

Convert pickle format files to VaspChargeDensity object.

**Parameters:**
- `density_file` (str): Path to the density .npy file
- `atoms_file` (str): Path to the atoms .pkl file
- `aug_chgcar_file` (str, optional): Path to augmentation CHGCAR file

**Returns:**
- VaspChargeDensity object

##### `convert_and_save(density_file: str, atoms_file: str, output_file: str, aug_chgcar_file: Optional[str] = None)`

Convert pickle format files and save to CHGCAR file.

**Parameters:**
- `density_file` (str): Path to the density .npy file
- `atoms_file` (str): Path to the atoms .pkl file
- `output_file` (str): Path to the output CHGCAR file
- `aug_chgcar_file` (str, optional): Path to augmentation CHGCAR file

## Data Structures

### Model Information Dictionary

```python
{
    "checkpoint_path": str,
    "device": str,
    "model_type": str,
    "num_parameters": int,
    "trainable_parameters": int
}
```

### Prediction Result Dictionary

```python
{
    "filename": str,
    "grid_shape": Tuple[int, ...],
    "prediction_time": float,
    "prediction_cube": Optional[numpy.ndarray],
    "note": Optional[str]
}
```

### Batch Prediction Result Dictionary

```python
{
    "predictions": List[Dict],
    "num_materials": int,
    "output_dir": Optional[str],
    "note": Optional[str]
}
```

## Error Handling

All classes include comprehensive error handling with informative error messages. Common error scenarios:

1. **FileNotFoundError**: When checkpoint or configuration files are missing
2. **ImportError**: When required dependencies are not installed
3. **RuntimeError**: When model loading or prediction fails
4. **ValueError**: When input parameters are invalid

## Examples

See the `examples/` folder for complete working examples of all API usage patterns.
