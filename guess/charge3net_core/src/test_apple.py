# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).


import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch.multiprocessing as mp
import torch

# Import the converter classes
from convert_chgcar_to_pkl import ChgcarToPklConverter
from convert_pkl_to_chgcar import PklToChgcarConverter

# Import the prediction classes
from .predictor import ChargeDensityPredictor, load_predictor
from .simple_predictor import SimpleChargeDensityPredictor, load_simple_predictor

os.environ["HYDRA_FULL_ERROR"] = "1"

# Since MPS is not CUDA, disable CUDA_VISIBLE_DEVICES or unset it (to avoid confusion)
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

sys.path.append(os.getcwd())
from guess.charge3net_core.src.test import test  # Your test module here


@hydra.main(config_path=None, version_base=None)
def test_from_config(cfg):
    # Setup environment for distributed processing backend - use 'gloo' for CPU/MPS
    env = {
        "backend": "gloo",  # Gloo works with CPU/MPS; NCCL requires CUDA GPUs
        "master_port": str(29501),
        "master_addr": "localhost",
        "world_size": cfg.nnodes * cfg.nprocs,
        "group_rank": int(os.environ.get("SLURM_NODEID", "0")),
    }

    if cfg.nnodes > 1:
        cmd = "scontrol show hostnames " + os.getenv("SLURM_JOB_NODELIST", "")
        env["master_addr"] = subprocess.check_output(cmd.split()).decode().splitlines()[0]
        print(f"Master address set to: {env['master_addr']}")

    os.environ["MASTER_ADDR"] = env["master_addr"]
    os.environ["MASTER_PORT"] = env["master_port"]
    os.environ["WORLD_SIZE"] = str(env["world_size"])

    OmegaConf.resolve(cfg)

    try:
        # Set PyTorch multiprocessing sharing strategy to avoid shared memory issues on macOS
        import torch.multiprocessing as mp
        mp.set_sharing_strategy("file_system")

        # Spawn multiple processes as per config with the test function
        mp.spawn(test, args=(cfg, env), nprocs=cfg.nprocs)
    except Exception as e:
        print(f"An error occurred during multiprocessing: {e}")
        exit(1)


def run_conversion_pipeline():
    """Run the conversion pipeline using the importable classes."""
    
    # Initialize converters
    chgcar_to_pkl = ChgcarToPklConverter()
    pkl_to_chgcar = PklToChgcarConverter()
    
    # Step 1: Convert CHGCAR to pickle format
    print("Converting CHGCAR to pickle format...")
    chgcar_to_pkl.convert_file(
        chgcar_file="./charge3net_inputs_test/si.CHGCAR",
        output_file="si",
        overwrite=True,
        spin=False
    )
    
    # Step 2: Convert prediction back to CHGCAR format
    print("Converting prediction back to CHGCAR format...")
    pkl_to_chgcar.convert_and_save(
        density_file="/Users/geonda/root/machine_learning/_charge3net/charge3net_inputs_test/prediction_lmax_4/2025-08-07/17-08-43/cubes/si.npy",
        atoms_file="/Users/geonda/root/machine_learning/_charge3net/charge3net_inputs_test/si_atoms.pkl",
        output_file="Si.CHGCAR"
    )
    
    print("Conversion pipeline completed!")


def run_prediction_pipeline():
    """Run the prediction pipeline using the SimpleChargeDensityPredictor class."""
    
    print("=== Running Prediction Pipeline ===")
    
    # Initialize the predictor
    checkpoint_path = "./models/charge3net_mp.pt"
    config_path = "configs/charge3net/test_chgcar_inputs.yaml"
    
    try:
        # Load the simple predictor
        print(f"Loading simple predictor from checkpoint: {checkpoint_path}")
        predictor = load_simple_predictor(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=None  # Auto-detect device
        )
        
        # Get model information
        model_info = predictor.get_model_info()
        print(f"Model loaded successfully:")
        print(f"  - Device: {model_info['device']}")
        print(f"  - Model type: {model_info['model_type']}")
        print(f"  - Parameters: {model_info['num_parameters']:,}")
        
        # Test model loading
        if predictor.test_model_loading():
            print("✓ Model test passed")
        else:
            print("✗ Model test failed")
        
        # Example: Convert existing prediction back to CHGCAR
        print("\nConverting existing prediction to CHGCAR format...")
        pkl_to_chgcar = PklToChgcarConverter()
        pkl_to_chgcar.convert_and_save(
            density_file="/Users/geonda/root/machine_learning/_charge3net/charge3net_inputs_test/prediction_lmax_4/2025-08-07/17-08-43/cubes/si.npy",
            atoms_file="/Users/geonda/root/machine_learning/_charge3net/charge3net_inputs_test/si_atoms.pkl",
            output_file="./predictions/Si_prediction.CHGCAR"
        )
        
        print("Prediction pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in prediction pipeline: {e}")
        import traceback
        traceback.print_exc()


def run_full_pipeline():
    """Run the complete pipeline: conversion + prediction + conversion back."""
    
    print("=== Running Full Pipeline ===")
    
    # Step 1: Convert CHGCAR to pickle format
    print("Step 1: Converting CHGCAR to pickle format...")
    chgcar_to_pkl = ChgcarToPklConverter()
    chgcar_to_pkl.convert_file(
        chgcar_file="./charge3net_inputs_test/si.CHGCAR",
        output_file="si",
        overwrite=True,
        spin=False
    )
    
    # Step 2: Load predictor
    print("Step 2: Loading predictor...")
    predictor = load_simple_predictor(
        checkpoint_path="./models/charge3net_mp.pt",
        config_path="configs/charge3net/test_chgcar_inputs.yaml"
    )
    
    model_info = predictor.get_model_info()
    print(f"✓ Predictor loaded successfully on {model_info['device']}")
    
    # Step 3: Convert existing prediction back to CHGCAR
    print("Step 3: Converting existing prediction to CHGCAR format...")
    pkl_to_chgcar = PklToChgcarConverter()
    pkl_to_chgcar.convert_and_save(
        density_file="/Users/geonda/root/machine_learning/_charge3net/charge3net_inputs_test/prediction_lmax_4/2025-08-07/17-08-43/cubes/si.npy",
        atoms_file="./si_atoms.pkl",
        output_file="./predictions/Si_full_pipeline.CHGCAR"
    )
    
    print("Full pipeline completed successfully!")


if __name__ == "__main__":
    # Check command line arguments to determine what to run
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--run-conversion":
            run_conversion_pipeline()
        elif command == "--run-prediction":
            run_prediction_pipeline()
        elif command == "--run-full-pipeline":
            run_full_pipeline()
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  --run-conversion: Run conversion pipeline only")
            print("  --run-prediction: Run prediction pipeline only")
            print("  --run-full-pipeline: Run complete pipeline")
            print("  (no args): Run original test function")
    else:
        # Run the original test function
        test_from_config()
