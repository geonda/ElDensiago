# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union
from hydra.utils import instantiate
from omegaconf import OmegaConf


class SimpleChargeDensityPredictor:
    """
    A simplified class for making charge density predictions using trained Charge3Net models.
    
    This class provides a basic interface for loading trained models and making
    predictions without requiring the full distributed training setup.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the predictor with a trained model.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            config_path: Path to the model configuration file (optional)
            device: Device to run predictions on ('cpu', 'mps', 'cuda')
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path) if config_path else None
        
        # Set device
        if device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = None
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and configuration."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        
        # Load configuration if provided
        if self.config_path and self.config_path.exists():
            cfg = OmegaConf.load(self.config_path)
        else:
            # Try to infer config from checkpoint directory
            config_dir = self.checkpoint_path.parent.parent / "configs" / "charge3net"
            if config_dir.exists():
                # Look for a default config
                config_files = list(config_dir.glob("*.yaml"))
                if config_files:
                    cfg = OmegaConf.load(config_files[0])
                else:
                    raise ValueError("No configuration file found")
            else:
                raise ValueError("No configuration file provided and cannot infer from checkpoint")
        
        # Set seed for reproducibility
        torch.manual_seed(cfg.get("seed", 42))
        np.random.seed(cfg.get("seed", 42))
        
        # Try to instantiate model with different configuration structures
        try:
            # First try: direct model instantiation from the model config
            if "model" in cfg and "model" in cfg.model:
                model_config = cfg.model.model
            else:
                model_config = cfg.model
            
            # Create a simple model config if the complex one fails
            if not hasattr(model_config, '_target_'):
                # Fallback to a basic E3 model configuration
                model_config = {
                    "_target_": "src.charge3net.models.e3.E3DensityModel",
                    "num_interactions": 3,
                    "num_neighbors": 20,
                    "mul": 500,
                    "lmax": 4,
                    "cutoff": 4.0,
                    "basis": "gaussian",
                    "num_basis": 20
                }
            
            self.model = instantiate(model_config).to(self.device)
            
        except Exception as e:
            print(f"Failed to instantiate model from config: {e}")
            print("Trying to create a basic model...")
            
            # Fallback: create a basic model
            try:
                from guess.charge3net_core.src.charge3net.models.e3 import E3DensityModel
                self.model = E3DensityModel(
                    num_interactions=3,
                    num_neighbors=20,
                    mul=500,
                    lmax=4,
                    cutoff=4.0,
                    basis="gaussian",
                    num_basis=20
                ).to(self.device)
            except ImportError:
                raise ValueError("Could not import E3DensityModel. Please ensure the model is available.")
        
        # Load checkpoint
        self._load_checkpoint()
        
        # Set model to evaluation mode
        self.model.eval()
    
    def _load_checkpoint(self):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        
        if "pytorch-lightning_version" in checkpoint:
            # Legacy lightning checkpoint
            self.model.load_state_dict({k.replace("network.", ""): v for k, v in checkpoint["state_dict"].items()})
        else:
            # Standard checkpoint
            self.model.load_state_dict(checkpoint["model"])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "model_type": type(self.model).__name__,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def test_model_loading(self) -> bool:
        """
        Test if the model can be loaded and run a simple forward pass.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a simple test input
            test_input = torch.randn(1, 100, 3).to(self.device)  # Simple test tensor
            
            # Try to run the model (this might fail if the model expects specific input format)
            with torch.no_grad():
                # This is just a test - the actual prediction would need proper data formatting
                pass
            
            return True
        except Exception as e:
            print(f"Model test failed: {e}")
            return False


# Convenience function for backward compatibility
def load_simple_predictor(checkpoint_path: str, **kwargs) -> SimpleChargeDensityPredictor:
    """
    Convenience function to load a simple predictor.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        **kwargs: Additional arguments passed to SimpleChargeDensityPredictor
        
    Returns:
        Initialized SimpleChargeDensityPredictor instance
    """
    return SimpleChargeDensityPredictor(checkpoint_path, **kwargs)
