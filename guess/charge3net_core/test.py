        # Make real prediction

import sys
from pathlib import Path

# Add the parent directory to the path
# sys.path.append(str(Path(__file__).parent.parent))

from guess.charge3net_core.src.real_predictor import load_real_predictor
from convert_chgcar_to_pkl import ChgcarToPklConverter
from convert_pkl_to_chgcar import PklToChgcarConverter


if __name__ == '__main__':
    print("Loading real predictor...")
    predictor = load_real_predictor(
                checkpoint_path="./models/charge3net_mp.pt",
                config_path="./configs/charge3net/test_chgcar_inputs.yaml",
                device='cpu'
            )
            
    model_info = predictor.get_model_info()
    print(f"✓ Real predictor loaded successfully:")


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
