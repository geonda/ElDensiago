# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import argparse
import numpy as np

from pymatgen.io.vasp import Chgcar
from ase.calculators.vasp import VaspChargeDensity

from guess.charge3net_core.src.utils.data import load_atoms_file, load_density_file


class PklToChgcarConverter:
    """Convert pickle format files back to CHGCAR format."""
    
    def __init__(self):
        pass
    
    def convert(self, density_file, atoms_file, aug_chgcar_file=None) -> VaspChargeDensity:
        """
        Convert density and atoms files to VaspChargeDensity object.
        
        Args:
            density_file: Path to .npy density file
            atoms_file: Path to .pkl atoms file
            aug_chgcar_file: Optional path to original CHGCAR file for augmentation data
            
        Returns:
            VaspChargeDensity object
        """
        density = load_density_file(density_file)
        atoms = load_atoms_file(atoms_file)
        
        # retrieve augmentation, if requested
        if aug_chgcar_file is not None:
            aug = Chgcar.from_file(aug_chgcar_file).data_aug
        else:
            aug = None
            
        # extract spin, if available
        if len(density.shape) == 4:  # implies a spin channel exists
            charge_grid = density[..., 0]
            spin_grid = density[..., 1]
        else:
            charge_grid = density
            spin_grid = np.zeros_like(density)
            
        # create Chgcar object
        vcd = VaspChargeDensity(filename=None)
        vcd.atoms.append(atoms)
        vcd.chg.append(charge_grid)
        vcd.chgdiff.append(spin_grid)
        if aug is not None:
            vcd.aug = "".join(aug["total"])
            vcd.augdiff = "".join(aug["diff"])
            
        return vcd
    
    def convert_and_save(self, density_file, atoms_file, output_file, aug_chgcar_file=None):
        """
        Convert density and atoms files to CHGCAR format and save to file.
        
        Args:
            density_file: Path to .npy density file
            atoms_file: Path to .pkl atoms file
            output_file: Path to output CHGCAR file
            aug_chgcar_file: Optional path to original CHGCAR file for augmentation data
        """
        chgcar = self.convert(density_file, atoms_file, aug_chgcar_file=aug_chgcar_file)
        chgcar.write(output_file, format="chgcar")


# Keep the original function for backward compatibility
def deepdft_to_chgcar(density_file, atoms_file, aug_chgcar_file=None) -> VaspChargeDensity:
    converter = PklToChgcarConverter()
    return converter.convert(density_file, atoms_file, aug_chgcar_file)


parser = argparse.ArgumentParser()
parser.add_argument("--density_file", type=str, help="path to .npy density file")
parser.add_argument("--atoms_file", type=str, help="path to .pkl atoms file")
parser.add_argument("--output_file", type=str, help="path to CHGCAR file to save out")
parser.add_argument("--aug_chgcar_file", type=str, default=None, help="path to original CHGCAR file to retrieve augmentation")


if __name__ == "__main__":
    args = parser.parse_args()
    
    converter = PklToChgcarConverter()
    converter.convert_and_save(args.density_file, args.atoms_file, args.output_file, aug_chgcar_file=args.aug_chgcar_file)