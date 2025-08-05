import os
import json
import math
import timeit
import logging
from tqdm import tqdm
import numpy as np
import torch
import ase.io
from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.io.ase import AseAtomsAdaptor
import ase.io.cube
import pkg_resources

import guess.densitymodel as densitymodel
import guess.dataset as dataset
import guess.utils as utils


def get_package_data_path(relative_path):
    """
    Get the absolute path to a file within the package data.
    
    :param relative_path: Relative path within the package
    :type relative_path: str
    :return: Absolute path to the file
    :rtype: str
    """
    try:
        return pkg_resources.resource_filename('guess', relative_path)
    except Exception as e:
        raise FileNotFoundError(f"Could not find package data: {relative_path}. Error: {e}")


# Class for lazy creation of spatial grid coordinates
class LazyMeshGrid:
    """Creates a spatial grid for calculating atomic densities."""
    def __init__(self, cell, grid_step, origin=None, adjust_grid_step=False):
        """
        Initializes the grid with an option to automatically adjust the grid step size.
        
        :param cell: Unit cell of the atomic structure
        :type cell: np.ndarray
        :param grid_step: Grid spacing
        :type grid_step: float
        :param origin: Origin point of the grid, defaults to None
        :type origin: Optional[np.ndarray]
        :param adjust_grid_step: Automatically adjust the grid step size, defaults to False
        :type adjust_grid_step: bool
        """
        self.cell = cell
        if adjust_grid_step:
            n_steps = np.round(cell.lengths() / grid_step)
            self.scaled_grid_vectors = [np.arange(n) / n for n in n_steps]
            self.adjusted_grid_step = cell.lengths() / n_steps
        else:
            self.scaled_grid_vectors = [
                np.arange(0, length, grid_step) / length for length in cell.lengths()
            ]
        self.shape = np.array([len(g) for g in self.scaled_grid_vectors] + [3])
        self.origin = np.zeros(3) if origin is None else origin
        self.origin = np.expand_dims(self.origin, 0)

    def __getitem__(self, indices):
        """
        Retrieves positions within the grid based on given indices.
        
        :param indices: Index positions in the grid
        :type indices: np.ndarray
        :return: Positions in the unit cell
        :rtype: np.ndarray
        """
        indices = np.array(indices)
        if not (indices.ndim == 2 and indices.shape[0] == 3):
            raise NotImplementedError("Indexing must be a 3xN array-like object")
        grid_a = self.scaled_grid_vectors[0][indices[0]]
        grid_b = self.scaled_grid_vectors[1][indices[1]]
        grid_c = self.scaled_grid_vectors[2][indices[2]]
        grid_pos = np.stack([grid_a, grid_b, grid_c], axis=1)
        grid_pos = np.dot(grid_pos, self.cell)
        grid_pos += self.origin
        return grid_pos


class MlDensity:
    """Class for predicting electronic density using machine learning models."""
    def __init__(
        self,
        model="nmc_schnet",
        device="cpu",
        grid_step=0.05,
        vacuum=1.0,
        probe_count=5000,
        ignore_pbc=False,
        force_pbc=False,
        output_dir="./model_prediction",
        compute_iri=False,
        compute_dori=False,
        compute_hessian_eig=False,
    ):
        """
        Initializes the ML Density predictor.
        
        :param model: Name of the pretrained model, defaults to "nmc_schnet"
        :type model: str
        :param device: Device for computation ('cpu' or 'cuda'), defaults to 'cpu'
        :type device: str
        :param grid_step: Step size for the computational grid, defaults to 0.05
        :type grid_step: float
        :param vacuum: Vacuum padding around molecules, defaults to 1.0
        :type vacuum: float
        :param probe_count: Number of probe points for density calculation, defaults to 5000
        :type probe_count: int
        :param ignore_pbc: Ignore periodic boundary conditions, defaults to False
        :type ignore_pbc: bool
        :param force_pbc: Force periodic boundary conditions, defaults to False
        :type force_pbc: bool
        :param output_dir: Directory for saving output files, defaults to './model_prediction'
        :type output_dir: str
        :param compute_iri: Compute IRIs, defaults to False
        :type compute_iri: bool
        :param compute_dori: Compute DORIs, defaults to False
        :type compute_dori: bool
        :param compute_hessian_eig: Compute Hessian eigenvalues, defaults to False
        :type compute_hessian_eig: bool
        """
        self.model_name = model
        # Get the package directory for pretrained models
        self.model_dir = pkg_resources.resource_filename('guess', f'pretrained_models/{self.model_name}/')
        self.device = torch.device(device)
        self.grid_step = grid_step
        self.vacuum = vacuum
        self.probe_count = probe_count
        self.ignore_pbc = ignore_pbc
        self.force_pbc = force_pbc
        self.output_dir = output_dir
        self.compute_iri = compute_iri
        self.compute_dori = compute_dori
        self.compute_hessian_eig = compute_hessian_eig

        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logging()
        self.model, self.cutoff = self._load_model()

    def _setup_logging(self):
        """Sets up logging to both console and log file."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, "predict.log"), mode="w"),
                logging.StreamHandler(),
            ],
        )


    def _load_model(self):
        try:
            # Load model arguments
            args_path = os.path.join(self.model_dir, "arguments.json")
            if not os.path.exists(args_path):
                raise FileNotFoundError(f"Model arguments file not found: {args_path}")
                
            with open(args_path, "r") as f:
                runner_args = json.load(f)
                
            # Determine correct model class
            if runner_args.get('use_painn_model', False):
                model = densitymodel.PainnDensityModel(runner_args['num_interactions'],
                                                        runner_args['node_size'],
                                                        runner_args['cutoff'])
            else:
                model = densitymodel.DensityModel(runner_args['num_interactions'],
                                                  runner_args['node_size'],
                                                  runner_args['cutoff'])
            model.to(self.device)
            
            # Load model weights
            model_path = os.path.join(self.model_dir, "best_model.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weights file not found: {model_path}")
                
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict["model"])
            logging.info(f"Model '{self.model_name}' loaded successfully from {self.model_dir}")
            return model, runner_args["cutoff"]
            
        except Exception as e:
            logging.error(f"Failed to load model '{self.model_name}': {e}")
            logging.error(f"Model directory: {self.model_dir}")
            logging.error(f"Available models: {self._list_available_models()}")
            raise
    
    def _list_available_models(self):
        """List all available pretrained models."""
        try:
            models_dir = pkg_resources.resource_filename('guess', 'pretrained_models')
            if os.path.exists(models_dir):
                return [d for d in os.listdir(models_dir) 
                       if os.path.isdir(os.path.join(models_dir, d))]
            else:
                return []
        except Exception:
            return []
    
    @classmethod
    def list_available_models(cls):
        """List all available pretrained models."""
        try:
            models_dir = pkg_resources.resource_filename('guess', 'pretrained_models')
            if os.path.exists(models_dir):
                models = [d for d in os.listdir(models_dir) 
                         if os.path.isdir(os.path.join(models_dir, d))]
                return sorted(models)
            else:
                return []
        except Exception:
            return []

    @staticmethod
    def ceil_float(x, step_size):
        """
        Rounds a number up to the nearest multiple of step_size.
        
        :param x: Input value
        :type x: float
        :param step_size: Step increment
        :type step_size: float
        :return: Ceiling-floored value
        :rtype: float
        """
        result = math.ceil(x / step_size) * step_size
        eps = 2 * np.finfo(float).eps * result
        return result - eps

    def load_atoms(self, atomspath):
        """
        Loads an atomic structure and generates corresponding grid positions.
        
        :param atomspath: Path to the atomic structure file
        :type atomspath: str
        :return: Dictionary containing atoms, origin, and grid positions
        :rtype: dict
        """
        atoms = ase.io.read(atomspath)
        if any(atoms.get_pbc()):
            atoms, grid_pos, origin = self.load_material(atoms)
        else:
            atoms, grid_pos, origin = self.load_molecule(atoms)
        metadata = {"filename": atomspath}
        return {
            "atoms": atoms,
            "origin": origin,
            "grid_position": grid_pos,
            "metadata": metadata,
        }

    def load_material(self, atoms):
        """
        Loads material structures with support for periodic boundaries.
        
        :param atoms: Atomic structure
        :type atoms: ase.Atoms
        :return: Atoms, grid positions, and origin point
        :rtype: Tuple[ase.Atoms, LazyMeshGrid, np.ndarray]
        """
        atoms = atoms.copy()
        grid_pos = LazyMeshGrid(atoms.get_cell(), self.grid_step, adjust_grid_step=True)
        origin = np.zeros(3)
        return atoms, grid_pos, origin

    def load_molecule(self, atoms):
        """
        Loads molecular structures with added vacuum padding.
        
        :param atoms: Atomic structure
        :type atoms: ase.Atoms
        :return: Atoms, grid positions, and origin point
        :rtype: Tuple[ase.Atoms, LazyMeshGrid, np.ndarray]
        """
        atoms = atoms.copy()
        atoms.center(vacuum=self.vacuum)
        a, b, c, _, _, _ = atoms.get_cell_lengths_and_angles()
        a = self.ceil_float(a, self.grid_step)
        b = self.ceil_float(b, self.grid_step)
        c = self.ceil_float(c, self.grid_step)
        atoms.set_cell((a, b, c))
        origin = np.zeros(3)
        grid_pos = LazyMeshGrid(atoms.get_cell(), self.grid_step)
        return atoms, grid_pos, origin

    def save_chgcar(self, chgcar_path="CHGCAR"):
        """
        Converts a cube file from DeepDFT prediction into CHGCAR format and saves it.
        
        :param chgcar_path: Output path for CHGCAR file, defaults to "CHGCAR"
        :type chgcar_path: str
        """
        with open(f"{self.output_dir}/prediction.cube", "rb") as f:
            cube_data = ase.io.cube.read_cube(f)
        density = cube_data["data"]
        atoms = cube_data["atoms"]
        structure = AseAtomsAdaptor.get_structure(atoms)
        data = {"total": density.astype(float)}
        chgcar = Chgcar(structure, data)
        chgcar.write_file(chgcar_path)
        logging.info("Cube converted to CHGCAR successfully.")

    def predict(self, atoms_file):
        """
        Predicts the electronic density for a given atomic structure.
        
        :param atoms_file: Filepath to the atomic structure
        :type atoms_file: str
        """
        logging.info(f"Starting prediction for {atoms_file}")
        density_dict = self.load_atoms(atoms_file)
        device = self.device
        grid_shape = density_dict["grid_position"].shape[:3]
        full_density = np.zeros(grid_shape, dtype=np.float32)

        if self.ignore_pbc and self.force_pbc:
            raise ValueError("Parameters ignore_pbc and force_pbc are mutually exclusive!")

        set_pbc = None
        if self.ignore_pbc:
            set_pbc = False
        elif self.force_pbc:
            set_pbc = True

        start_time = timeit.default_timer()

        with torch.no_grad():
            logging.debug("Calculating atom-to-atom graph...")
            collate_fn = dataset.CollateFuncAtoms(
                cutoff=self.cutoff,
                pin_memory=(device.type == "cuda"),
                set_pbc_to=set_pbc,
            )
            graph_dict = collate_fn([density_dict])
            logging.debug("Calculating atomic representations...")
            device_batch = {k: v.to(device=device, non_blocking=True) for k, v in graph_dict.items()}

            if isinstance(self.model, densitymodel.PainnDensityModel):
                scalar_rep, vector_rep = self.model.atom_model(device_batch)
            else:
                rep = self.model.atom_model(device_batch)

            logging.debug("Atomic representations calculated.")

            density_iter = dataset.DensityGridIterator(
                density_dict, self.probe_count, self.cutoff, set_pbc_to=set_pbc
            )

            for slice_id, probe_graph_dict in tqdm(
                enumerate(density_iter),
                desc="Predicting density...",
            ):
                probe_dict = dataset.collate_list_of_dicts([probe_graph_dict])
                probe_dict = {k: v.to(device=device, non_blocking=True) for k, v in probe_dict.items()}

                device_batch.update(
                    {
                        "probe_edges": probe_dict["probe_edges"],
                        "probe_edges_displacement": probe_dict["probe_edges_displacement"],
                        "probe_xyz": probe_dict["probe_xyz"],
                        "num_probe_edges": probe_dict["num_probe_edges"],
                        "num_probes": probe_dict["num_probes"],
                    }
                )

                if isinstance(self.model, densitymodel.PainnDensityModel):
                    results = self.model.probe_model(
                        device_batch,
                        scalar_rep,
                        vector_rep,
                        compute_iri=self.compute_iri,
                        compute_dori=self.compute_dori,
                        compute_hessian=self.compute_hessian_eig,
                    )
                else:
                    results = self.model.probe_model(
                        device_batch,
                        rep,
                        compute_iri=self.compute_iri,
                        compute_dori=self.compute_dori,
                        compute_hessian=self.compute_hessian_eig,
                    )

                if self.compute_iri or self.compute_dori or self.compute_hessian_eig:
                    density, outputs = results
                else:
                    density = results

                # Map predicted density slice to correct full_density positions
                start_index = slice_id * self.probe_count
                end_index = min(start_index + self.probe_count, np.prod(grid_shape))
                linear_indices = np.arange(start_index, end_index)
                grid_indices = np.unravel_index(linear_indices, grid_shape)
                full_density[grid_indices] = density.cpu().detach().numpy().flatten()

        end_time = timeit.default_timer()
        logging.info(f"Prediction done. Time elapsed: {end_time - start_time:.2f} seconds")

        # Save CHGCAR as before
        structure = AseAtomsAdaptor.get_structure(density_dict["atoms"])
        data = {"total": full_density.astype(float)}
        chgcar = Chgcar(structure, data)
        chgcar.write_file(os.path.join(self.output_dir, "CHGCAR"))
        logging.info(f"Saved predicted density to CHGCAR at {self.output_dir}/CHGCAR")
