import torch
import numpy as np
from hydra.utils import instantiate
from torch.distributed import destroy_process_group, init_process_group

from guess.charge3net_core.src.trainer import Trainer

def test(rank, cfg, env):
    print(f"Initializing on rank {rank} with environment {env}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Compute global rank
    global_rank = env["group_rank"] * cfg.nprocs + rank

    # Initialize distributed process group with gloo backend (supports CPU and MPS)
    init_process_group(backend="gloo", init_method="env://", rank=global_rank, world_size=env["world_size"])

    # Set device: prefer MPS if available, else CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print(f"Using MPS device for rank {rank}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device for rank {rank}")

    # Instantiate model, then move to device
    model = instantiate(cfg.model.model).to(device)
    optimizer = instantiate(cfg.model.optimizer)(model.parameters())
    scheduler = instantiate(cfg.model.lr_scheduler)(optimizer)
    criterion = instantiate(cfg.model.criterion).to(device)

    # Instantiate data module
    datamodule = instantiate(cfg.data)

    # Create trainer and run test with data loader on the chosen device
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        log_dir=cfg.log_dir,
        gpu_id=rank,
        global_rank=global_rank,
        load_checkpoint_path=cfg.checkpoint_path,
    )

    if cfg.cube_dir is not None:
        assert cfg.data.test_probes is None, "Cannot write cubes without data.test_probes=null"

    trainer.test(test_dl=datamodule.test_dataloader(), cube_dir=cfg.cube_dir)

    # Cleanup distributed group
    destroy_process_group()
