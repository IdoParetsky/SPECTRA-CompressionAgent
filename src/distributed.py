"""
Single source of truth for (optional) ``torch.distributed`` usage.

SPECTRA is designed to run on a single GPU (e.g. one rtx_6000). Distributed
execution is opt-in: the process group is initialised only when a launcher
(``torchrun`` / ``srun``) exported a world size greater than one. Every other
module must query rank/device through this helper instead of calling
``torch.distributed`` directly, so that a plain ``python
a2c_agent_reinforce_runner.py ...`` invocation never touches NCCL.
"""

import os

import torch
import torch.distributed as dist

_LAUNCHER_WORLD_SIZE_KEYS = ("WORLD_SIZE", "SLURM_NTASKS")
_LAUNCHER_LOCAL_RANK_KEYS = ("LOCAL_RANK", "SLURM_LOCALID")


def _env_int(keys, default=0) -> int:
    for key in keys:
        value = os.environ.get(key)
        if value:
            try:
                return int(value)
            except ValueError:
                continue
    return default


def launcher_world_size() -> int:
    """World size advertised by the launcher, before the group is created."""
    return _env_int(_LAUNCHER_WORLD_SIZE_KEYS, default=1)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    """Rank within the node. Unlike the global rank this is a valid GPU index."""
    return _env_int(_LAUNCHER_LOCAL_RANK_KEYS, default=0)


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if is_distributed():
        dist.barrier()


def maybe_init_distributed() -> bool:
    """
    Initialise the process group only when launched with more than one process.

    Returns:
        bool: True when distributed execution is active.
    """
    if is_distributed():
        return True
    if not dist.is_available() or launcher_world_size() <= 1:
        return False
    # env:// requires MASTER_ADDR/MASTER_PORT; torchrun and srun both provide them.
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return True


def _flag_tensor(value: bool) -> torch.Tensor:
    device = torch.device(f"cuda:{get_local_rank()}") if torch.cuda.is_available() else torch.device("cpu")
    return torch.tensor([1.0 if value else 0.0], device=device)


def broadcast_flag(value: bool, src: int = 0) -> bool:
    """
    Adopt rank `src`'s answer on every rank.

    Loop conditions in the training loop depend on per-rank state (rewards, wall clock). If
    each rank decided independently, one could leave the loop while another waited inside a
    collective, deadlocking the job. A single authoritative verdict keeps them in lockstep.
    """
    if not is_distributed():
        return value
    tensor = _flag_tensor(value)
    dist.broadcast(tensor, src=src)
    return bool(tensor.item())


def all_agree(value: bool) -> bool:
    """True only when every rank reports True."""
    if not is_distributed():
        return value
    tensor = _flag_tensor(value)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(tensor.item())


def summary() -> str:
    """One-line description of the execution topology, for the run log."""
    if not is_distributed():
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return f"single process, {visible} GPU(s) visible, using cuda:{get_local_rank()}"
    return f"distributed, rank {get_rank()}/{get_world_size()} on cuda:{get_local_rank()}"


def resolve_device() -> torch.device:
    """Device owned by this process; also pins the CUDA context to it."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{get_local_rank()}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module of a (possibly nested) DataParallel/DDP wrapper."""
    while isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model = model.module
    return model
