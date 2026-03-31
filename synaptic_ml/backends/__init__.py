from .base import Backend, BackendNotAvailableError
from .cpu import CPUBackend
from .loihi2 import Loihi2Backend
from .brainscales import BrainScalesBackend


def get_backend(target: str) -> Backend:
    """
    Get a backend by name.

    Parameters
    ----------
    target : str
        'cpu', 'loihi2', or 'brainscales'

    Returns
    -------
    Backend instance
    """
    backends = {
        "cpu": CPUBackend,
        "loihi2": Loihi2Backend,
        "brainscales": BrainScalesBackend,
    }
    if target not in backends:
        raise ValueError(
            f"Unknown backend: '{target}'. "
            f"Available: {list(backends.keys())}"
        )
    return backends[target]()


__all__ = [
    "Backend", "BackendNotAvailableError",
    "CPUBackend", "Loihi2Backend", "BrainScalesBackend",
    "get_backend",
]
