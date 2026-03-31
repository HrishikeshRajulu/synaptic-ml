from .base import Backend, BackendNotAvailableError
from .cpu import CPUBackend
from .loihi2 import Loihi2Backend
from .brainscales import BrainScalesBackend
from .akida import AkidaBackend


def get_backend(target: str, **kwargs) -> Backend:
    """
    Get a backend by name.

    Parameters
    ----------
    target : str
        'cpu', 'loihi2', 'brainscales', or 'akida'
    **kwargs
        Passed to backend constructor.
        e.g. use_hardware=True for AkidaBackend

    Returns
    -------
    Backend instance
    """
    backends = {
        "cpu": CPUBackend,
        "loihi2": Loihi2Backend,
        "brainscales": BrainScalesBackend,
        "akida": AkidaBackend,
    }
    if target not in backends:
        raise ValueError(
            f"Unknown backend: '{target}'. "
            f"Available: {list(backends.keys())}"
        )
    return backends[target](**kwargs)


__all__ = [
    "Backend", "BackendNotAvailableError",
    "CPUBackend", "Loihi2Backend", "BrainScalesBackend", "AkidaBackend",
    "get_backend",
]
