"""
Abstract backend interface — the hardware abstraction layer.

Each backend implements this interface, allowing the same SpikingNet
to run on CPU simulation, Intel Loihi 2, BrainScaleS, etc.
"""

import numpy as np
from abc import ABC, abstractmethod


class BackendNotAvailableError(Exception):
    """Raised when required hardware SDK is not installed."""
    pass


class Backend(ABC):
    """
    Abstract base class for neuromorphic hardware backends.

    Implement this to add support for new hardware.
    """

    @abstractmethod
    def load_network(self, network) -> None:
        """
        Load a SpikingNet onto the backend.

        Parameters
        ----------
        network : SpikingNet
        """

    @abstractmethod
    def run(self, input_spikes: np.ndarray, time_steps: int) -> np.ndarray:
        """
        Run the network for one inference.

        Parameters
        ----------
        input_spikes : np.ndarray, shape (time_steps, n_input)
        time_steps : int

        Returns
        -------
        output_scores : np.ndarray, shape (n_output,)
        """

    @abstractmethod
    def get_energy_estimate(self) -> dict:
        """
        Return energy consumption estimates.

        Returns
        -------
        dict with keys: energy_nJ, power_mW, ops_per_inference
        """

    @abstractmethod
    def get_hardware_info(self) -> str:
        """Return a human-readable string describing the hardware."""
