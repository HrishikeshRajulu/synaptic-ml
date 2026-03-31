"""
Rate coding — spike probability proportional to input value.

The simplest and most common encoding scheme. Each neuron fires
at a rate proportional to its input intensity.
"""

import numpy as np


class RateEncoder:
    """
    Rate encoder: converts real-valued inputs to Poisson spike trains.

    For each neuron, spike probability per timestep = value * max_rate * dt/1000.

    Parameters
    ----------
    time_steps : int
        Number of timesteps to generate (default: 100).
    max_rate : float
        Maximum firing rate in Hz (default: 100 Hz).
    dt : float
        Timestep in ms (default: 1ms).
    """

    def __init__(self, time_steps: int = 100, max_rate: float = 100.0, dt: float = 1.0):
        self.time_steps = time_steps
        self.max_rate = max_rate
        self.dt = dt

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a real-valued vector as a Poisson spike train.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,)
            Input values in [0, 1].

        Returns
        -------
        spikes : np.ndarray, shape (time_steps, n_features), dtype float32
        """
        x = np.clip(x, 0.0, 1.0)
        # Spike probability per timestep
        prob = x * self.max_rate * (self.dt / 1000.0)
        # Sample Poisson spikes
        rand = np.random.rand(self.time_steps, len(x))
        spikes = (rand < prob[None, :]).astype(np.float32)
        return spikes

    def decode(self, spike_train: np.ndarray) -> np.ndarray:
        """
        Decode spike train back to rates in [0, 1].

        Parameters
        ----------
        spike_train : np.ndarray, shape (time_steps, n_neurons)

        Returns
        -------
        rates : np.ndarray, shape (n_neurons,)
        """
        rates = spike_train.mean(axis=0)
        max_possible = self.max_rate * (self.dt / 1000.0)
        return np.clip(rates / (max_possible + 1e-8), 0.0, 1.0)

    def __repr__(self) -> str:
        return f"RateEncoder(T={self.time_steps}, max_rate={self.max_rate}Hz, dt={self.dt}ms)"


class RateDecoder:
    """Decodes spike trains to real values using mean firing rate."""

    def decode(self, spike_train: np.ndarray) -> np.ndarray:
        return spike_train.mean(axis=0)
