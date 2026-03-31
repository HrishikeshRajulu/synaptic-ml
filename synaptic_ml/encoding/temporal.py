"""
Temporal coding — information encoded in precise spike timing.

Earlier spike = stronger input. The most energy-efficient encoding:
at most one spike per neuron per sample (time-to-first-spike).
"""

import numpy as np


class TemporalEncoder:
    """
    Time-to-First-Spike (TTFS) encoder.

    Neurons with higher input values fire earlier in the time window.
    Very energy efficient: exactly one spike per active neuron.

    Parameters
    ----------
    time_steps : int
        Duration of the encoding window.
    threshold : float
        Minimum value to generate any spike (default: 0.01).
    """

    def __init__(self, time_steps: int = 100, threshold: float = 0.01):
        self.time_steps = time_steps
        self.threshold = threshold

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode inputs using time-to-first-spike.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,), values in [0, 1]

        Returns
        -------
        spikes : np.ndarray, shape (time_steps, n_features)
            Each neuron fires at most once, at timestep proportional to 1/x.
        """
        x = np.clip(x, 0.0, 1.0)
        n = len(x)
        spikes = np.zeros((self.time_steps, n), dtype=np.float32)

        for i, val in enumerate(x):
            if val >= self.threshold:
                # Higher value → earlier spike time
                t_fire = int((1.0 - val) * (self.time_steps - 1))
                spikes[t_fire, i] = 1.0

        return spikes

    def decode(self, spike_train: np.ndarray) -> np.ndarray:
        """
        Decode TTFS spike train back to real values.

        Parameters
        ----------
        spike_train : np.ndarray, shape (time_steps, n_neurons)

        Returns
        -------
        values : np.ndarray, shape (n_neurons,), in [0, 1]
        """
        n = spike_train.shape[1]
        values = np.zeros(n, dtype=np.float32)
        for i in range(n):
            spike_times = np.where(spike_train[:, i] > 0)[0]
            if len(spike_times) > 0:
                t = spike_times[0]
                values[i] = 1.0 - t / (self.time_steps - 1)
        return values

    def __repr__(self) -> str:
        return f"TemporalEncoder(T={self.time_steps}, threshold={self.threshold})"


class PhaseEncoder:
    """
    Phase coding — spike timing relative to a global oscillation.

    More complex but carries more information per spike than TTFS.
    Used in hippocampal theta oscillations in the brain.

    Parameters
    ----------
    time_steps : int
    n_cycles : int
        Number of oscillation cycles within time_steps.
    """

    def __init__(self, time_steps: int = 100, n_cycles: int = 5):
        self.time_steps = time_steps
        self.n_cycles = n_cycles

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        n = len(x)
        spikes = np.zeros((self.time_steps, n), dtype=np.float32)
        cycle_len = self.time_steps // self.n_cycles

        for cycle in range(self.n_cycles):
            for i, val in enumerate(x):
                # Fire at phase proportional to value within each cycle
                phase_offset = int((1.0 - val) * (cycle_len - 1))
                t_fire = cycle * cycle_len + phase_offset
                if t_fire < self.time_steps:
                    spikes[t_fire, i] = 1.0

        return spikes

    def __repr__(self) -> str:
        return f"PhaseEncoder(T={self.time_steps}, cycles={self.n_cycles})"
