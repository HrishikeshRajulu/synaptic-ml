"""
Population coding and delta encoding.

Population coding: a value is represented by the pattern of activity
across a population of neurons with overlapping Gaussian tuning curves
(like how the visual cortex encodes orientation).

Delta encoding: spikes only on changes — ideal for event-driven sensors
and neuromorphic IoT applications.
"""

import numpy as np


class PopulationEncoder:
    """
    Population encoder with Gaussian tuning curves.

    Each neuron responds maximally to a different preferred value,
    with a Gaussian falloff. Biologically realistic and robust to noise.

    Parameters
    ----------
    n_neurons : int
        Neurons per input feature (population size, default: 10).
    sigma : float
        Width of Gaussian tuning curves (default: 0.5).
    time_steps : int
        How long each encoded value is held.
    """

    def __init__(self, n_neurons: int = 10, sigma: float = 0.5, time_steps: int = 100):
        self.n_neurons = n_neurons
        self.sigma = sigma
        self.time_steps = time_steps

        # Preferred values evenly spaced in [0, 1]
        self.preferred = np.linspace(0.0, 1.0, n_neurons)

    def encode_scalar(self, x: float) -> np.ndarray:
        """
        Encode a single scalar value as population activity.

        Returns
        -------
        activity : np.ndarray, shape (n_neurons,), values in [0, 1]
        """
        return np.exp(-0.5 * ((x - self.preferred) / self.sigma) ** 2).astype(np.float32)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a feature vector using population coding.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,)

        Returns
        -------
        spikes : np.ndarray, shape (time_steps, n_features * n_neurons)
        """
        x = np.clip(x, 0.0, 1.0)
        n_features = len(x)
        activities = np.zeros(n_features * self.n_neurons, dtype=np.float32)

        for i, val in enumerate(x):
            act = self.encode_scalar(val)
            activities[i * self.n_neurons:(i + 1) * self.n_neurons] = act

        # Repeat activity as rate-coded spikes
        prob = activities * 0.8  # max 80% spike probability
        rand = np.random.rand(self.time_steps, len(activities))
        spikes = (rand < prob[None, :]).astype(np.float32)
        return spikes

    @property
    def output_size(self):
        return None  # depends on input; must be multiplied by n_features

    def __repr__(self) -> str:
        return f"PopulationEncoder(n_neurons_per_feature={self.n_neurons}, sigma={self.sigma})"


class DeltaEncoder:
    """
    Delta / change encoder — asynchronous, event-driven.

    Produces a spike only when the input changes by more than `threshold`.
    This is how biological sensory systems work and is ideal for:
    - IoT sensors
    - DVS (Dynamic Vision Sensors) cameras
    - Neuromorphic edge computing

    Dramatically reduces spike count (and energy) for slowly-changing signals.

    Parameters
    ----------
    threshold : float
        Minimum change to trigger a spike (default: 0.1).
    time_steps : int
        Output window length.
    """

    def __init__(self, threshold: float = 0.1, time_steps: int = 100):
        self.threshold = threshold
        self.time_steps = time_steps
        self._last_value = None

    def encode_series(self, time_series: np.ndarray) -> np.ndarray:
        """
        Encode a time series as ON/OFF change events.

        Parameters
        ----------
        time_series : np.ndarray, shape (T, n_features)
            Sensor readings over time.

        Returns
        -------
        spikes : np.ndarray, shape (T, 2 * n_features)
            ON events (positive change) and OFF events (negative change).
        """
        T, n = time_series.shape
        spikes = np.zeros((T, 2 * n), dtype=np.float32)

        prev = time_series[0].copy()
        for t in range(1, T):
            delta = time_series[t] - prev
            on_events = delta > self.threshold    # positive change
            off_events = delta < -self.threshold  # negative change
            spikes[t, :n] = on_events.astype(np.float32)
            spikes[t, n:] = off_events.astype(np.float32)
            prev = time_series[t].copy()

        return spikes

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a static input relative to last state.
        For streaming use, call encode_series instead.
        """
        if self._last_value is None:
            self._last_value = np.zeros_like(x)

        delta = x - self._last_value
        on = (delta > self.threshold).astype(np.float32)
        off = (delta < -self.threshold).astype(np.float32)
        self._last_value = x.copy()

        spikes_one = np.concatenate([on, off])
        return np.tile(spikes_one, (self.time_steps, 1))

    def reset(self):
        self._last_value = None

    def __repr__(self) -> str:
        return f"DeltaEncoder(threshold={self.threshold}, T={self.time_steps})"
