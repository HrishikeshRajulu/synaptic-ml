"""
Synapse models — weighted connections between neuron populations.
"""

import numpy as np
from typing import Optional, Literal


def _xavier_init(n_pre: int, n_post: int) -> np.ndarray:
    # Scale up by 3x vs standard xavier so neurons actually fire initially
    limit = np.sqrt(6.0 / (n_pre + n_post)) * 3.0
    return np.random.uniform(-limit, limit, (n_pre, n_post)).astype(np.float32)


def _gaussian_init(n_pre: int, n_post: int, scale: float = 0.1) -> np.ndarray:
    return np.random.normal(0, scale, (n_pre, n_post)).astype(np.float32)


def _uniform_init(n_pre: int, n_post: int, low: float = 0.0, high: float = 0.5) -> np.ndarray:
    return np.random.uniform(low, high, (n_pre, n_post)).astype(np.float32)


class DenseSynapse:
    """
    Fully-connected synapse between two neuron populations.

    Stores a weight matrix W of shape (n_pre, n_post).
    Forward pass: post_current = pre_spikes @ W

    Parameters
    ----------
    n_pre : int
        Number of presynaptic neurons.
    n_post : int
        Number of postsynaptic neurons.
    init : str
        Weight initialization: 'xavier', 'gaussian', or 'uniform'.
    w_min, w_max : float
        Weight clipping bounds.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        init: Literal["xavier", "gaussian", "uniform"] = "xavier",
        w_min: float = -1.0,
        w_max: float = 1.0,
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.w_min = w_min
        self.w_max = w_max

        if init == "xavier":
            self.weights = _xavier_init(n_pre, n_post)
        elif init == "gaussian":
            self.weights = _gaussian_init(n_pre, n_post)
        elif init == "uniform":
            self.weights = _uniform_init(n_pre, n_post)
        else:
            raise ValueError(f"Unknown init: {init}. Choose 'xavier', 'gaussian', or 'uniform'.")

        np.clip(self.weights, w_min, w_max, out=self.weights)

    def forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        """
        Compute postsynaptic currents from presynaptic spikes.

        Parameters
        ----------
        pre_spikes : np.ndarray, shape (n_pre,) or (batch, n_pre)

        Returns
        -------
        np.ndarray, shape (n_post,) or (batch, n_post)
        """
        return pre_spikes @ self.weights

    def update_weights(self, delta_w: np.ndarray):
        """Apply weight update and clip to [w_min, w_max]."""
        self.weights += delta_w
        np.clip(self.weights, self.w_min, self.w_max, out=self.weights)

    @property
    def shape(self):
        return self.weights.shape

    def __repr__(self) -> str:
        return f"DenseSynapse({self.n_pre} -> {self.n_post}, w∈[{self.w_min},{self.w_max}])"


class SparseSynapse:
    """
    Sparse synapse for large networks where most connections are zero.

    Uses COO (coordinate) format internally, converts to dense for computation.
    Suitable for networks with <5% connectivity.

    Parameters
    ----------
    n_pre, n_post : int
    connectivity : float
        Fraction of connections to create (default: 0.1 = 10%).
    """

    def __init__(self, n_pre: int, n_post: int, connectivity: float = 0.1, w_min: float = -1.0, w_max: float = 1.0):
        self.n_pre = n_pre
        self.n_post = n_post
        self.w_min = w_min
        self.w_max = w_max

        n_connections = int(n_pre * n_post * connectivity)
        self.rows = np.random.randint(0, n_pre, n_connections)
        self.cols = np.random.randint(0, n_post, n_connections)
        self.values = np.random.uniform(0.0, 0.3, n_connections).astype(np.float32)

        # Build dense weight matrix from sparse
        self.weights = np.zeros((n_pre, n_post), dtype=np.float32)
        self.weights[self.rows, self.cols] = self.values

    def forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        return pre_spikes @ self.weights

    def update_weights(self, delta_w: np.ndarray):
        self.weights += delta_w
        np.clip(self.weights, self.w_min, self.w_max, out=self.weights)

    def __repr__(self) -> str:
        n_conn = len(self.values)
        pct = 100 * n_conn / (self.n_pre * self.n_post)
        return f"SparseSynapse({self.n_pre} -> {self.n_post}, {pct:.1f}% connected)"


class DelayedSynapse:
    """
    Synapse with axonal transmission delay via a ring buffer.

    Spikes travel from pre to post with a fixed delay in timesteps.
    Biologically, axons have different conduction velocities.

    Parameters
    ----------
    n_pre, n_post : int
    delay : int
        Delay in timesteps (default: 5).
    """

    def __init__(self, n_pre: int, n_post: int, delay: int = 5, init: str = "xavier"):
        self.inner = DenseSynapse(n_pre, n_post, init=init)
        self.delay = delay
        self.buffer = np.zeros((delay, n_pre), dtype=np.float32)
        self._ptr = 0

    @property
    def weights(self):
        return self.inner.weights

    def forward(self, pre_spikes: np.ndarray) -> np.ndarray:
        # Write current spikes into buffer
        self.buffer[self._ptr] = pre_spikes
        # Read delayed spikes
        read_ptr = (self._ptr + 1) % self.delay
        delayed = self.buffer[read_ptr]
        self._ptr = (self._ptr + 1) % self.delay
        return self.inner.forward(delayed)

    def update_weights(self, delta_w: np.ndarray):
        self.inner.update_weights(delta_w)

    def reset(self):
        self.buffer[:] = 0
        self._ptr = 0

    def __repr__(self) -> str:
        return f"DelayedSynapse({self.inner.n_pre} -> {self.inner.n_post}, delay={self.delay})"
