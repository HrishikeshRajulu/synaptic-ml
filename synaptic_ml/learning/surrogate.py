"""
Surrogate gradient methods for training SNNs with backpropagation.

The spike function is a step function — not differentiable.
Surrogate gradients replace the true derivative with a smooth
approximation during the backward pass only.

This lets us use standard backprop to train SNNs efficiently.
Reference: Neftci et al., "Surrogate Gradient Learning in Spiking Neural Networks" (2019)
"""

import numpy as np
from typing import Tuple


class SurrogateGradient:
    """Base class for surrogate gradient functions."""

    def forward(self, v: np.ndarray, threshold: float) -> np.ndarray:
        """Forward pass: spike if v >= threshold (Heaviside step function)."""
        return (v >= threshold).astype(np.float32)

    def backward(self, v: np.ndarray, threshold: float) -> np.ndarray:
        """Backward pass: surrogate derivative (smooth approximation)."""
        raise NotImplementedError


class SigmoidSurrogate(SurrogateGradient):
    """
    Sigmoid surrogate gradient.

    Forward:  H(v - threshold)  [Heaviside]
    Backward: sigmoid'(β * (v - threshold))

    Parameters
    ----------
    beta : float
        Sharpness of the surrogate. Higher = closer to true step. (default: 5.0)
    """

    def __init__(self, beta: float = 5.0):
        self.beta = beta

    def backward(self, v: np.ndarray, threshold: float) -> np.ndarray:
        x = self.beta * (v - threshold)
        s = 1.0 / (1.0 + np.exp(-x))
        return self.beta * s * (1.0 - s)

    def __repr__(self) -> str:
        return f"SigmoidSurrogate(beta={self.beta})"


class PiecewiseLinearSurrogate(SurrogateGradient):
    """
    Piecewise linear (triangular) surrogate gradient.

    Simple and fast. Derivative = max(0, 1 - |v - threshold| / width).

    Parameters
    ----------
    width : float
        Half-width of the triangular function (default: 1.0).
    """

    def __init__(self, width: float = 1.0):
        self.width = width

    def backward(self, v: np.ndarray, threshold: float) -> np.ndarray:
        return np.maximum(0.0, 1.0 - np.abs(v - threshold) / self.width).astype(np.float32)

    def __repr__(self) -> str:
        return f"PiecewiseLinearSurrogate(width={self.width})"


class FastSigmoidSurrogate(SurrogateGradient):
    """
    Fast sigmoid surrogate (used in snnTorch).

    Computationally cheaper than standard sigmoid.
    Backward: 1 / (1 + |β * (v - threshold)|)²
    """

    def __init__(self, beta: float = 10.0):
        self.beta = beta

    def backward(self, v: np.ndarray, threshold: float) -> np.ndarray:
        return 1.0 / (1.0 + np.abs(self.beta * (v - threshold))) ** 2

    def __repr__(self) -> str:
        return f"FastSigmoidSurrogate(beta={self.beta})"


class SuperSpike(SurrogateGradient):
    """
    SuperSpike surrogate gradient (Zenke & Ganguli, 2018).

    Normalized negative derivative of a fast sigmoid.
    Used in biologically-inspired online learning.
    """

    def __init__(self, beta: float = 100.0):
        self.beta = beta

    def backward(self, v: np.ndarray, threshold: float) -> np.ndarray:
        return 1.0 / (self.beta * np.abs(v - threshold) + 1.0) ** 2

    def __repr__(self) -> str:
        return f"SuperSpike(beta={self.beta})"


SURROGATE_REGISTRY = {
    "sigmoid": SigmoidSurrogate,
    "piecewise_linear": PiecewiseLinearSurrogate,
    "fast_sigmoid": FastSigmoidSurrogate,
    "superspike": SuperSpike,
}


def get_surrogate(name: str = "fast_sigmoid", **kwargs) -> SurrogateGradient:
    """Get a surrogate gradient by name."""
    if name not in SURROGATE_REGISTRY:
        raise ValueError(f"Unknown surrogate: {name}. Choose from {list(SURROGATE_REGISTRY.keys())}")
    return SURROGATE_REGISTRY[name](**kwargs)
