"""
ANN-to-SNN Conversion — the fast track to neuromorphic deployment.

Convert a trained PyTorch or Keras model to a SpikingNet using
threshold balancing. This lets you:
1. Train with standard deep learning tools (fast, well-understood)
2. Convert to SNN for energy-efficient neuromorphic deployment

Method: threshold balancing (Diehl et al., 2015)
- Run calibration data through the ANN
- Set each layer's threshold so max activation = max spike rate
- Scale weights accordingly
"""

import numpy as np
from typing import Optional, List


def convert_from_numpy_weights(
    weights: List[np.ndarray],
    biases: Optional[List[np.ndarray]],
    calibration_data: np.ndarray,
    time_steps: int = 100,
    percentile: float = 99.9,
) -> "SpikingNet":
    """
    Convert a list of weight matrices to a SpikingNet using threshold balancing.

    Parameters
    ----------
    weights : list of np.ndarray
        Weight matrices for each layer, shape (n_pre, n_post).
    biases : list of np.ndarray or None
        Bias vectors per layer.
    calibration_data : np.ndarray
        A sample of input data used to calibrate thresholds, shape (N, n_input).
    time_steps : int
        Simulation timesteps for the output SNN.
    percentile : float
        Percentile of activations to use as threshold (default: 99.9).

    Returns
    -------
    SpikingNet
    """
    from ..core.network import SpikingNet

    n_layers = len(weights)
    layer_sizes = [weights[0].shape[0]] + [w.shape[1] for w in weights]

    # Forward pass through ANN to collect activations
    activations = [calibration_data.copy()]
    current = calibration_data.astype(np.float32)

    for i, w in enumerate(weights):
        current = current @ w
        if biases is not None and biases[i] is not None:
            current += biases[i]
        current = np.maximum(0, current)  # ReLU
        activations.append(current)

    # Compute threshold for each layer (threshold balancing)
    thresholds = []
    for act in activations[1:]:
        thresh = np.percentile(act, percentile)
        thresholds.append(max(thresh, 1e-6))

    # Build SpikingNet and scale weights
    model = SpikingNet(layer_sizes, neuron="lif", time_steps=time_steps)

    weight_layers = [l for l in model.layers if hasattr(l, "weights")]
    for i, (layer, w, thresh) in enumerate(zip(weight_layers, weights, thresholds)):
        # Scale weights: divide by threshold so max activation ≈ 1 spike/timestep
        prev_thresh = thresholds[i - 1] if i > 0 else 1.0
        scaled_w = w * (prev_thresh / thresh)
        layer.weights = scaled_w.astype(np.float32)

    print(f"[synaptic_ml] ANN->SNN conversion complete.")
    print(f"  Layer sizes: {layer_sizes}")
    print(f"  Thresholds: {[f'{t:.4f}' for t in thresholds]}")

    return model


def convert_from_pytorch(
    model,
    calibration_data: np.ndarray,
    time_steps: int = 100,
    percentile: float = 99.9,
) -> "SpikingNet":
    """
    Convert a PyTorch Sequential model to SpikingNet.

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch model with Linear layers (and optionally ReLU activations).
    calibration_data : np.ndarray, shape (N, n_features)
    time_steps : int
    percentile : float

    Returns
    -------
    SpikingNet

    Examples
    --------
    >>> import torch.nn as nn
    >>> ann = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    >>> snn = convert_from_pytorch(ann, X_calibration)
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch not found. Install with: pip install torch\n"
            "Or use convert_from_numpy_weights() with raw weight arrays."
        )

    weights = []
    biases = []

    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name == "Linear":
            w = module.weight.detach().numpy().T  # PyTorch stores as (out, in)
            b = module.bias.detach().numpy() if module.bias is not None else None
            weights.append(w)
            biases.append(b)

    return convert_from_numpy_weights(weights, biases, calibration_data, time_steps, percentile)


def convert_from_keras(
    model,
    calibration_data: np.ndarray,
    time_steps: int = 100,
    percentile: float = 99.9,
) -> "SpikingNet":
    """
    Convert a Keras/TensorFlow Sequential model to SpikingNet.

    Parameters
    ----------
    model : keras.Model
        A Keras model with Dense layers.
    calibration_data : np.ndarray
    time_steps : int
    percentile : float

    Returns
    -------
    SpikingNet

    Examples
    --------
    >>> from tensorflow import keras
    >>> ann = keras.Sequential([keras.layers.Dense(256, activation='relu'), keras.layers.Dense(10)])
    >>> snn = convert_from_keras(ann, X_calibration)
    """
    weights = []
    biases = []

    for layer in model.layers:
        cls_name = type(layer).__name__
        if cls_name == "Dense":
            layer_weights = layer.get_weights()
            w = layer_weights[0]  # (in, out) — already correct orientation
            b = layer_weights[1] if len(layer_weights) > 1 else None
            weights.append(w)
            biases.append(b)

    return convert_from_numpy_weights(weights, biases, calibration_data, time_steps, percentile)
