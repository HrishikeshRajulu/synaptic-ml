"""
SpikingNet — the top-level model class.

Analogous to keras.Sequential or torch.nn.Module, but for spiking neural networks.
"""

import numpy as np
import json
import os
from typing import List, Optional, Union, Literal
from .layers import LIFLayer, AdaptiveLIFLayer, IzhikevichLayer, OutputLayer, InputLayer


class SpikingNet:
    """
    High-level spiking neural network model.

    Provides a simple interface to define, train, and deploy SNNs
    on neuromorphic hardware without a PhD in neuroscience.

    Parameters
    ----------
    layer_sizes : list of int
        Network topology. [784, 256, 10] creates:
        InputLayer(784) -> LIFLayer(256) -> OutputLayer(10)
    neuron : str
        Neuron model: 'lif', 'adaptive_lif', or 'izhikevich'.
    dt : float
        Simulation timestep in ms.
    time_steps : int
        Number of simulation timesteps per inference.

    Examples
    --------
    >>> import synaptic_ml as sml
    >>> model = sml.SpikingNet([784, 256, 10])
    >>> model.train(X_train, y_train, epochs=10)
    >>> predictions = model.predict(X_test)
    >>> model.deploy(target='loihi2')
    """

    def __init__(
        self,
        layer_sizes: List[int],
        neuron: Literal["lif", "adaptive_lif", "izhikevich"] = "lif",
        dt: float = 1.0,
        time_steps: int = 100,
        decoder: str = "rate",
    ):
        if len(layer_sizes) < 2:
            raise ValueError("Need at least 2 layer sizes (input + output).")

        self.layer_sizes = layer_sizes
        self.neuron_type = neuron
        self.dt = dt
        self.time_steps = time_steps

        self.layers = self._build_layers(layer_sizes, neuron, decoder)
        self._total_synaptic_ops = 0
        self._inference_count = 0

    def _build_layers(self, sizes, neuron_type, decoder):
        layers = [InputLayer(sizes[0])]

        for i in range(1, len(sizes) - 1):
            if neuron_type == "lif":
                layers.append(LIFLayer(sizes[i - 1], sizes[i]))
            elif neuron_type == "adaptive_lif":
                layers.append(AdaptiveLIFLayer(sizes[i - 1], sizes[i]))
            elif neuron_type == "izhikevich":
                layers.append(IzhikevichLayer(sizes[i - 1], sizes[i]))
            else:
                raise ValueError(f"Unknown neuron type: {neuron_type}. Choose 'lif', 'adaptive_lif', or 'izhikevich'.")

        layers.append(OutputLayer(sizes[-2], sizes[-1], decoder=decoder))
        return layers

    def _reset_all(self):
        for layer in self.layers:
            layer.reset_state()

    def _forward_one_step(self, spikes: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            spikes = layer.forward(spikes, self.dt)
        return spikes

    def run(self, encoded_input: np.ndarray) -> np.ndarray:
        """
        Run the network for one sample over `time_steps` timesteps.

        Parameters
        ----------
        encoded_input : np.ndarray, shape (time_steps, n_input)
            Pre-encoded spike train.

        Returns
        -------
        class_scores : np.ndarray, shape (n_output,)
        """
        self._reset_all()
        spike_count = 0

        for t in range(self.time_steps):
            x = encoded_input[t] if t < len(encoded_input) else np.zeros(self.layer_sizes[0])
            spikes = x
            for layer in self.layers:
                spikes = layer.forward(spikes, self.dt)
            spike_count += int(np.sum(spikes))

        self._total_synaptic_ops += spike_count
        self._inference_count += 1

        return self.layers[-1].decode()

    def predict(self, X: np.ndarray, encoder=None, verbose: bool = True) -> np.ndarray:
        """
        Predict class labels for a dataset.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data (values in [0, 1]).
        encoder : Encoder, optional
            Spike encoder. Defaults to RateEncoder.
        verbose : bool

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,) of int
        """
        from ..encoding.rate import RateEncoder

        if encoder is None:
            encoder = RateEncoder(time_steps=self.time_steps)

        predictions = []
        n = len(X)

        for i, x in enumerate(X):
            encoded = encoder.encode(x)
            scores = self.run(encoded)
            predictions.append(int(np.argmax(scores)))

            if verbose and (i + 1) % 100 == 0:
                print(f"  Predicted {i+1}/{n}", end="\r")

        if verbose:
            print(f"  Predicted {n}/{n}")

        return np.array(predictions)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        learning_rule: Literal["surrogate", "stdp"] = "surrogate",
        learning_rate: float = 0.001,
        encoder=None,
        verbose: bool = True,
    ):
        """
        Train the network.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
        y_train : np.ndarray, shape (n_samples,) — integer class labels
        epochs : int
        learning_rule : str
            'surrogate' uses backprop with surrogate gradients (recommended).
            'stdp' uses unsupervised Spike-Timing Dependent Plasticity.
        learning_rate : float
        encoder : Encoder, optional
        verbose : bool
        """
        from ..training.trainer import Trainer

        trainer = Trainer(self, learning_rule=learning_rule, learning_rate=learning_rate)
        trainer.fit(X_train, y_train, epochs=epochs, encoder=encoder, verbose=verbose)

    def estimate_energy(self, backend: str = "cpu") -> dict:
        """
        Estimate energy consumption for inferences run so far.

        Returns
        -------
        dict with keys: synaptic_ops, energy_joules, energy_per_inference_nJ,
                        equivalent_gpu_watts, neuromorphic_watts, speedup_factor
        """
        from ..utils.metrics import estimate_energy_joules

        if self._inference_count == 0:
            print("Run some inferences first (model.predict or model.run).")
            return {}

        ops_per_inf = self._total_synaptic_ops / self._inference_count
        energy = estimate_energy_joules(self._total_synaptic_ops, backend)

        # GPU comparison: ~1 TFLOP/s, ~300W, typical image ~1M FLOPs
        n_params = sum(
            layer.weights.size
            for layer in self.layers
            if hasattr(layer, "weights")
        )
        gpu_energy_per_inf = n_params * 2 * 1e-12 * 300  # FLOPs × energy/FLOP

        return {
            "synaptic_ops_total": self._total_synaptic_ops,
            "synaptic_ops_per_inference": ops_per_inf,
            "energy_total_nJ": energy * 1e9,
            "energy_per_inference_nJ": energy * 1e9 / self._inference_count,
            "gpu_energy_per_inference_nJ": gpu_energy_per_inf * 1e9,
            "efficiency_gain": gpu_energy_per_inf / (energy / self._inference_count + 1e-20),
        }

    def deploy(self, target: Literal["cpu", "loihi2", "brainscales"] = "cpu"):
        """
        Deploy this network to neuromorphic hardware (or CPU simulation).

        Parameters
        ----------
        target : str
            'cpu'        — CPU simulation (always available, no hardware needed)
            'loihi2'     — Intel Loihi 2 (requires Intel NxSDK)
            'brainscales' — BrainScaleS-2 (requires PyNN + pynn_brainscales)
        """
        from ..backends import get_backend

        backend = get_backend(target)
        backend.load_network(self)
        print(f"\n[synaptic_ml] Deployed to: {backend}")
        print(backend.get_hardware_info())
        return backend

    def summary(self):
        """Print a summary of the network architecture."""
        print("=" * 52)
        print(f"  SpikingNet  ({self.neuron_type.upper()}, dt={self.dt}ms, T={self.time_steps})")
        print("=" * 52)
        total_params = 0
        for i, layer in enumerate(self.layers):
            params = layer.weights.size if hasattr(layer, "weights") else 0
            total_params += params
            print(f"  [{i}] {str(layer):<40} params={params:,}")
        print("=" * 52)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Neuron type:      {self.neuron_type}")
        print(f"  Timesteps:        {self.time_steps}")
        print("=" * 52)

    def save(self, path: str):
        """Save model weights and config to a .snm file."""
        data = {
            "config": {
                "layer_sizes": self.layer_sizes,
                "neuron_type": self.neuron_type,
                "dt": self.dt,
                "time_steps": self.time_steps,
            },
            "weights": [
                layer.weights.tolist()
                for layer in self.layers
                if hasattr(layer, "weights")
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[synaptic_ml] Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SpikingNet":
        """Load a saved .snm model."""
        with open(path, "r") as f:
            data = json.load(f)

        cfg = data["config"]
        model = cls(
            layer_sizes=cfg["layer_sizes"],
            neuron=cfg["neuron_type"],
            dt=cfg["dt"],
            time_steps=cfg["time_steps"],
        )

        weight_layers = [l for l in model.layers if hasattr(l, "weights")]
        for layer, w in zip(weight_layers, data["weights"]):
            layer.weights = np.array(w, dtype=np.float32)

        print(f"[synaptic_ml] Model loaded from {path}")
        return model

    def __repr__(self) -> str:
        topology = " -> ".join(str(s) for s in self.layer_sizes)
        return f"SpikingNet([{topology}], neuron={self.neuron_type}, dt={self.dt}ms, T={self.time_steps})"
