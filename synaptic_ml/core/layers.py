"""
Spiking neural network layers — composable building blocks.
"""

import numpy as np
from typing import Optional, List, Literal
from .neurons import LIFNeuron, AdaptiveLIFNeuron, IzhikevichNeuron, NEURON_REGISTRY
from .synapses import DenseSynapse


class BaseLayer:
    """Abstract base for all spiking layers."""

    def forward(self, spikes_in: np.ndarray, dt: float = 1.0) -> np.ndarray:
        raise NotImplementedError

    def reset_state(self):
        raise NotImplementedError

    @property
    def n_neurons(self) -> int:
        raise NotImplementedError


class InputLayer(BaseLayer):
    """
    Pass-through input layer. Accepts pre-encoded spike trains.

    Parameters
    ----------
    n_neurons : int
        Number of input neurons (= encoded input dimensionality).
    """

    def __init__(self, n_neurons: int):
        self._n = n_neurons
        self.last_spikes = np.zeros(n_neurons, dtype=np.float32)

    def forward(self, spikes_in: np.ndarray, dt: float = 1.0) -> np.ndarray:
        self.last_spikes = spikes_in.astype(np.float32)
        return self.last_spikes

    def reset_state(self):
        self.last_spikes = np.zeros(self._n, dtype=np.float32)

    @property
    def n_neurons(self) -> int:
        return self._n

    def __repr__(self) -> str:
        return f"InputLayer(n={self._n})"


class LIFLayer(BaseLayer):
    """
    Layer of Leaky Integrate-and-Fire neurons.

    Takes presynaptic spikes, computes synaptic currents, steps neurons.

    Parameters
    ----------
    n_pre : int
        Number of inputs (presynaptic neurons).
    n_neurons : int
        Number of LIF neurons in this layer.
    record : bool
        Whether to record membrane potentials and spikes over time.
    """

    def __init__(
        self,
        n_pre: int,
        n_neurons: int,
        record: bool = True,
        synapse_init: str = "xavier",
        **neuron_kwargs,
    ):
        self._n_pre = n_pre
        self._n = n_neurons
        self.synapse = DenseSynapse(n_pre, n_neurons, init=synapse_init)
        self.neurons = LIFNeuron(n_neurons, **neuron_kwargs)
        self.record = record

        self.spike_history: List[np.ndarray] = []
        self.voltage_history: List[np.ndarray] = []
        self.last_spikes = np.zeros(n_neurons, dtype=np.float32)

    def forward(self, spikes_in: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Propagate spikes through synapses, then step neurons.

        Parameters
        ----------
        spikes_in : np.ndarray, shape (n_pre,)
        dt : float

        Returns
        -------
        spikes_out : np.ndarray, shape (n_neurons,)
        """
        current = self.synapse.forward(spikes_in)
        voltage, spikes = self.neurons.step(current, dt)
        self.last_spikes = spikes

        if self.record:
            self.spike_history.append(spikes.copy())
            self.voltage_history.append(voltage.copy())

        return spikes

    def reset_state(self):
        self.neurons.reset_state()
        self.last_spikes = np.zeros(self._n, dtype=np.float32)
        self.spike_history.clear()
        self.voltage_history.clear()

    @property
    def weights(self) -> np.ndarray:
        return self.synapse.weights

    @weights.setter
    def weights(self, w: np.ndarray):
        self.synapse.weights = w

    @property
    def n_neurons(self) -> int:
        return self._n

    @property
    def spike_trains(self) -> np.ndarray:
        """Return recorded spikes as array of shape (T, n_neurons)."""
        return np.array(self.spike_history, dtype=np.float32)

    @property
    def voltage_traces(self) -> np.ndarray:
        """Return recorded membrane potentials as array of shape (T, n_neurons)."""
        return np.array(self.voltage_history, dtype=np.float32)

    def __repr__(self) -> str:
        return f"LIFLayer({self._n_pre} -> {self._n}, {self.neurons})"


class AdaptiveLIFLayer(BaseLayer):
    """Layer of Adaptive LIF neurons (spike-frequency adaptation)."""

    def __init__(self, n_pre: int, n_neurons: int, record: bool = True, **neuron_kwargs):
        self._n_pre = n_pre
        self._n = n_neurons
        self.synapse = DenseSynapse(n_pre, n_neurons)
        self.neurons = AdaptiveLIFNeuron(n_neurons, **neuron_kwargs)
        self.record = record
        self.spike_history: List[np.ndarray] = []
        self.voltage_history: List[np.ndarray] = []
        self.last_spikes = np.zeros(n_neurons, dtype=np.float32)

    def forward(self, spikes_in: np.ndarray, dt: float = 1.0) -> np.ndarray:
        current = self.synapse.forward(spikes_in)
        voltage, spikes = self.neurons.step(current, dt)
        self.last_spikes = spikes
        if self.record:
            self.spike_history.append(spikes.copy())
            self.voltage_history.append(voltage.copy())
        return spikes

    def reset_state(self):
        self.neurons.reset_state()
        self.last_spikes = np.zeros(self._n, dtype=np.float32)
        self.spike_history.clear()
        self.voltage_history.clear()

    @property
    def n_neurons(self) -> int:
        return self._n

    @property
    def weights(self) -> np.ndarray:
        return self.synapse.weights

    @weights.setter
    def weights(self, w: np.ndarray):
        self.synapse.weights = w

    def __repr__(self) -> str:
        return f"AdaptiveLIFLayer({self._n_pre} -> {self._n})"


class IzhikevichLayer(BaseLayer):
    """Layer of Izhikevich neurons — can model many biological neuron types."""

    def __init__(self, n_pre: int, n_neurons: int, preset: str = "regular_spiking", record: bool = True):
        self._n_pre = n_pre
        self._n = n_neurons
        self.synapse = DenseSynapse(n_pre, n_neurons)
        self.neurons = IzhikevichNeuron(n_neurons, preset=preset)
        self.record = record
        self.spike_history: List[np.ndarray] = []
        self.last_spikes = np.zeros(n_neurons, dtype=np.float32)

    def forward(self, spikes_in: np.ndarray, dt: float = 1.0) -> np.ndarray:
        current = self.synapse.forward(spikes_in) * 10.0  # scale for Izhikevich units
        voltage, spikes = self.neurons.step(current, dt)
        self.last_spikes = spikes
        if self.record:
            self.spike_history.append(spikes.copy())
        return spikes

    def reset_state(self):
        self.neurons.reset_state()
        self.last_spikes = np.zeros(self._n, dtype=np.float32)
        self.spike_history.clear()

    @property
    def n_neurons(self) -> int:
        return self._n

    @property
    def weights(self) -> np.ndarray:
        return self.synapse.weights

    def __repr__(self) -> str:
        return f"IzhikevichLayer({self._n_pre} -> {self._n}, preset='{self.neurons.preset}')"


class OutputLayer(BaseLayer):
    """
    Output layer: integrates spikes over time and decodes to class probabilities.

    Parameters
    ----------
    n_pre : int
    n_neurons : int
        Number of output classes.
    decoder : str
        'rate' — use mean firing rate. 'max_voltage' — track max membrane potential.
    """

    def __init__(self, n_pre: int, n_neurons: int, decoder: Literal["rate", "max_voltage"] = "rate"):
        self._n_pre = n_pre
        self._n = n_neurons
        self.synapse = DenseSynapse(n_pre, n_neurons)
        self.neurons = LIFNeuron(n_neurons)
        self.decoder = decoder

        self.spike_accumulator = np.zeros(n_neurons, dtype=np.float32)
        self.max_voltage = np.full(n_neurons, -np.inf, dtype=np.float32)
        self.current_accum = np.zeros(n_neurons, dtype=np.float32)
        self.spike_count = 0
        self.last_spikes = np.zeros(n_neurons, dtype=np.float32)

    def forward(self, spikes_in: np.ndarray, dt: float = 1.0) -> np.ndarray:
        current = self.synapse.forward(spikes_in)
        voltage, spikes = self.neurons.step(current, dt)
        self.last_spikes = spikes
        self.spike_accumulator += spikes
        self.max_voltage = np.maximum(self.max_voltage, voltage)
        self.current_accum += current  # track total synaptic input
        self.spike_count += 1
        return spikes

    def decode(self) -> np.ndarray:
        """Return class scores. Uses integrated current as fallback when neurons don't fire."""
        if self.decoder == "rate":
            if self.spike_accumulator.sum() > 0:
                return self.spike_accumulator / (self.spike_count + 1e-8)
            # Fallback: integrated current reflects weights even without spikes
            return self.current_accum / (self.spike_count + 1e-8)
        elif self.decoder == "max_voltage":
            # Use max_voltage only if neurons actually moved
            if not np.all(self.max_voltage == -np.inf) and not np.all(self.max_voltage == self.max_voltage[0]):
                return self.max_voltage
            # Fallback: integrated current
            return self.current_accum / (self.spike_count + 1e-8)
        else:
            raise ValueError(f"Unknown decoder: {self.decoder}")

    def predict_class(self) -> int:
        return int(np.argmax(self.decode()))

    def reset_state(self):
        self.neurons.reset_state()
        self.spike_accumulator = np.zeros(self._n, dtype=np.float32)
        self.max_voltage = np.full(self._n, -np.inf, dtype=np.float32)
        self.current_accum = np.zeros(self._n, dtype=np.float32)
        self.spike_count = 0
        self.last_spikes = np.zeros(self._n, dtype=np.float32)

    @property
    def n_neurons(self) -> int:
        return self._n

    @property
    def weights(self) -> np.ndarray:
        return self.synapse.weights

    @weights.setter
    def weights(self, w: np.ndarray):
        self.synapse.weights = w

    def __repr__(self) -> str:
        return f"OutputLayer({self._n_pre} -> {self._n}, decoder='{self.decoder}')"
