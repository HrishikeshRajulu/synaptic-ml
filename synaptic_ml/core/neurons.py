"""
Neuron models for spiking neural networks.

All neuron classes operate on arrays of N neurons simultaneously (vectorized).
Each implements a `step(input_current, dt)` method returning (voltage, spike).
"""

import numpy as np
from typing import Tuple, Optional


class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) neuron model.

    The workhorse of neuromorphic computing. Simple, efficient, and
    biologically plausible. Models the membrane as a leaky capacitor.

    dV/dt = -(V - V_rest) / tau_m + R * I

    Parameters
    ----------
    n_neurons : int
        Number of neurons in this population.
    tau_m : float
        Membrane time constant in ms (default: 20ms).
    v_rest : float
        Resting membrane potential in mV (default: -65mV).
    v_thresh : float
        Spike threshold in mV (default: -50mV).
    v_reset : float
        Reset potential after spike in mV (default: -65mV).
    R : float
        Membrane resistance (default: 1.0).
    refractory_period : float
        Refractory period in ms (default: 2ms).
    """

    def __init__(
        self,
        n_neurons: int,
        tau_m: float = 20.0,
        v_rest: float = -65.0,
        v_thresh: float = -50.0,
        v_reset: float = -65.0,
        R: float = 1.0,
        refractory_period: float = 2.0,
    ):
        self.n_neurons = n_neurons
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.R = R
        self.refractory_period = refractory_period

        self.reset_state()

    def reset_state(self):
        """Reset membrane potentials and refractory counters to initial state."""
        self.v = np.full(self.n_neurons, self.v_rest, dtype=np.float32)
        self.refractory_count = np.zeros(self.n_neurons, dtype=np.float32)

    def step(self, input_current: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance neuron state by one timestep.

        Parameters
        ----------
        input_current : np.ndarray, shape (n_neurons,)
            Input current to each neuron.
        dt : float
            Timestep size in ms.

        Returns
        -------
        v : np.ndarray
            Membrane potentials after this step.
        spikes : np.ndarray of bool
            Boolean spike array (True where neuron fired).
        """
        # Neurons in refractory period don't integrate
        in_refractory = self.refractory_count > 0
        self.refractory_count = np.maximum(0.0, self.refractory_count - dt)

        # Euler integration of membrane equation
        dv = (-(self.v - self.v_rest) / self.tau_m + self.R * input_current) * dt
        self.v = np.where(in_refractory, self.v_reset, self.v + dv)

        # Detect spikes
        spikes = self.v >= self.v_thresh

        # Reset spiked neurons and start refractory period
        self.v = np.where(spikes, self.v_reset, self.v)
        self.refractory_count = np.where(spikes, self.refractory_period, self.refractory_count)

        return self.v.copy(), spikes.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"LIFNeuron(n={self.n_neurons}, tau_m={self.tau_m}ms, "
            f"v_thresh={self.v_thresh}mV, refractory={self.refractory_period}ms)"
        )


class AdaptiveLIFNeuron:
    """
    Adaptive Leaky Integrate-and-Fire neuron.

    Extends LIF with a spike-frequency adaptation current `w` that
    accumulates with each spike and slows down firing — matching
    biological cortical neurons more closely.

    dV/dt = -(V - V_rest) / tau_m + R * (I - w)
    dw/dt = -w / tau_w  [+ b on each spike]

    Parameters
    ----------
    n_neurons : int
    tau_m : float
        Membrane time constant in ms.
    tau_w : float
        Adaptation time constant in ms (default: 100ms).
    b : float
        Adaptation increment per spike (default: 0.1).
    """

    def __init__(
        self,
        n_neurons: int,
        tau_m: float = 20.0,
        v_rest: float = -65.0,
        v_thresh: float = -50.0,
        v_reset: float = -65.0,
        R: float = 1.0,
        tau_w: float = 100.0,
        b: float = 0.1,
        refractory_period: float = 2.0,
    ):
        self.n_neurons = n_neurons
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.R = R
        self.tau_w = tau_w
        self.b = b
        self.refractory_period = refractory_period
        self.reset_state()

    def reset_state(self):
        self.v = np.full(self.n_neurons, self.v_rest, dtype=np.float32)
        self.w = np.zeros(self.n_neurons, dtype=np.float32)
        self.refractory_count = np.zeros(self.n_neurons, dtype=np.float32)

    def step(self, input_current: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        in_refractory = self.refractory_count > 0
        self.refractory_count = np.maximum(0.0, self.refractory_count - dt)

        # Adaptation decay
        dw = (-self.w / self.tau_w) * dt
        self.w += dw

        # Membrane integration with adaptation
        dv = (-(self.v - self.v_rest) / self.tau_m + self.R * (input_current - self.w)) * dt
        self.v = np.where(in_refractory, self.v_reset, self.v + dv)

        spikes = self.v >= self.v_thresh

        # Spike: reset voltage, increment adaptation
        self.v = np.where(spikes, self.v_reset, self.v)
        self.w += np.where(spikes, self.b, 0.0)
        self.refractory_count = np.where(spikes, self.refractory_period, self.refractory_count)

        return self.v.copy(), spikes.astype(np.float32)

    def __repr__(self) -> str:
        return f"AdaptiveLIFNeuron(n={self.n_neurons}, tau_m={self.tau_m}ms, tau_w={self.tau_w}ms, b={self.b})"


class IzhikevichNeuron:
    """
    Izhikevich neuron model.

    A 2-variable model that reproduces ~20 known firing patterns of
    biological neurons with low computational cost.

    dv/dt = 0.04*v² + 5*v + 140 - u + I
    du/dt = a * (b*v - u)
    if v >= 30: v = c, u = u + d

    Parameters
    ----------
    n_neurons : int
    preset : str
        One of 'regular_spiking', 'fast_spiking', 'bursting',
        'chattering', 'intrinsically_bursting'.
    """

    PRESETS = {
        "regular_spiking":        {"a": 0.02, "b": 0.2,  "c": -65.0, "d": 8.0},
        "fast_spiking":           {"a": 0.1,  "b": 0.2,  "c": -65.0, "d": 2.0},
        "bursting":               {"a": 0.02, "b": 0.2,  "c": -50.0, "d": 2.0},
        "chattering":             {"a": 0.02, "b": 0.2,  "c": -50.0, "d": 2.0},
        "intrinsically_bursting": {"a": 0.02, "b": 0.2,  "c": -55.0, "d": 4.0},
        "thalamo_cortical":       {"a": 0.02, "b": 0.25, "c": -65.0, "d": 0.05},
    }

    def __init__(self, n_neurons: int, preset: str = "regular_spiking", **kwargs):
        self.n_neurons = n_neurons
        params = {**self.PRESETS[preset], **kwargs}
        self.a = params["a"]
        self.b = params["b"]
        self.c = params["c"]
        self.d = params["d"]
        self.preset = preset
        self.reset_state()

    def reset_state(self):
        self.v = np.full(self.n_neurons, -65.0, dtype=np.float32)
        self.u = self.b * self.v

    def step(self, input_current: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        # Izhikevich uses 0.5ms sub-steps for numerical stability
        for _ in range(int(dt / 0.5)):
            dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + input_current) * 0.5
            du = (self.a * (self.b * self.v - self.u)) * 0.5
            self.v += dv
            self.u += du

        spikes = self.v >= 30.0
        self.v = np.where(spikes, self.c, self.v)
        self.u = np.where(spikes, self.u + self.d, self.u)

        return self.v.copy(), spikes.astype(np.float32)

    def __repr__(self) -> str:
        return f"IzhikevichNeuron(n={self.n_neurons}, preset='{self.preset}')"


NEURON_REGISTRY = {
    "lif": LIFNeuron,
    "adaptive_lif": AdaptiveLIFNeuron,
    "izhikevich": IzhikevichNeuron,
}
