"""
synaptic_ml — The TensorFlow for Neuromorphic Computing.

High-level framework for programming spiking neural networks on
neuromorphic hardware (Intel Loihi 2, BrainScaleS, etc.) without
a neuroscience PhD.

Quick Start
-----------
>>> import synaptic_ml as sml
>>>
>>> # Build a spiking network
>>> model = sml.SpikingNet([784, 256, 10])
>>> model.summary()
>>>
>>> # Train with surrogate gradients
>>> model.train(X_train, y_train, epochs=10)
>>>
>>> # Evaluate
>>> predictions = model.predict(X_test)
>>>
>>> # See energy advantage
>>> print(sml.energy_comparison_table(model))
>>>
>>> # Deploy to neuromorphic hardware
>>> model.deploy(target='loihi2')   # Intel Loihi 2
>>> model.deploy(target='cpu')      # CPU simulation (default)

Neuron Models
-------------
- LIF: Leaky Integrate-and-Fire (default, fast)
- AdaptiveLIF: LIF + spike-frequency adaptation
- Izhikevich: biologically rich, many firing patterns

Encoders
--------
- RateEncoder: Poisson spikes proportional to value (default)
- TemporalEncoder: time-to-first-spike (most energy efficient)
- PopulationEncoder: Gaussian tuning curves (most biologically realistic)
- DeltaEncoder: event-driven change detection (ideal for IoT)

Backends
--------
- cpu: pure numpy simulation (always works)
- loihi2: Intel Loihi 2 (requires NxSDK + hardware)
- brainscales: BrainScaleS-2 (requires pynn_brainscales + EBRAINS access)
"""

__version__ = "0.1.3"
__author__ = "synaptic_ml contributors"

# Core
from .core.network import SpikingNet
from .core.layers import LIFLayer, AdaptiveLIFLayer, IzhikevichLayer, InputLayer, OutputLayer
from .core.neurons import LIFNeuron, AdaptiveLIFNeuron, IzhikevichNeuron
from .core.synapses import DenseSynapse, SparseSynapse, DelayedSynapse

# Encoding
from .encoding import RateEncoder, TemporalEncoder, PopulationEncoder, DeltaEncoder, encode

# Learning
from .learning import (
    STDPRule, RewardModulatedSTDP,
    SigmoidSurrogate, FastSigmoidSurrogate, get_surrogate,
    convert_from_pytorch, convert_from_keras, convert_from_numpy_weights,
)

# Training
from .training import Trainer

# Backends
from .backends import CPUBackend, Loihi2Backend, BrainScalesBackend, AkidaBackend, get_backend

# Utils
from .utils import (
    spike_rate, estimate_energy_joules, van_rossum_distance,
    coincidence_factor, energy_comparison_table,
    plot_raster, plot_membrane, plot_weight_distribution,
    plot_energy_comparison, plot_training_history,
)

__all__ = [
    # Core
    "SpikingNet",
    "LIFLayer", "AdaptiveLIFLayer", "IzhikevichLayer", "InputLayer", "OutputLayer",
    "LIFNeuron", "AdaptiveLIFNeuron", "IzhikevichNeuron",
    "DenseSynapse", "SparseSynapse", "DelayedSynapse",
    # Encoding
    "RateEncoder", "TemporalEncoder", "PopulationEncoder", "DeltaEncoder", "encode",
    # Learning
    "STDPRule", "RewardModulatedSTDP",
    "SigmoidSurrogate", "FastSigmoidSurrogate", "get_surrogate",
    "convert_from_pytorch", "convert_from_keras", "convert_from_numpy_weights",
    # Training
    "Trainer",
    # Backends
    "CPUBackend", "Loihi2Backend", "BrainScalesBackend", "AkidaBackend", "get_backend",
    # Utils
    "spike_rate", "estimate_energy_joules", "van_rossum_distance",
    "coincidence_factor", "energy_comparison_table",
    "plot_raster", "plot_membrane", "plot_weight_distribution",
    "plot_energy_comparison", "plot_training_history",
]
