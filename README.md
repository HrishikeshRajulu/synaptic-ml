# synaptic_ml

**The TensorFlow for Neuromorphic Computing.**

Train spiking neural networks. Deploy to neuromorphic chips. Use 1000× less energy than GPU.

```python
import synaptic_ml as sml

# Build a spiking network
model = sml.SpikingNet([784, 256, 10])
model.summary()

# Train with surrogate gradients (like backprop, but for spikes)
model.train(X_train, y_train, epochs=10)

# Evaluate
predictions = model.predict(X_test)

# See the energy advantage
print(sml.energy_comparison_table(model))

# Deploy to neuromorphic hardware
model.deploy(target='loihi2')      # Intel Loihi 2
model.deploy(target='brainscales') # BrainScaleS-2
model.deploy(target='cpu')         # CPU simulation (default, always works)
```

---

## Why neuromorphic?

| | GPU (A100) | Intel Loihi 2 | Human Brain |
|--|--|--|--|
| Power | 300W | 30mW | 20W |
| Energy/inference | ~1000 nJ | ~0.1 nJ | ~0.001 nJ |
| Efficiency | baseline | **10,000× better** | **1,000,000× better** |

The brain uses 20 watts. GPT-4 uses 50 megawatts. That's a 2.5 million× gap.  
Neuromorphic chips close that gap — and `synaptic_ml` makes them programmable.

---

## Installation

```bash
pip install synaptic-ml
# or from source:
git clone https://github.com/yourusername/synaptic-ml
pip install -e .
```

**Optional extras:**
```bash
pip install synaptic-ml[torch]    # PyTorch ANN→SNN conversion
pip install synaptic-ml[sklearn]  # MNIST examples
pip install synaptic-ml[all]      # Everything
```

---

## Core Concepts

### Neuron Models

```python
# Leaky Integrate-and-Fire (default, fast, efficient)
model = sml.SpikingNet([784, 256, 10], neuron='lif')

# Adaptive LIF (spike-frequency adaptation)
model = sml.SpikingNet([784, 256, 10], neuron='adaptive_lif')

# Izhikevich (biologically rich: bursting, chattering, fast-spiking)
model = sml.SpikingNet([784, 256, 10], neuron='izhikevich')
```

### Spike Encoders

```python
import synaptic_ml as sml
import numpy as np

x = np.random.rand(784)  # your data, values in [0, 1]

# Rate coding: spike probability ∝ value (most common)
enc = sml.RateEncoder(time_steps=100, max_rate=100)
spikes = enc.encode(x)  # shape: (100, 784)

# Temporal: earlier spike = stronger signal (most energy efficient)
enc = sml.TemporalEncoder(time_steps=100)
spikes = enc.encode(x)  # at most 1 spike per neuron

# Population: Gaussian tuning curves (most biologically realistic)
enc = sml.PopulationEncoder(n_neurons=10, sigma=0.5)
spikes = enc.encode(x)  # shape: (100, 7840)

# Delta: spikes only on change — perfect for IoT/sensors
enc = sml.DeltaEncoder(threshold=0.05)
spikes = enc.encode_series(time_series)  # event-driven
```

### Learning Rules

```python
# Surrogate gradients (recommended) — backprop through spikes
model.train(X, y, learning_rule='surrogate', learning_rate=0.001)

# STDP — unsupervised, no labels needed, biologically inspired
model.train(X, y, learning_rule='stdp')

# Custom trainer
trainer = sml.Trainer(model, learning_rule='surrogate', learning_rate=0.001)
trainer.fit(X_train, y_train, epochs=10, validation_split=0.1)
trainer.evaluate(X_test, y_test)
```

### ANN → SNN Conversion

Convert your existing PyTorch/Keras models to SNNs in one line:

```python
import torch.nn as nn
import synaptic_ml as sml

# Your existing trained ANN
ann = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

# Convert to SNN (threshold balancing method)
snn = sml.convert_from_pytorch(ann, X_calibration, time_steps=100)

# Deploy to neuromorphic hardware
snn.deploy(target='loihi2')
```

### Hardware Backends

```python
# CPU simulation — always works, no hardware needed
backend = model.deploy(target='cpu')

# Intel Loihi 2 — requires NxSDK (Intel Research Program)
# Apply at: intel.com/loihi
backend = model.deploy(target='loihi2')

# BrainScaleS-2 — requires EBRAINS access (Human Brain Project)  
# Apply at: ebrains.eu
backend = model.deploy(target='brainscales')
```

### Energy Analysis

```python
# After running predictions:
model.predict(X_test)

# Detailed energy breakdown
energy = model.estimate_energy()
print(f"Ops/inference:    {energy['synaptic_ops_per_inference']:,.0f}")
print(f"Energy/inference: {energy['energy_per_inference_nJ']:.4f} nJ")
print(f"vs GPU:           {energy['efficiency_gain']:.0f}× more efficient")

# Full comparison table
print(sml.energy_comparison_table(model))
```

### Save / Load

```python
model.save('my_snn.snm')
model = sml.SpikingNet.load('my_snn.snm')
```

---

## Examples

```bash
# MNIST classification with LIF neurons
python examples/mnist_lif.py

# Unsupervised pattern learning with STDP
python examples/pattern_recognition.py

# IoT anomaly detection with delta encoding
python examples/edge_sensor.py

# Quick self-test
python -m synaptic_ml
```

---

## Architecture

```
synaptic_ml/
├── core/
│   ├── neurons.py      # LIF, Adaptive LIF, Izhikevich
│   ├── synapses.py     # Dense, Sparse, Delayed
│   ├── layers.py       # LIFLayer, OutputLayer, etc.
│   └── network.py      # SpikingNet (main model class)
├── encoding/
│   ├── rate.py         # Poisson rate coding
│   ├── temporal.py     # Time-to-first-spike, phase coding
│   └── population.py   # Gaussian tuning, delta/event coding
├── learning/
│   ├── stdp.py         # STDP, reward-modulated STDP
│   ├── surrogate.py    # Surrogate gradient functions
│   └── conversion.py   # ANN→SNN threshold balancing
├── backends/
│   ├── cpu.py          # Pure numpy simulation
│   ├── loihi2.py       # Intel Loihi 2 (NxSDK)
│   └── brainscales.py  # BrainScaleS-2 (PyNN)
├── training/
│   └── trainer.py      # Training loop, loss functions
└── utils/
    ├── metrics.py      # Energy, spike metrics, Van Rossum distance
    └── visualization.py # Raster plots, membrane traces, energy bars
```

---

## Hardware Access

Neither Loihi 2 nor BrainScaleS requires purchase — they're available through research programs:

- **Intel Loihi 2**: [Intel Neuromorphic Research Community](https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html)
- **BrainScaleS-2**: [EBRAINS / Human Brain Project](https://www.ebrains.eu/)

Until you get hardware access, the CPU backend gives exact simulation results.

---

## Roadmap

- [ ] GPU-accelerated simulation (CuPy backend)
- [ ] Innatera T1 chip support
- [ ] Online/streaming inference API
- [ ] Quantization-aware training
- [ ] Multi-chip deployment
- [ ] Web dashboard for spike visualization

---

## License

MIT License. See LICENSE.

---

*The brain uses 20 watts. GPT-4 uses 50 megawatts. synaptic_ml bridges that gap.*
