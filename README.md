# synaptic_ml

[![PyPI version](https://img.shields.io/pypi/v/synaptic-ml)](https://pypi.org/project/synaptic-ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-111%20passed-brightgreen)](https://github.com/HrishikeshRajulu/synaptic-ml)

**The TensorFlow for Neuromorphic Computing.**

Train spiking neural networks. Deploy to neuromorphic chips. Use 1000x less energy than GPU.

```python
import synaptic_ml as sml

# Build a spiking network
model = sml.SpikingNet([784, 256, 10])
model.summary()

# Train with surrogate gradients
model.train(X_train, y_train, epochs=10)

# See the energy advantage
print(sml.energy_comparison_table(model))

# Deploy to neuromorphic hardware
model.deploy(target='akida')       # BrainChip Akida (works today, pip install akida)
model.deploy(target='loihi2')      # Intel Loihi 2 (via Lava framework)
model.deploy(target='brainscales') # BrainScaleS-2 (via PyNN)
model.deploy(target='cpu')         # CPU simulation (default, always works)
```

---

## Why neuromorphic?

| | GPU (A100) | Intel Loihi 2 | BrainChip Akida | Human Brain |
|--|--|--|--|--|
| Power | 300W | 30mW | 30mW | 20W |
| Energy/inference | ~1000 nJ | ~0.1 nJ | ~1.4 nJ | ~0.001 nJ |
| Available | Yes | Research only | **Yes — buy now** | No |

The brain uses 20 watts. GPT-4 uses 50 megawatts. That is a 2.5 million x gap.
Neuromorphic chips close that gap. `synaptic_ml` makes them programmable.

---

## Installation

```bash
pip install synaptic-ml
```

**Optional hardware backends:**
```bash
pip install akida        # BrainChip Akida (recommended — works without hardware)
pip install lava-nc      # Intel Loihi 2 via Lava (requires Python 3.10)
pip install pyNN         # BrainScaleS-2 / SpiNNaker 2
```

---

## Hardware Backends

| Backend | Chip | Install | Hardware needed? |
|---------|------|---------|-----------------|
| `cpu` | CPU simulation | built-in | No |
| `akida` | BrainChip AKD1000/AKD1500 | `pip install akida` | No (virtual mode) |
| `loihi2` | Intel Loihi 2 | `pip install lava-nc` | No (CPU sim via Lava) |
| `brainscales` | BrainScaleS-2 | `pip install pyNN` | Apply at ebrains.eu |

### BrainChip Akida — works today

```python
# No hardware needed — runs in virtual simulation
backend = model.deploy(target='akida')

# With real AKD1000/AKD1500 hardware connected
backend = model.deploy(target='akida', use_hardware=True)
```

Buy Akida hardware: [brainchipinc.com/products](https://brainchipinc.com/products)

### Intel Loihi 2 — via Lava

```python
# Lava is Intel's official open-source neuromorphic framework
# synaptic_ml sits on top of it as a friendly API layer
backend = model.deploy(target='loihi2')
```

For real Loihi 2 hardware, join the INRC: [neuromorphic.intel.com](http://neuromorphic.intel.com)

---

## Neuron Models

```python
# Leaky Integrate-and-Fire (default, fast)
model = sml.SpikingNet([784, 256, 10], neuron='lif')

# Adaptive LIF (spike-frequency adaptation)
model = sml.SpikingNet([784, 256, 10], neuron='adaptive_lif')

# Izhikevich (biologically rich: bursting, chattering, fast-spiking)
model = sml.SpikingNet([784, 256, 10], neuron='izhikevich')
```

---

## Spike Encoders

```python
import numpy as np
x = np.random.rand(784)  # values in [0, 1]

# Rate coding — spike probability proportional to value (most common)
enc = sml.RateEncoder(time_steps=100, max_rate=100)
spikes = enc.encode(x)  # shape: (100, 784)

# Temporal — earlier spike = stronger signal (most energy efficient)
enc = sml.TemporalEncoder(time_steps=100)
spikes = enc.encode(x)  # at most 1 spike per neuron

# Population — Gaussian tuning curves (most biologically realistic)
enc = sml.PopulationEncoder(n_neurons=10, sigma=0.5)
spikes = enc.encode(x)

# Delta — spikes only on change (perfect for IoT sensors)
enc = sml.DeltaEncoder(threshold=0.05)
spikes = enc.encode_series(time_series)
```

---

## Learning Rules

```python
# Surrogate gradients (recommended) — backprop through spike nonlinearity
# Uses voltage-based soft activations + Adam optimizer for stable training.
model.train(X, y, learning_rule='surrogate', learning_rate=0.001, epochs=30)

# STDP — unsupervised, no labels needed, biologically inspired
model.train(X, y, learning_rule='stdp')

# Custom trainer with validation split
trainer = sml.Trainer(model, learning_rule='surrogate', learning_rate=0.001)
trainer.fit(X_train, y_train, epochs=50, validation_split=0.1)
trainer.evaluate(X_test, y_test)
```

---

## ANN to SNN Conversion

Convert your existing PyTorch or Keras models to SNNs in one line:

```python
import torch.nn as nn

# Your trained ANN
ann = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

# Convert to SNN (threshold balancing)
snn = sml.convert_from_pytorch(ann, X_calibration, time_steps=100)

# Deploy to neuromorphic hardware
snn.deploy(target='akida')
```

---

## Energy Analysis

```python
# After running predictions
model.predict(X_test)

# Detailed energy breakdown
energy = model.estimate_energy()
print(f"Ops/inference:    {energy['synaptic_ops_per_inference']:,.0f}")
print(f"Energy/inference: {energy['energy_per_inference_nJ']:.4f} nJ")
print(f"vs GPU:           {energy['efficiency_gain']:.0f}x more efficient")

# Full comparison table
print(sml.energy_comparison_table(model))
```

---

## Save / Load

```python
model.save('my_snn.snm')
model = sml.SpikingNet.load('my_snn.snm')
```

---

## Examples

```bash
# MNIST classification with LIF neurons + surrogate gradients
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
│   └── conversion.py   # ANN->SNN threshold balancing
├── backends/
│   ├── cpu.py          # Pure numpy simulation
│   ├── akida.py        # BrainChip Akida (pip install akida)
│   ├── loihi2.py       # Intel Loihi 2 (via Lava)
│   └── brainscales.py  # BrainScaleS-2 (via PyNN)
├── training/
│   └── trainer.py      # Training loop, surrogate + STDP
└── utils/
    ├── metrics.py      # Energy, spike metrics, Van Rossum distance
    └── visualization.py # Raster plots, membrane traces, energy bars
```

---

## Roadmap

- [x] LIF, Adaptive LIF, Izhikevich neuron models
- [x] Rate, Temporal, Population, Delta encoders
- [x] LIF, Adaptive LIF, Izhikevich neuron models
- [x] Rate, Temporal, Population, Delta encoders
- [x] Surrogate gradient training (BPTT with momentum, voltage-based soft activations)
- [x] STDP + Reward-modulated STDP
- [x] ANN->SNN conversion (threshold balancing)
- [x] BrainChip Akida backend (CPU virtual mode + hardware mode)
- [x] Intel Loihi 2 backend (via Lava framework)
- [x] BrainScaleS-2 backend (via PyNN)
- [x] 111-test suite (neurons, encoders, layers, network, backends, learning)
- [ ] GPU simulation (CuPy backend)
- [ ] SpiNNaker 2 backend
- [ ] Innatera T1 backend
- [ ] IBM NorthPole backend (when SDK releases)
- [ ] Quantization-aware training
- [ ] Online/streaming inference API
- [ ] Pretrained model hub

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

Most needed right now:
- Someone with **Loihi 2 hardware** to test and fix the Loihi 2 backend
- Someone with **Akida AKD1000/AKD1500** hardware to test hardware mode
- **GPU simulation** via CuPy
- **MNIST/CIFAR benchmarks** — real-world accuracy numbers

---

## License

MIT License — see [LICENSE](LICENSE).

Copyright (c) 2026 Hrishikesh Rajulu

---

*The brain uses 20 watts. GPT-4 uses 50 megawatts. synaptic_ml bridges that gap.*
