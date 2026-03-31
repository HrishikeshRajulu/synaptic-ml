# Contributing to synaptic_ml

First off — thank you. This project needs people who care about making
neuromorphic hardware accessible to normal developers.

## Where help is most needed

| Area | What's needed | Skill required |
|------|--------------|----------------|
| **Hardware backends** | Someone with Loihi 2 / BrainScaleS access to test and fix backends | NxSDK or PyNN experience |
| **Training** | Proper BPTT, better surrogate gradient implementation | Deep learning internals |
| **GPU simulation** | CuPy or PyTorch backend for fast simulation | CUDA / CuPy |
| **Tests** | pytest coverage for all modules | Basic Python testing |
| **Examples** | More real-world demos (audio, vision, robotics) | Domain knowledge |
| **Docs** | Better docstrings, tutorials | Technical writing |

---

## How to add a new neuron model

1. Open `synaptic_ml/core/neurons.py`
2. Create a class with:
   - `reset_state()` method
   - `step(input_current, dt)` method returning `(voltage, spikes)`
   - Vectorized over N neurons using numpy
3. Register it in `NEURON_REGISTRY` at the bottom of the file
4. Add a corresponding `Layer` class in `layers.py`

```python
class MyNeuron:
    def __init__(self, n_neurons: int, **kwargs):
        self.n_neurons = n_neurons
        self.reset_state()

    def reset_state(self):
        self.v = np.zeros(self.n_neurons, dtype=np.float32)

    def step(self, input_current: np.ndarray, dt: float = 1.0):
        # your dynamics here
        spikes = (self.v >= threshold).astype(np.float32)
        return self.v.copy(), spikes
```

---

## How to add a new hardware backend

1. Create `synaptic_ml/backends/yourchip.py`
2. Subclass `Backend` from `backends/base.py`
3. Implement `load_network()`, `run()`, `get_energy_estimate()`, `get_hardware_info()`
4. Register in `backends/__init__.py` and `get_backend()`

If the SDK isn't publicly available, raise `BackendNotAvailableError` with helpful installation instructions — see `loihi2.py` for the pattern.

---

## How to add a new encoder

1. Create or add to `synaptic_ml/encoding/`
2. Implement `encode(x: np.ndarray) -> np.ndarray` returning shape `(time_steps, n_neurons)`
3. Export from `encoding/__init__.py`

---

## Running tests

```bash
pip install pytest
pytest tests/
```

---

## Code style

- Python 3.8+
- Type hints on all public functions
- Docstrings in NumPy style
- No dependencies beyond `numpy`, `scipy`, `matplotlib` for the core
- Hardware SDKs go in `extras_require` in `setup.py`

---

## Submitting a PR

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request with a clear description of what and why

---

## Questions?

Open a GitHub Issue. Label it `question` — no question is too basic.

The goal is a framework that a developer with zero neuroscience background
can use in an afternoon. If something is confusing, that's a bug.
