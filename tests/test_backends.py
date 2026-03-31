"""Tests for hardware backends."""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synaptic_ml.core.network import SpikingNet
from synaptic_ml.backends import get_backend, BackendNotAvailableError
from synaptic_ml.backends.cpu import CPUBackend
from synaptic_ml.encoding.rate import RateEncoder


class TestGetBackend:

    def test_cpu_backend(self):
        backend = get_backend("cpu")
        assert isinstance(backend, CPUBackend)

    def test_invalid_backend(self):
        with pytest.raises(ValueError):
            get_backend("invalid")

    def test_all_backends_instantiate(self):
        for name in ["cpu", "loihi2", "brainscales", "akida"]:
            backend = get_backend(name)
            assert backend is not None


class TestCPUBackend:

    def setup_method(self):
        self.model = SpikingNet([8, 4, 2], time_steps=10)
        self.backend = CPUBackend()
        self.backend.load_network(self.model)
        self.encoder = RateEncoder(time_steps=10)

    def test_load_network(self):
        assert self.backend.network is not None

    def test_run_output_shape(self):
        encoded = self.encoder.encode(np.random.rand(8))
        output = self.backend.run(encoded, time_steps=10)
        assert output.shape == (2,)

    def test_run_without_load_raises(self):
        backend = CPUBackend()
        with pytest.raises(RuntimeError):
            backend.run(np.zeros((10, 8)), time_steps=10)

    def test_energy_estimate_empty_before_run(self):
        backend = CPUBackend()
        backend.load_network(self.model)
        energy = backend.get_energy_estimate()
        assert energy == {}

    def test_energy_estimate_after_run(self):
        encoded = self.encoder.encode(np.random.rand(8))
        self.backend.run(encoded, time_steps=10)
        energy = self.backend.get_energy_estimate()
        assert "neuromorphic_energy_nJ" in energy
        assert "gpu_equivalent_energy_nJ" in energy
        assert energy["efficiency_gain"] > 0

    def test_hardware_info(self):
        info = self.backend.get_hardware_info()
        assert "CPU" in info or "cpu" in info.lower()

    def test_repr(self):
        assert "CPUBackend" in repr(self.backend)


class TestLoihi2Backend:

    def test_instantiates(self):
        from synaptic_ml.backends.loihi2 import Loihi2Backend
        backend = Loihi2Backend()
        assert backend is not None

    def test_hardware_info_contains_loihi(self):
        from synaptic_ml.backends.loihi2 import Loihi2Backend
        backend = Loihi2Backend()
        info = backend.get_hardware_info()
        assert "Loihi" in info

    def test_load_without_lava_raises(self):
        from synaptic_ml.backends.loihi2 import Loihi2Backend
        backend = Loihi2Backend()
        if not backend._lava_available:
            model = SpikingNet([8, 4, 2])
            with pytest.raises(BackendNotAvailableError):
                backend.load_network(model)


class TestAkidaBackend:

    def test_instantiates(self):
        from synaptic_ml.backends.akida import AkidaBackend
        backend = AkidaBackend()
        assert backend is not None

    def test_hardware_info(self):
        from synaptic_ml.backends.akida import AkidaBackend
        backend = AkidaBackend()
        info = backend.get_hardware_info()
        assert "Akida" in info

    def test_load_and_run_if_sdk_available(self):
        from synaptic_ml.backends.akida import AkidaBackend
        backend = AkidaBackend()
        if backend._sdk_available:
            model = SpikingNet([16, 8, 4], time_steps=20)
            backend.load_network(model)
            encoder = RateEncoder(time_steps=20)
            encoded = encoder.encode(np.random.rand(16))
            output = backend.run(encoded)
            assert output.shape == (4,)
        else:
            pytest.skip("Akida SDK not installed")
