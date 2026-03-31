"""Tests for SpikingNet model."""

import numpy as np
import pytest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synaptic_ml.core.network import SpikingNet
from synaptic_ml.encoding.rate import RateEncoder


class TestSpikingNet:

    def test_build_lif(self):
        model = SpikingNet([16, 8, 4], neuron="lif")
        assert model.layer_sizes == [16, 8, 4]
        assert model.neuron_type == "lif"

    def test_build_adaptive_lif(self):
        model = SpikingNet([16, 8, 4], neuron="adaptive_lif")
        assert model.neuron_type == "adaptive_lif"

    def test_build_izhikevich(self):
        model = SpikingNet([16, 8, 4], neuron="izhikevich")
        assert model.neuron_type == "izhikevich"

    def test_invalid_neuron_type(self):
        with pytest.raises(ValueError):
            SpikingNet([16, 8, 4], neuron="invalid")

    def test_minimum_layers(self):
        with pytest.raises(ValueError):
            SpikingNet([16])

    def test_run_output_shape(self):
        model = SpikingNet([16, 8, 4], time_steps=20)
        encoder = RateEncoder(time_steps=20)
        encoded = encoder.encode(np.random.rand(16))
        scores = model.run(encoded)
        assert scores.shape == (4,)

    def test_run_increments_inference_count(self):
        model = SpikingNet([8, 4, 2], time_steps=10)
        encoder = RateEncoder(time_steps=10)
        encoded = encoder.encode(np.random.rand(8))
        model.run(encoded)
        assert model._inference_count == 1

    def test_predict_output_shape(self):
        model = SpikingNet([8, 4, 3], time_steps=10)
        X = np.random.rand(5, 8).astype(np.float32)
        preds = model.predict(X, verbose=False)
        assert preds.shape == (5,)

    def test_predict_valid_classes(self):
        model = SpikingNet([8, 4, 3], time_steps=10)
        X = np.random.rand(10, 8).astype(np.float32)
        preds = model.predict(X, verbose=False)
        assert np.all(preds >= 0) and np.all(preds < 3)

    def test_summary_runs(self, capsys):
        model = SpikingNet([16, 8, 4])
        model.summary()
        captured = capsys.readouterr()
        assert "SpikingNet" in captured.out

    def test_save_load(self, tmp_path):
        model = SpikingNet([8, 4, 2], time_steps=10)
        path = str(tmp_path / "test_model.snm")
        model.save(path)
        loaded = SpikingNet.load(path)
        assert loaded.layer_sizes == model.layer_sizes
        assert loaded.neuron_type == model.neuron_type

    def test_save_load_weights_preserved(self, tmp_path):
        model = SpikingNet([8, 4, 2], time_steps=10)
        path = str(tmp_path / "test_model.snm")
        model.save(path)
        loaded = SpikingNet.load(path)
        # Check weights match
        for orig, load in zip(
            [l for l in model.layers if hasattr(l, "weights")],
            [l for l in loaded.layers if hasattr(l, "weights")]
        ):
            np.testing.assert_array_almost_equal(orig.weights, load.weights)

    def test_estimate_energy_empty(self):
        model = SpikingNet([8, 4, 2], time_steps=10)
        energy = model.estimate_energy()
        assert energy == {}

    def test_estimate_energy_after_inference(self):
        model = SpikingNet([8, 4, 2], time_steps=10)
        encoder = RateEncoder(time_steps=10)
        for _ in range(3):
            model.run(encoder.encode(np.random.rand(8)))
        energy = model.estimate_energy()
        assert "synaptic_ops_per_inference" in energy
        assert "efficiency_gain" in energy

    def test_repr(self):
        model = SpikingNet([16, 8, 4])
        r = repr(model)
        assert "SpikingNet" in r
        assert "16" in r

    def test_deploy_cpu(self):
        model = SpikingNet([8, 4, 2], time_steps=10)
        backend = model.deploy(target="cpu")
        assert backend is not None

    def test_deploy_invalid(self):
        model = SpikingNet([8, 4, 2])
        with pytest.raises(ValueError):
            model.deploy(target="invalid_chip")
