"""Tests for spiking neural network layers."""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synaptic_ml.core.layers import (
    LIFLayer, AdaptiveLIFLayer, IzhikevichLayer,
    InputLayer, OutputLayer
)


class TestInputLayer:

    def test_passthrough(self):
        layer = InputLayer(10)
        x = np.random.rand(10).astype(np.float32)
        out = layer.forward(x)
        np.testing.assert_array_equal(out, x.astype(np.float32))

    def test_n_neurons(self):
        layer = InputLayer(16)
        assert layer.n_neurons == 16

    def test_reset_state(self):
        layer = InputLayer(5)
        layer.last_spikes = np.ones(5)
        layer.reset_state()
        assert np.all(layer.last_spikes == 0)


class TestLIFLayer:

    def test_output_shape(self):
        layer = LIFLayer(8, 16)
        spikes = layer.forward(np.random.rand(8).astype(np.float32))
        assert spikes.shape == (16,)

    def test_n_neurons(self):
        layer = LIFLayer(4, 12)
        assert layer.n_neurons == 12

    def test_weights_shape(self):
        layer = LIFLayer(10, 20)
        assert layer.weights.shape == (10, 20)

    def test_records_history(self):
        layer = LIFLayer(4, 8, record=True)
        for _ in range(5):
            layer.forward(np.random.rand(4).astype(np.float32))
        assert len(layer.spike_history) == 5
        assert len(layer.voltage_history) == 5

    def test_spike_trains_property(self):
        layer = LIFLayer(4, 8)
        for _ in range(10):
            layer.forward(np.random.rand(4).astype(np.float32))
        trains = layer.spike_trains
        assert trains.shape == (10, 8)

    def test_reset_clears_history(self):
        layer = LIFLayer(4, 8)
        for _ in range(5):
            layer.forward(np.random.rand(4).astype(np.float32))
        layer.reset_state()
        assert len(layer.spike_history) == 0

    def test_output_binary(self):
        layer = LIFLayer(4, 8)
        spikes = layer.forward(np.random.rand(4).astype(np.float32))
        assert np.all((spikes == 0) | (spikes == 1))

    def test_repr(self):
        layer = LIFLayer(4, 8)
        assert "LIFLayer" in repr(layer)
        assert "4" in repr(layer)
        assert "8" in repr(layer)


class TestAdaptiveLIFLayer:

    def test_output_shape(self):
        layer = AdaptiveLIFLayer(6, 10)
        spikes = layer.forward(np.random.rand(6).astype(np.float32))
        assert spikes.shape == (10,)

    def test_n_neurons(self):
        layer = AdaptiveLIFLayer(4, 9)
        assert layer.n_neurons == 9


class TestIzhikevichLayer:

    def test_output_shape(self):
        layer = IzhikevichLayer(5, 8)
        spikes = layer.forward(np.random.rand(5).astype(np.float32))
        assert spikes.shape == (8,)

    def test_preset(self):
        layer = IzhikevichLayer(4, 6, preset="fast_spiking")
        assert layer.neurons.preset == "fast_spiking"


class TestOutputLayer:

    def test_output_shape(self):
        layer = OutputLayer(8, 4)
        spikes = layer.forward(np.random.rand(8).astype(np.float32))
        assert spikes.shape == (4,)

    def test_decode_rate(self):
        layer = OutputLayer(8, 4, decoder="rate")
        for _ in range(10):
            layer.forward(np.random.rand(8).astype(np.float32))
        scores = layer.decode()
        assert scores.shape == (4,)

    def test_predict_class(self):
        layer = OutputLayer(8, 4)
        for _ in range(10):
            layer.forward(np.random.rand(8).astype(np.float32))
        pred = layer.predict_class()
        assert 0 <= pred < 4

    def test_reset_clears_accumulator(self):
        layer = OutputLayer(8, 4)
        for _ in range(5):
            layer.forward(np.random.rand(8).astype(np.float32))
        layer.reset_state()
        assert np.all(layer.spike_accumulator == 0)
        assert layer.spike_count == 0
