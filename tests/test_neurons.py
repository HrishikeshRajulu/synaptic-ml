"""Tests for neuron models."""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synaptic_ml.core.neurons import LIFNeuron, AdaptiveLIFNeuron, IzhikevichNeuron


class TestLIFNeuron:

    def test_init(self):
        n = LIFNeuron(10)
        assert n.n_neurons == 10
        assert n.v.shape == (10,)
        assert np.all(n.v == n.v_rest)

    def test_step_returns_correct_shapes(self):
        n = LIFNeuron(8)
        current = np.zeros(8)
        v, spikes = n.step(current, dt=1.0)
        assert v.shape == (8,)
        assert spikes.shape == (8,)

    def test_no_spike_below_threshold(self):
        n = LIFNeuron(5)
        current = np.zeros(5)
        v, spikes = n.step(current, dt=1.0)
        assert np.all(spikes == 0)

    def test_spike_above_threshold(self):
        n = LIFNeuron(1)
        # Force voltage above threshold
        n.v = np.array([n.v_thresh + 1.0])
        current = np.zeros(1)
        v, spikes = n.step(current, dt=1.0)
        assert spikes[0] == 1.0

    def test_reset_after_spike(self):
        n = LIFNeuron(1)
        n.v = np.array([n.v_thresh + 1.0])
        v, spikes = n.step(np.zeros(1), dt=1.0)
        assert v[0] == n.v_reset

    def test_refractory_period(self):
        n = LIFNeuron(1, refractory_period=5.0)
        n.v = np.array([n.v_thresh + 1.0])
        _, spikes1 = n.step(np.zeros(1), dt=1.0)
        assert spikes1[0] == 1.0
        # Should not spike again immediately
        n.v = np.array([n.v_thresh + 1.0])
        _, spikes2 = n.step(np.zeros(1), dt=1.0)
        assert spikes2[0] == 0.0

    def test_reset_state(self):
        n = LIFNeuron(5)
        n.v = np.ones(5) * 100.0
        n.reset_state()
        assert np.all(n.v == n.v_rest)

    def test_vectorized_multiple_neurons(self):
        n = LIFNeuron(100)
        current = np.random.rand(100) * 5.0
        v, spikes = n.step(current, dt=1.0)
        assert v.shape == (100,)
        assert spikes.shape == (100,)
        assert spikes.dtype == np.float32

    def test_repr(self):
        n = LIFNeuron(10)
        assert "LIFNeuron" in repr(n)
        assert "10" in repr(n)


class TestAdaptiveLIFNeuron:

    def test_init(self):
        n = AdaptiveLIFNeuron(5)
        assert n.n_neurons == 5
        assert n.w.shape == (5,)
        assert np.all(n.w == 0)

    def test_step_shape(self):
        n = AdaptiveLIFNeuron(4)
        v, spikes = n.step(np.zeros(4), dt=1.0)
        assert v.shape == (4,)
        assert spikes.shape == (4,)

    def test_adaptation_increases_on_spike(self):
        n = AdaptiveLIFNeuron(1, b=0.5)
        n.v = np.array([n.v_thresh + 1.0])
        w_before = n.w[0]
        n.step(np.zeros(1), dt=1.0)
        assert n.w[0] > w_before

    def test_reset_state(self):
        n = AdaptiveLIFNeuron(3)
        n.w = np.ones(3) * 10.0
        n.reset_state()
        assert np.all(n.w == 0)


class TestIzhikevichNeuron:

    def test_init_presets(self):
        for preset in ["regular_spiking", "fast_spiking", "bursting"]:
            n = IzhikevichNeuron(5, preset=preset)
            assert n.n_neurons == 5
            assert n.preset == preset

    def test_step_shape(self):
        n = IzhikevichNeuron(6)
        v, spikes = n.step(np.zeros(6), dt=1.0)
        assert v.shape == (6,)
        assert spikes.shape == (6,)

    def test_fires_with_strong_input(self):
        n = IzhikevichNeuron(1)
        # Strong input should cause firing
        fired = False
        for _ in range(100):
            _, spikes = n.step(np.array([15.0]), dt=1.0)
            if spikes[0] > 0:
                fired = True
                break
        assert fired

    def test_reset_state(self):
        n = IzhikevichNeuron(3)
        n.v = np.ones(3) * 100.0
        n.reset_state()
        assert np.all(n.v == -65.0)

    def test_repr(self):
        n = IzhikevichNeuron(5, preset="fast_spiking")
        assert "fast_spiking" in repr(n)
