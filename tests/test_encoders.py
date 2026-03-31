"""Tests for spike encoders."""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synaptic_ml.encoding.rate import RateEncoder
from synaptic_ml.encoding.temporal import TemporalEncoder, PhaseEncoder
from synaptic_ml.encoding.population import PopulationEncoder, DeltaEncoder


class TestRateEncoder:

    def test_output_shape(self):
        enc = RateEncoder(time_steps=50)
        x = np.random.rand(10)
        spikes = enc.encode(x)
        assert spikes.shape == (50, 10)

    def test_output_dtype(self):
        enc = RateEncoder(time_steps=20)
        spikes = enc.encode(np.random.rand(5))
        assert spikes.dtype == np.float32

    def test_output_binary(self):
        enc = RateEncoder(time_steps=100)
        spikes = enc.encode(np.random.rand(20))
        assert np.all((spikes == 0) | (spikes == 1))

    def test_zero_input_no_spikes(self):
        enc = RateEncoder(time_steps=100, max_rate=100)
        spikes = enc.encode(np.zeros(10))
        assert np.all(spikes == 0)

    def test_higher_input_more_spikes(self):
        enc = RateEncoder(time_steps=1000)
        low_spikes = enc.encode(np.full(1, 0.1)).sum()
        high_spikes = enc.encode(np.full(1, 0.9)).sum()
        assert high_spikes > low_spikes

    def test_clips_input(self):
        enc = RateEncoder(time_steps=50)
        # Should not crash with out-of-range input
        spikes = enc.encode(np.array([2.0, -1.0, 0.5]))
        assert spikes.shape == (50, 3)

    def test_decode_roundtrip(self):
        enc = RateEncoder(time_steps=1000, max_rate=100)
        x = np.array([0.8])
        spikes = enc.encode(x)
        decoded = enc.decode(spikes)
        # Should be roughly close (stochastic so allow tolerance)
        assert abs(decoded[0] - x[0]) < 0.3

    def test_repr(self):
        enc = RateEncoder(time_steps=100)
        assert "RateEncoder" in repr(enc)


class TestTemporalEncoder:

    def test_output_shape(self):
        enc = TemporalEncoder(time_steps=50)
        spikes = enc.encode(np.random.rand(8))
        assert spikes.shape == (50, 8)

    def test_at_most_one_spike_per_neuron(self):
        enc = TemporalEncoder(time_steps=100)
        spikes = enc.encode(np.random.rand(20))
        assert np.all(spikes.sum(axis=0) <= 1)

    def test_higher_value_earlier_spike(self):
        enc = TemporalEncoder(time_steps=100)
        x = np.array([0.9, 0.1])
        spikes = enc.encode(x)
        times = [np.where(spikes[:, i] > 0)[0] for i in range(2)]
        if len(times[0]) > 0 and len(times[1]) > 0:
            assert times[0][0] < times[1][0]

    def test_zero_below_threshold_no_spike(self):
        enc = TemporalEncoder(time_steps=50, threshold=0.5)
        spikes = enc.encode(np.array([0.1, 0.2]))
        assert np.all(spikes == 0)

    def test_decode(self):
        enc = TemporalEncoder(time_steps=100)
        x = np.array([0.8, 0.3])
        spikes = enc.encode(x)
        decoded = enc.decode(spikes)
        assert decoded.shape == (2,)


class TestPopulationEncoder:

    def test_output_shape(self):
        enc = PopulationEncoder(n_neurons=5, time_steps=50)
        x = np.random.rand(4)
        spikes = enc.encode(x)
        assert spikes.shape == (50, 20)  # 4 features * 5 neurons

    def test_encode_scalar(self):
        enc = PopulationEncoder(n_neurons=10)
        act = enc.encode_scalar(0.5)
        assert act.shape == (10,)
        assert np.all(act >= 0) and np.all(act <= 1)

    def test_peak_at_preferred_value(self):
        enc = PopulationEncoder(n_neurons=10, sigma=0.1)
        # Neuron at index 5 prefers value ~0.55
        act = enc.encode_scalar(enc.preferred[5])
        assert act[5] == pytest.approx(1.0, abs=0.01)


class TestDeltaEncoder:

    def test_encode_series_shape(self):
        enc = DeltaEncoder(threshold=0.1)
        series = np.random.rand(50, 4)
        spikes = enc.encode_series(series)
        assert spikes.shape == (50, 8)  # ON + OFF per channel

    def test_no_spikes_constant_signal(self):
        enc = DeltaEncoder(threshold=0.05)
        # Constant signal — no changes, no spikes (after first step)
        series = np.full((20, 3), 0.5)
        spikes = enc.encode_series(series)
        assert spikes[1:].sum() == 0

    def test_spikes_on_large_change(self):
        enc = DeltaEncoder(threshold=0.1)
        series = np.zeros((10, 1))
        series[5, 0] = 0.8  # big jump
        spikes = enc.encode_series(series)
        assert spikes[5, 0] == 1.0  # ON event

    def test_fewer_spikes_than_rate(self):
        enc_delta = DeltaEncoder(threshold=0.05, time_steps=50)
        enc_rate = RateEncoder(time_steps=50)

        # Slowly changing signal
        x = np.sin(np.linspace(0, 0.5, 50)).reshape(-1, 1) * 0.5 + 0.5
        delta_spikes = enc_delta.encode_series(x).sum()

        rate_spikes = 0
        for i in range(50):
            rate_spikes += enc_rate.encode(x[i]).sum()

        assert delta_spikes < rate_spikes
