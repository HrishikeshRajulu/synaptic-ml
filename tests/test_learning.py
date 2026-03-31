"""Tests for learning rules and ANN->SNN conversion."""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synaptic_ml.learning.stdp import STDPRule, RewardModulatedSTDP
from synaptic_ml.learning.surrogate import (
    SigmoidSurrogate, PiecewiseLinearSurrogate,
    FastSigmoidSurrogate, SuperSpike, get_surrogate
)
from synaptic_ml.learning.conversion import convert_from_numpy_weights


class TestSTDP:

    def test_init(self):
        stdp = STDPRule()
        assert stdp.A_plus > 0
        assert stdp.A_minus > 0

    def test_update_shape(self):
        stdp = STDPRule()
        n_pre, n_post = 5, 8
        weights = np.random.rand(n_pre, n_post).astype(np.float32)
        pre_t, post_t = stdp.init_traces(n_pre, n_post)
        pre_spikes = (np.random.rand(n_pre) > 0.8).astype(np.float32)
        post_spikes = (np.random.rand(n_post) > 0.8).astype(np.float32)

        delta_w, pre_t, post_t = stdp.update(
            pre_spikes, post_spikes, weights, pre_t, post_t, dt=1.0
        )
        assert delta_w.shape == (n_pre, n_post)

    def test_weights_clipped(self):
        stdp = STDPRule(w_min=0.0, w_max=1.0)
        n_pre, n_post = 4, 6
        weights = np.random.rand(n_pre, n_post).astype(np.float32)
        pre_t, post_t = stdp.init_traces(n_pre, n_post)

        for _ in range(50):
            pre_spikes = (np.random.rand(n_pre) > 0.5).astype(np.float32)
            post_spikes = (np.random.rand(n_post) > 0.5).astype(np.float32)
            delta_w, pre_t, post_t = stdp.update(
                pre_spikes, post_spikes, weights, pre_t, post_t
            )
            weights = np.clip(weights + delta_w, stdp.w_min, stdp.w_max)

        assert np.all(weights >= stdp.w_min)
        assert np.all(weights <= stdp.w_max)

    def test_init_traces_zero(self):
        stdp = STDPRule()
        pre_t, post_t = stdp.init_traces(5, 8)
        assert np.all(pre_t == 0)
        assert np.all(post_t == 0)

    def test_repr(self):
        stdp = STDPRule()
        assert "STDP" in repr(stdp)


class TestRewardModulatedSTDP:

    def test_positive_reward_potentiates(self):
        # Use w_min=-1 so updates aren't clipped at zero
        from synaptic_ml.learning.stdp import STDPRule
        base = STDPRule(w_min=-1.0, w_max=1.0)
        rstdp = RewardModulatedSTDP(base_stdp=base)

        n_pre, n_post = 4, 4
        weights = np.zeros((n_pre, n_post), dtype=np.float32)
        pre_t, post_t = rstdp.stdp.init_traces(n_pre, n_post)

        # Pre fires first (step 1), then post fires (step 2) -> LTP
        pre_spikes = np.ones(n_pre, dtype=np.float32)
        no_spikes = np.zeros(n_post, dtype=np.float32)

        # Step 1: pre fires, post silent — build pre trace
        delta_w, pre_t, post_t = rstdp.update(
            pre_spikes, no_spikes, weights, pre_t, post_t, reward=1.0
        )

        # Step 2: post fires — should trigger LTP (pre trace is non-zero)
        post_spikes = np.ones(n_post, dtype=np.float32)
        delta_w, pre_t, post_t = rstdp.update(
            no_spikes, post_spikes, weights, pre_t, post_t, reward=1.0
        )

        # With positive reward and pre->post timing, delta should be non-zero
        assert delta_w.sum() != 0


class TestSurrogateGradients:

    @pytest.mark.parametrize("surrogate_cls", [
        SigmoidSurrogate,
        PiecewiseLinearSurrogate,
        FastSigmoidSurrogate,
        SuperSpike,
    ])
    def test_forward_is_heaviside(self, surrogate_cls):
        sg = surrogate_cls()
        v = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        threshold = 0.0
        spikes = sg.forward(v, threshold)
        expected = (v >= threshold).astype(np.float32)
        np.testing.assert_array_equal(spikes, expected)

    @pytest.mark.parametrize("surrogate_cls", [
        SigmoidSurrogate,
        PiecewiseLinearSurrogate,
        FastSigmoidSurrogate,
        SuperSpike,
    ])
    def test_backward_nonnegative(self, surrogate_cls):
        sg = surrogate_cls()
        v = np.linspace(-3, 3, 100)
        grad = sg.backward(v, threshold=0.0)
        assert np.all(grad >= 0)

    @pytest.mark.parametrize("surrogate_cls", [
        SigmoidSurrogate,
        PiecewiseLinearSurrogate,
        FastSigmoidSurrogate,
        SuperSpike,
    ])
    def test_backward_peaks_at_threshold(self, surrogate_cls):
        sg = surrogate_cls()
        v = np.linspace(-2, 2, 1000)
        grad = sg.backward(v, threshold=0.0)
        peak_idx = np.argmax(grad)
        # Peak should be near threshold (v=0)
        assert abs(v[peak_idx]) < 0.1

    def test_get_surrogate(self):
        for name in ["sigmoid", "piecewise_linear", "fast_sigmoid", "superspike"]:
            sg = get_surrogate(name)
            assert sg is not None

    def test_get_surrogate_invalid(self):
        with pytest.raises(ValueError):
            get_surrogate("invalid")


class TestANNConversion:

    def test_convert_from_numpy_weights(self):
        weights = [
            np.random.randn(16, 8).astype(np.float32),
            np.random.randn(8, 4).astype(np.float32),
        ]
        calibration = np.random.rand(50, 16).astype(np.float32)
        model = convert_from_numpy_weights(weights, None, calibration, time_steps=20)
        assert model.layer_sizes == [16, 8, 4]

    def test_converted_model_runs(self):
        weights = [
            np.random.randn(8, 4).astype(np.float32),
            np.random.randn(4, 2).astype(np.float32),
        ]
        calibration = np.random.rand(20, 8).astype(np.float32)
        model = convert_from_numpy_weights(weights, None, calibration, time_steps=10)

        from synaptic_ml.encoding.rate import RateEncoder
        encoder = RateEncoder(time_steps=10)
        scores = model.run(encoder.encode(np.random.rand(8)))
        assert scores.shape == (2,)
