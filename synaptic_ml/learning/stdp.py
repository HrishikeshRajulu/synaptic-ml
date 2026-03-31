"""
Spike-Timing Dependent Plasticity (STDP) — Hebbian learning for SNNs.

"Neurons that fire together, wire together."
More precisely: if pre fires BEFORE post → strengthen synapse.
If pre fires AFTER post → weaken synapse.

The timing window is exponential: Δw ∝ exp(-|Δt| / τ)
"""

import numpy as np


class STDPRule:
    """
    Classic STDP learning rule (Bi & Poo, 1998).

    Δw = A+ * exp(-Δt/τ+)  if post fires after pre  (potentiation)
    Δw = -A- * exp(Δt/τ-)  if pre fires after post   (depression)

    Parameters
    ----------
    A_plus : float
        LTP (long-term potentiation) amplitude (default: 0.01).
    A_minus : float
        LTD (long-term depression) amplitude (default: 0.012).
    tau_plus : float
        LTP time constant in ms (default: 20ms).
    tau_minus : float
        LTD time constant in ms (default: 20ms).
    w_min, w_max : float
        Weight bounds.
    """

    def __init__(
        self,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
    ):
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max

    def update(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        weights: np.ndarray,
        pre_trace: np.ndarray,
        post_trace: np.ndarray,
        dt: float = 1.0,
    ) -> tuple:
        """
        Compute weight update for one timestep using eligibility traces.

        Parameters
        ----------
        pre_spikes : np.ndarray, shape (n_pre,)
        post_spikes : np.ndarray, shape (n_post,)
        weights : np.ndarray, shape (n_pre, n_post)
        pre_trace : np.ndarray, shape (n_pre,) — exponentially decaying trace
        post_trace : np.ndarray, shape (n_post,) — exponentially decaying trace
        dt : float

        Returns
        -------
        delta_w : np.ndarray, shape (n_pre, n_post)
        new_pre_trace : np.ndarray
        new_post_trace : np.ndarray
        """
        # Decay traces
        pre_trace = pre_trace * np.exp(-dt / self.tau_plus) + pre_spikes
        post_trace = post_trace * np.exp(-dt / self.tau_minus) + post_spikes

        # LTP: pre spike → update based on post trace
        ltp = np.outer(pre_spikes, post_trace) * self.A_plus

        # LTD: post spike → update based on pre trace
        ltd = np.outer(pre_trace, post_spikes) * self.A_minus

        delta_w = ltp - ltd
        new_weights = np.clip(weights + delta_w, self.w_min, self.w_max)
        delta_w = new_weights - weights

        return delta_w, pre_trace, post_trace

    def init_traces(self, n_pre: int, n_post: int) -> tuple:
        """Initialize eligibility traces to zero."""
        return np.zeros(n_pre, dtype=np.float32), np.zeros(n_post, dtype=np.float32)

    def __repr__(self) -> str:
        return (
            f"STDPRule(A+={self.A_plus}, A-={self.A_minus}, "
            f"τ+={self.tau_plus}ms, τ-={self.tau_minus}ms)"
        )


class RewardModulatedSTDP:
    """
    R-STDP: STDP modulated by a reward/error signal.

    Combines the biological plausibility of STDP with reinforcement
    learning. The weight update is gated by a neuromodulator signal
    (dopamine in the brain).

    Δw = reward * STDP_eligibility_trace

    Parameters
    ----------
    base_stdp : STDPRule
    reward_decay : float
        Decay rate of reward signal (default: 0.95).
    """

    def __init__(self, base_stdp: STDPRule = None, reward_decay: float = 0.95):
        self.stdp = base_stdp or STDPRule()
        self.reward_decay = reward_decay
        self._eligibility = None

    def update(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        weights: np.ndarray,
        pre_trace: np.ndarray,
        post_trace: np.ndarray,
        reward: float,
        dt: float = 1.0,
    ) -> tuple:
        """
        Compute R-STDP weight update.

        Parameters
        ----------
        reward : float
            Scalar reward signal (+1 correct, -1 incorrect, 0 neutral).
        """
        delta_w_raw, pre_trace, post_trace = self.stdp.update(
            pre_spikes, post_spikes, weights, pre_trace, post_trace, dt
        )

        # Maintain eligibility trace
        if self._eligibility is None:
            self._eligibility = np.zeros_like(delta_w_raw)
        self._eligibility = self._eligibility * self.reward_decay + delta_w_raw

        # Apply reward gating
        delta_w = reward * self._eligibility

        new_weights = np.clip(weights + delta_w, self.stdp.w_min, self.stdp.w_max)
        return new_weights - weights, pre_trace, post_trace

    def reset(self):
        self._eligibility = None

    def __repr__(self) -> str:
        return f"RewardModulatedSTDP(decay={self.reward_decay})"
