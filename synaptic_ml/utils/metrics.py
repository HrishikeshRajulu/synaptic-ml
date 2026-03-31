"""
Metrics and energy estimation for spiking neural networks.
"""

import numpy as np


def spike_rate(spike_train: np.ndarray) -> np.ndarray:
    """
    Compute mean firing rate from spike train.

    Parameters
    ----------
    spike_train : np.ndarray, shape (T, n_neurons)

    Returns
    -------
    rates : np.ndarray, shape (n_neurons,) -- spikes per timestep
    """
    return spike_train.mean(axis=0)


def synaptic_operations(network, spike_trains: list) -> int:
    """
    Count total synaptic operations during inference.

    A synaptic operation occurs each time a presynaptic neuron fires
    and its spike propagates through a synapse.

    Parameters
    ----------
    network : SpikingNet
    spike_trains : list of np.ndarray
        One per layer, shape (T, n_neurons).

    Returns
    -------
    total_ops : int
    """
    total = 0
    layers_with_weights = [l for l in network.layers if hasattr(l, "weights")]

    for i, layer in enumerate(layers_with_weights):
        if i < len(spike_trains):
            # Each presynaptic spike x n_postsynaptic connections
            n_pre_spikes = int(spike_trains[i].sum())
            n_post = layer.weights.shape[1]
            total += n_pre_spikes * n_post

    return total


def estimate_energy_joules(synaptic_ops: int, backend: str = "cpu") -> float:
    """
    Estimate energy consumption in joules.

    Based on published neuromorphic hardware benchmarks.

    Parameters
    ----------
    synaptic_ops : int
        Total number of synaptic operations.
    backend : str
        'loihi2' (8.6 fJ/op), 'brainscales' (~1 pJ/op), 'cpu' (estimates Loihi 2).

    Returns
    -------
    energy_joules : float
    """
    # Energy per synaptic op (femtojoules -> joules)
    energy_per_op = {
        "loihi2": 8.6e-15,       # 8.6 fJ (Intel benchmark)
        "brainscales": 1e-12,    # ~1 pJ (analog circuits)
        "cpu": 8.6e-15,          # Estimate as Loihi 2 for projection purposes
    }

    fj_per_op = energy_per_op.get(backend, 8.6e-15)
    return synaptic_ops * fj_per_op


def van_rossum_distance(s1: np.ndarray, s2: np.ndarray, tau: float = 10.0) -> float:
    """
    Van Rossum distance between two spike trains.

    Convolves each train with an exponential kernel, then computes L2 distance.
    tau: time constant of the kernel in timesteps.

    Parameters
    ----------
    s1, s2 : np.ndarray, shape (T,) -- binary spike trains
    tau : float -- kernel time constant

    Returns
    -------
    distance : float
    """
    T = max(len(s1), len(s2))
    s1 = np.pad(s1, (0, T - len(s1)))
    s2 = np.pad(s2, (0, T - len(s2)))

    # Exponential kernel
    t = np.arange(T, dtype=np.float32)
    kernel = np.exp(-t / tau)

    # Convolve
    f1 = np.convolve(s1, kernel, mode="full")[:T]
    f2 = np.convolve(s2, kernel, mode="full")[:T]

    return float(np.sqrt(np.sum((f1 - f2) ** 2)))


def coincidence_factor(s1: np.ndarray, s2: np.ndarray, delta: float = 2.0) -> float:
    """
    Gamma coincidence factor -- measures spike train similarity.

    Counts spikes in s2 within ±delta timesteps of each spike in s1.
    Returns a value in [-1, 1]: 1 = perfect match, 0 = random, negative = anticorrelated.

    Parameters
    ----------
    s1, s2 : np.ndarray, shape (T,)
    delta : float -- coincidence window in timesteps

    Returns
    -------
    gamma : float
    """
    n_coincidences = 0
    spike_times_1 = np.where(s1 > 0)[0]
    spike_times_2 = np.where(s2 > 0)[0]

    for t in spike_times_1:
        if np.any(np.abs(spike_times_2 - t) <= delta):
            n_coincidences += 1

    n1 = len(spike_times_1)
    n2 = len(spike_times_2)
    T = len(s1)

    if n1 == 0 or n2 == 0:
        return 0.0

    # Expected coincidences by chance
    n_expected = 2 * delta * n1 * n2 / T

    gamma = (n_coincidences - n_expected) / (0.5 * (n1 + n2))
    return float(np.clip(gamma, -1.0, 1.0))


def energy_comparison_table(model, n_inferences: int = 1000) -> str:
    """
    Generate a human-readable energy comparison table.

    Compares neuromorphic hardware vs GPU for the given model.

    Returns
    -------
    table : str
    """
    # Estimate synaptic ops
    n_params = sum(l.weights.size for l in model.layers if hasattr(l, "weights"))
    avg_sparsity = 0.05  # typical SNN spike rate ~5%
    ops_per_inf = n_params * avg_sparsity * model.time_steps

    loihi_energy_nJ = ops_per_inf * 8.6e-15 * 1e9
    gpu_flops = n_params * 2  # forward pass FLOPs (multiply + add)
    gpu_energy_nJ = gpu_flops * 50e-12 * 1e9  # 50 pJ/FLOP

    speedup = gpu_energy_nJ / (loihi_energy_nJ + 1e-20)

    lines = [
        "=" * 58,
        "  Energy Comparison: Neuromorphic vs GPU",
        "=" * 58,
        f"  Model parameters:   {n_params:,}",
        f"  Avg spike rate:     {avg_sparsity*100:.0f}% (sparse!)",
        f"  Ops per inference:  {ops_per_inf:,.0f}",
        "-" * 58,
        f"  {'Hardware':<20} {'Energy/inference':>16} {'Notes':<18}",
        f"  {'-'*20} {'-'*16} {'-'*18}",
        f"  {'GPU (A100)':<20} {gpu_energy_nJ:>12.1f} nJ   300W total",
        f"  {'Intel Loihi 2':<20} {loihi_energy_nJ:>12.4f} nJ   30mW total",
        f"  {'BrainScaleS-2':<20} {'~analog':>12}     1uW/neuron",
        "-" * 58,
        f"  Efficiency gain:    {speedup:,.0f}x less energy on Loihi 2",
        f"  This is the {speedup/1e6:.1f}Mx brain-vs-datacenter gap -- closing it.",
        "=" * 58,
    ]
    return "\n".join(lines)
