"""
Visualization tools for spiking neural networks.

Spike raster plots, membrane traces, weight distributions,
and the dramatic energy comparison bar chart.
"""

import numpy as np


def plot_raster(spike_trains: np.ndarray, title: str = "Spike Raster", dt: float = 1.0, ax=None):
    """
    Plot a spike raster: each row is a neuron, each dot is a spike.

    Parameters
    ----------
    spike_trains : np.ndarray, shape (T, n_neurons)
    title : str
    dt : float — timestep in ms (for x-axis)
    ax : matplotlib axis (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return

    T, n = spike_trains.shape
    t_ms = np.arange(T) * dt

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, max(3, n // 10)))

    spike_times, neuron_ids = np.where(spike_trains.T > 0)
    ax.scatter(t_ms[neuron_ids], spike_times, s=1, c="black", alpha=0.7)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    ax.set_xlim(0, T * dt)
    ax.set_ylim(-1, n)

    total_spikes = int(spike_trains.sum())
    rate = total_spikes / (T * n + 1e-8)
    ax.text(
        0.98, 0.98, f"Spikes: {total_spikes:,}  Rate: {rate:.3f}/step",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7),
    )

    return ax


def plot_membrane(
    voltages: np.ndarray,
    threshold: float = -50.0,
    neuron_indices: list = None,
    title: str = "Membrane Potentials",
    dt: float = 1.0,
    ax=None,
):
    """
    Plot membrane potential traces over time.

    Parameters
    ----------
    voltages : np.ndarray, shape (T, n_neurons)
    threshold : float — spike threshold line
    neuron_indices : list — which neurons to plot (default: first 5)
    title : str
    dt : float
    ax : matplotlib axis
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return

    T, n = voltages.shape
    t_ms = np.arange(T) * dt

    if neuron_indices is None:
        neuron_indices = list(range(min(5, n)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    for i in neuron_indices:
        ax.plot(t_ms, voltages[:, i], linewidth=0.8, label=f"Neuron {i}", alpha=0.8)

    ax.axhline(threshold, color="red", linestyle="--", linewidth=1, label=f"Threshold ({threshold}mV)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane potential (mV)")
    ax.set_title(title)
    ax.legend(fontsize=8)

    return ax


def plot_weight_distribution(weights: np.ndarray, title: str = "Synaptic Weight Distribution", ax=None):
    """
    Histogram of synaptic weights.

    Parameters
    ----------
    weights : np.ndarray, shape (n_pre, n_post)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    flat = weights.flatten()
    ax.hist(flat, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(flat.mean(), color="red", linestyle="--", label=f"Mean={flat.mean():.3f}")
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    return ax


def plot_energy_comparison(
    model,
    title: str = "Energy: Neuromorphic vs GPU",
    ax=None,
):
    """
    Bar chart comparing energy consumption on different hardware.

    Shows the dramatic energy advantage of neuromorphic computing.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return

    n_params = sum(l.weights.size for l in model.layers if hasattr(l, "weights"))
    sparsity = 0.05
    ops_per_inf = n_params * sparsity * model.time_steps

    # Energy estimates in nanojoules
    energies = {
        "GPU (A100)\n300W": n_params * 2 * 50e-12 * 1e9,
        "CPU (x86)\n65W": n_params * 2 * 200e-12 * 1e9,
        "Loihi 2\n30mW": ops_per_inf * 8.6e-15 * 1e9,
        "Brain\n20W": ops_per_inf * 1e-15 * 1e9,
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    names = list(energies.keys())
    values = list(energies.values())
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]

    bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Energy per inference (nJ) — log scale")
    ax.set_title(title)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.5,
            f"{val:.2e} nJ",
            ha="center", fontsize=8,
        )

    # Annotate the gap
    gpu_e = values[0]
    loihi_e = values[2]
    ax.annotate(
        f"{gpu_e/loihi_e:,.0f}× less energy",
        xy=(2, loihi_e),
        xytext=(0.5, gpu_e * 0.1),
        arrowprops=dict(arrowstyle="->", color="green"),
        color="green", fontsize=10, fontweight="bold",
    )

    return ax


def plot_training_history(train_losses: list, train_accuracies: list):
    """Plot loss and accuracy curves from training."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, "b-o", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_accuracies, "g-o", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig
