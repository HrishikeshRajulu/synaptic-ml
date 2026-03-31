"""
Unsupervised Pattern Recognition with STDP.

Shows how STDP (spike-timing dependent plasticity) allows networks to
autonomously discover structure in data — no labels required.

This is how the brain learns: through local, unsupervised Hebbian rules.

Run:
    python examples/pattern_recognition.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import synaptic_ml as sml


def make_patterns(n_patterns=3, pattern_size=20, n_samples=300, noise=0.1):
    """
    Generate synthetic binary patterns with noise.

    Each sample is one of n_patterns with additive noise.
    The SNN should learn to recognize these without any labels.
    """
    # Define clean patterns
    base_patterns = []
    for i in range(n_patterns):
        p = np.zeros(pattern_size, dtype=np.float32)
        # Each pattern activates a different region
        region = slice(i * (pattern_size // n_patterns), (i + 1) * (pattern_size // n_patterns))
        p[region] = 1.0
        base_patterns.append(p)

    # Generate noisy samples
    X = []
    labels = []
    for _ in range(n_samples):
        idx = np.random.randint(n_patterns)
        pattern = base_patterns[idx].copy()
        # Add noise
        pattern += np.random.rand(pattern_size) * noise
        pattern = np.clip(pattern, 0, 1)
        X.append(pattern)
        labels.append(idx)

    return np.array(X), np.array(labels), base_patterns


def main():
    print("=" * 60)
    print("  synaptic_ml — Unsupervised Pattern Learning (STDP)")
    print("=" * 60)

    np.random.seed(42)

    # --- Generate patterns ---
    n_patterns = 3
    pattern_size = 20
    n_samples = 300

    print(f"\nGenerating {n_samples} samples of {n_patterns} patterns ({pattern_size}D)...")
    X, labels, base_patterns = make_patterns(n_patterns, pattern_size, n_samples)

    print("Base patterns:")
    for i, p in enumerate(base_patterns):
        bar = "".join("#" if v > 0.5 else "." for v in p)
        print(f"  Pattern {i}: {bar}")

    # --- Build network ---
    # Input -> hidden (STDP learns) -> output (one neuron per pattern ideally)
    model = sml.SpikingNet(
        layer_sizes=[pattern_size, 10, n_patterns],
        neuron="lif",
        dt=1.0,
        time_steps=30,
    )
    model.summary()

    # --- Train with STDP (unsupervised) ---
    print("\nTraining with STDP (no labels used)...")
    model.train(
        X, labels,  # labels used only for accuracy tracking, not for STDP update
        epochs=3,
        learning_rule="stdp",
        learning_rate=0.01,
    )

    # --- Visualize what STDP learned ---
    print("\nAnalyzing learned representations...")
    encoder = sml.RateEncoder(time_steps=30)

    # Run each clean pattern through the network
    print("\nNetwork responses to clean patterns:")
    for i, pattern in enumerate(base_patterns):
        encoded = encoder.encode(pattern)
        scores = model.run(encoded)
        predicted = int(np.argmax(scores))
        bar = " ".join(f"{s:.2f}" for s in scores)
        print(f"  Pattern {i} -> output rates: [{bar}] -> predicted class: {predicted}")

    # --- Show weight structure ---
    hidden_layer = model.layers[1]
    print(f"\nHidden layer weight stats:")
    w = hidden_layer.weights
    print(f"  Shape:  {w.shape}")
    print(f"  Mean:   {w.mean():.4f}")
    print(f"  Std:    {w.std():.4f}")
    print(f"  Min/Max: [{w.min():.4f}, {w.max():.4f}]")

    # --- Visualize ---
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("STDP Unsupervised Pattern Learning", fontsize=14)

        # Weight matrix heatmap
        ax = axes[0, 0]
        im = ax.imshow(hidden_layer.weights.T, cmap="RdBu_r", aspect="auto")
        ax.set_title("Learned Synaptic Weights (Input -> Hidden)")
        ax.set_xlabel("Input neuron")
        ax.set_ylabel("Hidden neuron")
        plt.colorbar(im, ax=ax)

        # Pattern activations
        ax = axes[0, 1]
        responses = []
        for pattern in base_patterns:
            encoded = encoder.encode(pattern)
            scores = model.run(encoded)
            responses.append(scores)
        responses = np.array(responses)
        im2 = ax.imshow(responses, cmap="Greens", aspect="auto")
        ax.set_title("Output Responses to Each Pattern")
        ax.set_xlabel("Output neuron")
        ax.set_ylabel("Pattern index")
        ax.set_yticks(range(n_patterns))
        ax.set_yticklabels([f"Pattern {i}" for i in range(n_patterns)])
        plt.colorbar(im2, ax=ax)

        # Spike raster for one pattern
        encoded = encoder.encode(base_patterns[0])
        model.run(encoded)
        if hasattr(hidden_layer, "spike_history") and hidden_layer.spike_history:
            spike_array = np.array(hidden_layer.spike_history)
            sml.plot_raster(spike_array, title="Hidden Layer Raster (Pattern 0)", ax=axes[1, 0])

        # Weight distribution
        sml.plot_weight_distribution(hidden_layer.weights, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig("stdp_patterns.png", dpi=150, bbox_inches="tight")
        print("\nVisualization saved: stdp_patterns.png")
        plt.show()

    except ImportError:
        print("\n(Install matplotlib for visualizations: pip install matplotlib)")

    print("\nKey insight: STDP learns structure WITHOUT labels.")
    print("This is how biological brains discover features in sensory data.")


if __name__ == "__main__":
    main()
