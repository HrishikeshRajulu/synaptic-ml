"""
MNIST Classification with Spiking Neural Networks.

This example trains a 784 -> 256 -> 10 LIF network on MNIST using
surrogate gradient learning, then shows the energy advantage vs GPU.

Run:
    python examples/mnist_lif.py

Requirements:
    pip install synaptic-ml scikit-learn
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import synaptic_ml as sml


def load_mnist_sklearn():
    """Load MNIST via scikit-learn (downloads ~55MB on first run)."""
    try:
        from sklearn.datasets import fetch_openml
        print("Loading MNIST (may download ~55MB on first run)...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int64)
        return X, y
    except ImportError:
        print("scikit-learn not found. Using synthetic data instead.")
        return None, None


def make_synthetic_data(n_train=1000, n_test=200, n_features=784, n_classes=10):
    """Generate synthetic classification data when MNIST isn't available."""
    print(f"Generating synthetic data: {n_train} train, {n_test} test samples")
    X = np.random.rand(n_train + n_test, n_features).astype(np.float32)
    # Make it slightly structured: class i has higher values in region i
    y = np.random.randint(0, n_classes, n_train + n_test)
    for i in range(n_classes):
        mask = y == i
        region = slice(i * (n_features // n_classes), (i + 1) * (n_features // n_classes))
        X[mask, region] += 0.5
    X = np.clip(X, 0, 1)
    return (X[:n_train], y[:n_train]), (X[n_train:], y[n_train:])


def main():
    print("=" * 60)
    print("  synaptic_ml — MNIST with Spiking Neural Networks")
    print("=" * 60)

    # --- Load data ---
    X, y = load_mnist_sklearn()

    if X is not None:
        # Use a subset for faster training
        n_train, n_test = 5000, 1000
        idx = np.random.permutation(len(X))
        X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
        X_test, y_test = X[idx[n_train:n_train + n_test]], y[idx[n_train:n_train + n_test]]
    else:
        (X_train, y_train), (X_test, y_test) = make_synthetic_data()

    print(f"\nDataset: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Input:   {X_train.shape[1]} features (pixels)")
    print(f"Classes: {len(np.unique(y_train))}")

    # --- Build model ---
    print("\nBuilding SpikingNet...")
    model = sml.SpikingNet(
        layer_sizes=[X_train.shape[1], 256, 10],
        neuron="lif",
        dt=1.0,
        time_steps=50,  # 50ms simulation window
    )
    model.summary()

    # --- Train ---
    print("\nTraining with surrogate gradients...")
    model.train(
        X_train, y_train,
        epochs=5,
        learning_rule="surrogate",
        learning_rate=0.001,
    )

    # --- Evaluate ---
    print("\nEvaluating on test set...")
    encoder = sml.RateEncoder(time_steps=50)
    predictions = model.predict(X_test, encoder=encoder)
    accuracy = (predictions == y_test).mean()
    print(f"\nTest Accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)})")

    # --- Energy comparison ---
    print()
    print(sml.energy_comparison_table(model))

    energy = model.estimate_energy()
    if energy:
        print(f"\nActual inference stats:")
        print(f"  Ops/inference:    {energy['synaptic_ops_per_inference']:,.0f}")
        print(f"  Energy/inference: {energy['energy_per_inference_nJ']:.4f} nJ")
        print(f"  GPU equivalent:   {energy['gpu_energy_per_inference_nJ']:.2f} nJ")
        print(f"  Efficiency gain:  {energy['efficiency_gain']:,.0f}x")

    # --- Deploy ---
    print("\nDeploying to CPU backend (simulation)...")
    backend = model.deploy(target="cpu")

    print("\n[synaptic_ml] For real hardware deployment:")
    print("  model.deploy(target='loihi2')    # Intel Loihi 2")
    print("  model.deploy(target='brainscales')  # BrainScaleS-2")

    # --- Save model ---
    model.save("mnist_snn.snm")
    loaded = sml.SpikingNet.load("mnist_snn.snm")
    print(f"\nLoaded back: {loaded}")

    # --- Visualize (if matplotlib available) ---
    try:
        import matplotlib.pyplot as plt

        encoder = sml.RateEncoder(time_steps=50)
        encoded = encoder.encode(X_test[0])
        model.run(encoded)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Spike raster of hidden layer
        hidden = model.layers[1]
        if hasattr(hidden, "spike_history") and hidden.spike_history:
            spike_array = np.array(hidden.spike_history)
            sml.plot_raster(spike_array, title="Hidden Layer Spikes", dt=1.0, ax=axes[0])

        # Membrane potential (first 5 neurons)
        if hasattr(hidden, "voltage_history") and hidden.voltage_history:
            v_array = np.array(hidden.voltage_history)
            sml.plot_membrane(v_array, title="Membrane Potentials", ax=axes[1])

        # Energy comparison
        sml.plot_energy_comparison(model, ax=axes[2])

        plt.tight_layout()
        plt.savefig("mnist_snn_results.png", dpi=150, bbox_inches="tight")
        print("\nVisualization saved: mnist_snn_results.png")
        plt.show()

    except ImportError:
        print("\n(Install matplotlib for visualizations: pip install matplotlib)")


if __name__ == "__main__":
    main()
