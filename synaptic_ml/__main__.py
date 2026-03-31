"""Entry point for `python -m synaptic_ml` and `synaptic-info` CLI."""

import synaptic_ml as sml
import numpy as np


def main():
    print(f"\nsynaptic_ml v{sml.__version__} — The TensorFlow for Neuromorphic Computing")
    print("=" * 60)

    # Quick smoke test
    print("\nRunning quick self-test...")
    model = sml.SpikingNet([16, 8, 4], neuron="lif", time_steps=20)

    encoder = sml.RateEncoder(time_steps=20)
    x = np.random.rand(16).astype(np.float32)
    encoded = encoder.encode(x)
    scores = model.run(encoded)

    print(f"  Model: {model}")
    print(f"  Input shape: {x.shape}")
    print(f"  Encoded spike train: {encoded.shape}")
    print(f"  Output scores: {scores}")
    print(f"  Predicted class: {np.argmax(scores)}")

    print()
    print(sml.energy_comparison_table(model))

    print("\nAvailable backends:")
    for name in ["cpu", "loihi2", "brainscales"]:
        try:
            backend = sml.get_backend(name)
            print(f"  {name}: {backend}")
        except Exception as e:
            print(f"  {name}: {e}")

    print("\nQuick start:")
    print("  import synaptic_ml as sml")
    print("  model = sml.SpikingNet([784, 256, 10])")
    print("  model.train(X_train, y_train, epochs=10)")
    print("  model.deploy(target='loihi2')")
    print()


if __name__ == "__main__":
    main()
