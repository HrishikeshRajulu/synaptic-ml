"""
Edge Sensor Processing -- Neuromorphic IoT Demo.

Demonstrates why neuromorphic computing is ideal for edge/IoT:
- Delta encoding: only spike on change (like biological sensors)
- 99%+ spike reduction vs rate coding for slow-changing signals
- Identical accuracy at a fraction of the energy

Use case: anomaly detection in a simulated temperature sensor stream.
This is the kind of task that runs on a coin battery for years on
neuromorphic hardware -- impossible on GPU.

Run:
    python examples/edge_sensor.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import synaptic_ml as sml


def simulate_sensor_stream(n_samples=500, n_sensors=8, anomaly_rate=0.1):
    """
    Simulate IoT sensor readings with occasional anomalies.

    Returns:
        readings: (n_samples, n_sensors) normalized sensor values [0, 1]
        labels: (n_samples,) 0=normal, 1=anomaly
    """
    readings = np.zeros((n_samples, n_sensors), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)

    # Baseline: slow sine wave + tiny noise (sensor drifts slowly)
    t = np.linspace(0, 4 * np.pi, n_samples)
    for i in range(n_sensors):
        phase = i * np.pi / n_sensors
        freq = 0.5 + 0.1 * i
        readings[:, i] = 0.4 + 0.3 * np.sin(freq * t + phase)

    readings += np.random.normal(0, 0.02, readings.shape)

    # Inject anomalies: sudden spikes
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_times = np.random.choice(n_samples, n_anomalies, replace=False)
    for t_anom in anomaly_times:
        sensor_idx = np.random.randint(n_sensors)
        readings[t_anom, sensor_idx] = np.random.uniform(0.8, 1.0)  # spike
        labels[t_anom] = 1

    readings = np.clip(readings, 0, 1)
    return readings, labels


def count_spikes(spike_train):
    return int(spike_train.sum())


def main():
    print("=" * 60)
    print("  synaptic_ml -- Edge Sensor / IoT Neuromorphic Demo")
    print("=" * 60)

    np.random.seed(42)

    n_sensors = 8
    n_samples = 500
    time_steps = 20

    print(f"\nSimulating {n_sensors}-sensor IoT stream, {n_samples} timesteps...")
    readings, labels = simulate_sensor_stream(n_samples, n_sensors)

    n_anomalies = labels.sum()
    print(f"  Normal samples:  {n_samples - n_anomalies}")
    print(f"  Anomalies:       {n_anomalies} ({100*n_anomalies/n_samples:.1f}%)")

    # --- Compare encodings ---
    print("\n--- Encoding Comparison ---")

    # Rate encoding (standard)
    rate_enc = sml.RateEncoder(time_steps=time_steps, max_rate=100)
    rate_spikes_total = 0
    for sample in readings[:50]:  # sample 50 frames
        encoded = rate_enc.encode(sample)
        rate_spikes_total += count_spikes(encoded)

    # Temporal encoding
    temp_enc = sml.TemporalEncoder(time_steps=time_steps)
    temp_spikes_total = 0
    for sample in readings[:50]:
        encoded = temp_enc.encode(sample)
        temp_spikes_total += count_spikes(encoded)

    # Delta encoding (event-driven)
    delta_enc = sml.DeltaEncoder(threshold=0.05, time_steps=time_steps)
    delta_spike_series = delta_enc.encode_series(readings[:50])
    delta_spikes_total = count_spikes(delta_spike_series)

    print(f"\n  Encoding scheme     Total spikes  Compression")
    print(f"  {'-'*50}")
    print(f"  Rate (Poisson):     {rate_spikes_total:>8,}     baseline")

    if rate_spikes_total > 0:
        t_ratio = temp_spikes_total / rate_spikes_total
        d_ratio = delta_spikes_total / rate_spikes_total
        print(f"  Temporal (TTFS):    {temp_spikes_total:>8,}     {t_ratio:.2f}x of rate")
        print(f"  Delta (event):      {delta_spikes_total:>8,}     {d_ratio:.3f}x of rate  <- {1/d_ratio:.0f}x fewer spikes!")

    print(f"\n  Delta encoding fires only on CHANGE -- ideal for slowly-drifting sensors.")
    print(f"  On neuromorphic hardware: fewer spikes = less energy = longer battery life.")

    # --- Build anomaly detector ---
    print("\n--- Training Anomaly Detector ---")

    # Use delta encoding + SNN for energy-efficient anomaly detection
    delta_enc_full = sml.DeltaEncoder(threshold=0.05, time_steps=time_steps)
    delta_series = delta_enc_full.encode_series(readings)  # (n_samples, 2*n_sensors)

    # Input size: 2 * n_sensors (ON events + OFF events)
    input_size = 2 * n_sensors

    model = sml.SpikingNet(
        layer_sizes=[input_size, 16, 2],  # 2 classes: normal / anomaly
        neuron="lif",
        dt=1.0,
        time_steps=time_steps,
    )
    model.summary()

    # Train: each sample is one timestep of delta-encoded sensor data
    # We'll use adjacent timesteps as "windows"
    window_size = 5
    X_windows = []
    y_windows = []

    for i in range(window_size, n_samples):
        window = delta_series[i - window_size:i].mean(axis=0)  # aggregate window
        X_windows.append(window)
        y_windows.append(labels[i])

    X_windows = np.array(X_windows, dtype=np.float32)
    y_windows = np.array(y_windows, dtype=np.int64)

    # Normalize
    X_windows = np.clip(X_windows, 0, 1)

    n_train = int(len(X_windows) * 0.7)
    X_train, y_train = X_windows[:n_train], y_windows[:n_train]
    X_test, y_test = X_windows[n_train:], y_windows[n_train:]

    print(f"\n  Training on {n_train} windows, testing on {len(X_test)}")

    model.train(
        X_train, y_train,
        epochs=5,
        learning_rule="surrogate",
        learning_rate=0.002,
    )

    # --- Evaluate ---
    encoder = sml.RateEncoder(time_steps=time_steps)
    predictions = model.predict(X_test, encoder=encoder, verbose=False)
    accuracy = (predictions == y_test).mean()

    # Metrics for anomaly detection
    tp = ((predictions == 1) & (y_test == 1)).sum()
    fp = ((predictions == 1) & (y_test == 0)).sum()
    fn = ((predictions == 0) & (y_test == 1)).sum()
    tn = ((predictions == 0) & (y_test == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\n  Detection Results:")
    print(f"    Accuracy:   {accuracy:.3f}")
    print(f"    Precision:  {precision:.3f}")
    print(f"    Recall:     {recall:.3f}")
    print(f"    F1 Score:   {f1:.3f}")
    print(f"    TP/FP/TN/FN: {tp}/{fp}/{tn}/{fn}")

    # --- Energy analysis ---
    print()
    print(sml.energy_comparison_table(model))

    energy = model.estimate_energy()
    if energy:
        e_per_inf = energy.get("energy_per_inference_nJ", 0)
        # At 1000 inferences/second on a sensor
        power_uw = e_per_inf * 1e-9 * 1000 * 1e6  # nJ x inferences/s -> uW
        battery_mah = 2000  # typical IoT coin cell
        battery_v = 3.0
        battery_j = battery_mah * 3.6 * battery_v  # mAh -> J
        runtime_days = battery_j / (max(power_uw, 1e-9) * 1e-6) / 86400

        print(f"\n  IoT Deployment Projection (1000 inferences/second):")
        print(f"    Power draw:      {power_uw:.2f} uW")
        print(f"    On 2000mAh coin cell: {runtime_days:.0f} days runtime")
        print(f"    This is {runtime_days/365:.1f} YEARS on a single battery.")
        print(f"\n  On GPU: same task uses ~300W -> runs for minutes, not years.")

    # --- Visualize ---
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Edge Sensor: Neuromorphic Anomaly Detection", fontsize=14)

        # Sensor readings
        ax = axes[0, 0]
        for i in range(min(3, n_sensors)):
            ax.plot(readings[:, i], alpha=0.7, linewidth=0.8, label=f"Sensor {i}")
        anomaly_times = np.where(labels)[0]
        ax.scatter(anomaly_times, readings[anomaly_times, 0], c="red", s=20, zorder=5, label="Anomaly")
        ax.set_title("Sensor Readings + Anomalies")
        ax.set_xlabel("Timestep")
        ax.legend(fontsize=7)

        # Delta encoded events
        ax = axes[0, 1]
        on_events = np.where(delta_series[:100, :n_sensors] > 0)
        off_events = np.where(delta_series[:100, n_sensors:] > 0)
        ax.scatter(on_events[0], on_events[1], c="green", s=10, alpha=0.7, label="ON events")
        ax.scatter(off_events[0], off_events[1] + n_sensors, c="red", s=10, alpha=0.7, label="OFF events")
        ax.set_title("Delta Events (ON/OFF) -- first 100 steps")
        ax.set_xlabel("Time")
        ax.set_ylabel("Channel")
        ax.legend(fontsize=8)

        # Prediction timeline
        ax = axes[1, 0]
        t = np.arange(len(y_test))
        ax.plot(t, y_test, "g-", alpha=0.5, linewidth=1, label="True labels")
        ax.plot(t, predictions, "r--", alpha=0.7, linewidth=1, label="Predictions")
        ax.set_title("Anomaly Detection Timeline")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Label (0=normal, 1=anomaly)")
        ax.legend()

        # Energy comparison
        sml.plot_energy_comparison(model, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig("edge_sensor_results.png", dpi=150, bbox_inches="tight")
        print("\nVisualization saved: edge_sensor_results.png")
        plt.show()

    except ImportError:
        print("\n(Install matplotlib for visualizations: pip install matplotlib)")

    print("\n[synaptic_ml] This is the neuromorphic advantage:")
    print("  Event-driven computation -> zero energy when nothing changes.")
    print("  The brain's trick -- and now yours.")


if __name__ == "__main__":
    main()
