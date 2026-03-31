"""
CPU simulation backend — always available, no hardware required.

This is the default backend. Uses pure numpy to simulate spiking
dynamics. Slower than real hardware but identical results.

Energy estimates are based on:
- Each synaptic operation: ~10 fJ (femtojoules) on Loihi 2
- For comparison: each multiply-accumulate on GPU: ~50 pJ (picojoules)
- Ratio: GPU is ~5000× less efficient than neuromorphic per operation
"""

import numpy as np
from .base import Backend


class CPUBackend(Backend):
    """
    Pure CPU/numpy simulation backend.

    Simulates spiking neural networks at full biological fidelity.
    Useful for:
    - Development and testing (no hardware needed)
    - Verifying model behavior before deploying to real chips
    - Research on network dynamics

    Energy estimates are projections of what Loihi 2 would consume.
    """

    # Energy constants (approximate, based on published benchmarks)
    ENERGY_PER_SYNAPTIC_OP_FJ = 10.0    # femtojoules per op on Loihi 2
    GPU_ENERGY_PER_FLOP_PJ = 50.0       # picojoules per FLOP on A100

    def __init__(self):
        self.network = None
        self._total_ops = 0
        self._inference_count = 0

    def load_network(self, network) -> None:
        self.network = network
        self._total_ops = 0
        self._inference_count = 0

    def run(self, input_spikes: np.ndarray, time_steps: int = None) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("No network loaded. Call load_network() first.")

        net = self.network
        if time_steps is None:
            time_steps = net.time_steps

        net._reset_all()
        spike_count = 0

        for t in range(time_steps):
            x = input_spikes[t] if t < len(input_spikes) else np.zeros(net.layer_sizes[0])
            spikes = x
            for layer in net.layers:
                spikes = layer.forward(spikes, net.dt)
            spike_count += int(np.sum(spikes))

        self._total_ops += spike_count
        self._inference_count += 1

        return net.layers[-1].decode()

    def get_energy_estimate(self) -> dict:
        if self._inference_count == 0:
            return {}

        ops_per_inf = self._total_ops / self._inference_count

        # Neuromorphic energy estimate (what Loihi 2 would use)
        neuro_energy_nJ = ops_per_inf * self.ENERGY_PER_SYNAPTIC_OP_FJ * 1e-6  # fJ → nJ

        # GPU comparison
        n_params = sum(
            layer.weights.size for layer in self.network.layers if hasattr(layer, "weights")
        )
        gpu_energy_nJ = n_params * 2 * self.GPU_ENERGY_PER_FLOP_PJ * 1e-3  # pJ → nJ

        return {
            "synaptic_ops_per_inference": ops_per_inf,
            "neuromorphic_energy_nJ": neuro_energy_nJ,
            "gpu_equivalent_energy_nJ": gpu_energy_nJ,
            "efficiency_gain": gpu_energy_nJ / (neuro_energy_nJ + 1e-20),
        }

    def get_hardware_info(self) -> str:
        return (
            "  Backend:     CPU Simulation (numpy)\n"
            "  Hardware:    Your CPU — simulating neuromorphic dynamics\n"
            "  Note:        Accurate simulation, not optimized for speed\n"
            "  Energy est.: Projecting Loihi 2 consumption\n"
            "  Deploy on:   Any machine with Python + numpy"
        )

    def __repr__(self) -> str:
        return "CPUBackend(numpy simulation)"
