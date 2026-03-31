"""
BrainScaleS-2 backend.

BrainScaleS-2 is a European neuromorphic supercomputer at Heidelberg University.
It runs 1000× faster than biological real time using analog circuits.

Access: https://electronicvisions.github.io/hxtorch/
Requires: pip install pynn_brainscales hxtorch

Key specs:
- 512 neurons per chip (analog, continuous-time)
- 130,000 synapses per chip
- Runs 1000× faster than real time
- ~1 µW per neuron
"""

import numpy as np
from .base import Backend, BackendNotAvailableError


class BrainScalesBackend(Backend):
    """
    BrainScaleS-2 neuromorphic backend (Heidelberg University).

    Access via the EBRAINS research infrastructure:
    https://www.ebrains.eu/tools/brainscales

    This backend uses PyNN as the interface layer, mapping synaptic_ml
    networks to BrainScaleS neuron populations and projections.
    """

    MAX_NEURONS_PER_CHIP = 512
    MAX_SYNAPSES_PER_CHIP = 131_072
    SPEEDUP_FACTOR = 1000  # 1000× faster than biological real time
    ENERGY_PER_NEURON_UW = 1.0  # ~1 µW per neuron

    def __init__(self):
        self._check_sdk()
        self.network = None
        self._populations = []
        self._projections = []

    def _check_sdk(self):
        try:
            import pynn_brainscales  # noqa: F401
            self._sdk_available = True
        except ImportError:
            self._sdk_available = False

    def _require_sdk(self):
        if not self._sdk_available:
            raise BackendNotAvailableError(
                "\n[synaptic_ml] pynn_brainscales not found.\n\n"
                "To deploy on BrainScaleS-2:\n"
                "  1. Get EBRAINS access: https://www.ebrains.eu/\n"
                "  2. Install: pip install pynn_brainscales hxtorch\n"
                "  3. Access is granted via the Human Brain Project infrastructure\n\n"
                "For now, use CPUBackend:\n"
                "  model.deploy(target='cpu')\n"
            )

    def load_network(self, network) -> None:
        """Map SpikingNet to PyNN populations for BrainScaleS."""
        self._require_sdk()

        total_neurons = sum(network.layer_sizes)
        if total_neurons > self.MAX_NEURONS_PER_CHIP:
            raise ValueError(
                f"Network has {total_neurons} neurons but BrainScaleS-2 supports "
                f"max {self.MAX_NEURONS_PER_CHIP} per chip. Use multi-chip mode."
            )

        import pyNN.brainscales2 as sim

        self.network = network
        sim.setup(timestep=network.dt)

        self._populations = []
        for i, size in enumerate(network.layer_sizes):
            cell_params = {
                "v_rest": -65.0,
                "v_thresh": -50.0,
                "v_reset": -65.0,
                "tau_m": 20.0,
                "tau_refrac": 2.0,
            }
            pop = sim.Population(size, sim.IF_curr_exp(**cell_params))
            self._populations.append(pop)

        self._projections = []
        for i in range(len(network.layer_sizes) - 1):
            layer = network.layers[i + 1]
            if hasattr(layer, "weights"):
                w = layer.weights
                # PyNN uses list-of-tuples connection format
                connections = [
                    (pre, post, float(w[pre, post]), network.dt)
                    for pre in range(w.shape[0])
                    for post in range(w.shape[1])
                    if abs(w[pre, post]) > 1e-6
                ]
                proj = sim.Projection(
                    self._populations[i],
                    self._populations[i + 1],
                    sim.FromListConnector(connections),
                    receptor_type="excitatory",
                )
                self._projections.append(proj)

        print(f"[BrainScaleS] Mapped {total_neurons} neurons, "
              f"{len(self._projections)} projection groups")

    def run(self, input_spikes: np.ndarray, time_steps: int = None) -> np.ndarray:
        self._require_sdk()

        import pyNN.brainscales2 as sim

        time_steps = time_steps or self.network.time_steps
        duration = time_steps * self.network.dt  # ms

        # Inject input spikes
        spike_times = []
        for t in range(len(input_spikes)):
            active = np.where(input_spikes[t] > 0)[0]
            for neuron in active:
                spike_times.append((neuron, t * self.network.dt))

        # Record from output layer
        self._populations[-1].record("spikes")

        sim.run(duration)

        spike_data = self._populations[-1].get_data("spikes").segments[0].spiketrains
        output = np.array([len(st) for st in spike_data], dtype=np.float32)
        output = output / (time_steps + 1e-8)

        sim.end()
        return output

    def get_energy_estimate(self) -> dict:
        if self.network is None:
            return {}
        total_neurons = sum(self.network.layer_sizes)
        duration_s = self.network.time_steps * self.network.dt * 1e-3
        energy_nJ = total_neurons * self.ENERGY_PER_NEURON_UW * duration_s * 1e3  # µW × s → nJ
        return {
            "energy_nJ": energy_nJ,
            "speedup_vs_realtime": self.SPEEDUP_FACTOR,
            "chip": "BrainScaleS-2",
        }

    def get_hardware_info(self) -> str:
        sdk_status = "Available" if self._sdk_available else "Not installed"
        return (
            f"  Backend:     BrainScaleS-2 (Heidelberg University)\n"
            f"  pynn_brainscales: {sdk_status}\n"
            f"  Max neurons: {self.MAX_NEURONS_PER_CHIP} per chip (analog)\n"
            f"  Speed:       {self.SPEEDUP_FACTOR}× biological real time\n"
            f"  Power:       ~{self.ENERGY_PER_NEURON_UW} µW per neuron\n"
            f"  Access:      ebrains.eu (Human Brain Project)"
        )

    def __repr__(self) -> str:
        return f"BrainScalesBackend(sdk={'available' if self._sdk_available else 'not installed'})"
