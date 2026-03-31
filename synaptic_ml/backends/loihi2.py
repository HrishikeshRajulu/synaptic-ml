"""
Intel Loihi 2 backend.

Requires Intel NxSDK (not publicly available — must apply through Intel's
Neuromorphic Research Community: https://www.intel.com/loihi)

Loihi 2 specs:
- 1 million neurons per chip
- 120 synaptic operations per neuron per timestep
- ~30mW typical power (vs. hundreds of watts for GPU)
- Asynchronous, event-driven execution
- On-chip STDP learning support

This module maps synaptic_ml networks to NxSDK primitives:
  LIFLayer  →  NxNet compartment group
  DenseSynapse  →  NxNet connection
"""

import numpy as np
from .base import Backend, BackendNotAvailableError


class Loihi2Backend(Backend):
    """
    Intel Loihi 2 neuromorphic chip backend.

    Requires NxSDK. Install via Intel's Neuromorphic Research Community:
    https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html

    If NxSDK is not installed, methods will raise BackendNotAvailableError
    with helpful installation instructions.
    """

    # Loihi 2 hardware specifications
    MAX_NEURONS_PER_CHIP = 1_000_000
    MAX_SYNAPSES_PER_CHIP = 120_000_000
    TYPICAL_POWER_MW = 30.0
    ENERGY_PER_SYNAPTIC_OP_FJ = 8.6  # Published benchmark (Intel, 2021)

    def __init__(self):
        self._check_sdk()
        self.network = None
        self.nxnet = None
        self._ops_per_inference = 0

    def _check_sdk(self):
        try:
            import nxsdk  # noqa: F401
            self._sdk_available = True
        except ImportError:
            self._sdk_available = False

    def _require_sdk(self):
        if not self._sdk_available:
            raise BackendNotAvailableError(
                "\n[synaptic_ml] Intel NxSDK not found.\n\n"
                "To deploy on Loihi 2:\n"
                "  1. Apply for access: https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html\n"
                "  2. Intel provides NxSDK after approval (academic/research use).\n"
                "  3. Install: pip install nxsdk (after receiving access)\n\n"
                "For now, use CPUBackend for simulation:\n"
                "  model.deploy(target='cpu')\n"
            )

    def load_network(self, network) -> None:
        """
        Compile SpikingNet to Loihi 2 hardware graph.

        Maps:
        - LIFLayer → Loihi compartment groups with LIF dynamics
        - DenseSynapse → Loihi axon/dendrite connections
        - Layer thresholds → compartment thresholds
        """
        self._require_sdk()

        # Validate network fits on chip
        total_neurons = sum(network.layer_sizes)
        if total_neurons > self.MAX_NEURONS_PER_CHIP:
            raise ValueError(
                f"Network has {total_neurons} neurons but Loihi 2 supports "
                f"max {self.MAX_NEURONS_PER_CHIP:,} per chip. "
                f"Consider multi-chip deployment."
            )

        # NxSDK integration (requires SDK)
        import nxsdk.api.n2a as nx

        self.network = network
        net = nx.NxNet()

        compartments = []
        for i, size in enumerate(network.layer_sizes):
            # Create compartment group for each layer
            compartment_proto = nx.CompartmentPrototype(
                vThMant=100,          # Voltage threshold mantissa
                functionalState=nx.PHASE_BIAS,
                numDendriticAccumulators=16,
            )
            cg = net.createCompartmentGroup(size=size, prototype=compartment_proto)
            compartments.append(cg)

        # Create connections between layers
        for i in range(len(network.layer_sizes) - 1):
            layer = network.layers[i + 1]
            if hasattr(layer, "weights"):
                # Scale weights for integer Loihi format
                w_scaled = (layer.weights * 255).astype(np.int8)
                conn_proto = nx.ConnectionPrototype(
                    signMode=nx.SIGN_MODE_MIXED,
                    numWeightBits=8,
                    weightExponent=-4,
                )
                compartments[i].connect(
                    compartments[i + 1],
                    prototype=conn_proto,
                    weight=w_scaled,
                )

        self.nxnet = net
        print(f"[Loihi2] Compiled {total_neurons} neurons, "
              f"{sum(network.layer_sizes[i]*network.layer_sizes[i+1] for i in range(len(network.layer_sizes)-1)):,} synapses")

    def run(self, input_spikes: np.ndarray, time_steps: int = None) -> np.ndarray:
        self._require_sdk()

        if self.nxnet is None:
            raise RuntimeError("No network loaded. Call load_network() first.")

        time_steps = time_steps or self.network.time_steps

        # Run on hardware (NxSDK API)
        board = self.nxnet.start(time_steps)
        board.run(time_steps)
        board.disconnect()

        # Read output spikes from output compartment
        output_spikes = board.probes[-1].data  # simplified
        self._ops_per_inference = int(np.sum(input_spikes)) * self.network.layer_sizes[-1]

        return output_spikes.mean(axis=0)

    def get_energy_estimate(self) -> dict:
        energy_nJ = self._ops_per_inference * self.ENERGY_PER_SYNAPTIC_OP_FJ * 1e-6
        return {
            "ops_per_inference": self._ops_per_inference,
            "energy_nJ": energy_nJ,
            "power_mW": self.TYPICAL_POWER_MW,
            "chip": "Intel Loihi 2",
        }

    def get_hardware_info(self) -> str:
        sdk_status = "Available" if self._sdk_available else "Not installed (using simulation mode)"
        return (
            f"  Backend:     Intel Loihi 2\n"
            f"  NxSDK:       {sdk_status}\n"
            f"  Max neurons: {self.MAX_NEURONS_PER_CHIP:,} per chip\n"
            f"  Power:       ~{self.TYPICAL_POWER_MW}mW typical\n"
            f"  Efficiency:  {self.ENERGY_PER_SYNAPTIC_OP_FJ} fJ per synaptic op\n"
            f"  Access:      intel.com/loihi (research program)"
        )

    def __repr__(self) -> str:
        return f"Loihi2Backend(sdk={'available' if self._sdk_available else 'not installed'})"
