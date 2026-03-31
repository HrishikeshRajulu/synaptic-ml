"""
Intel Loihi 2 backend — built on top of Intel's Lava framework.

Lava is Intel's official open-source framework for neuromorphic computing.
Install it with: pip install lava-nc

Think of it this way:
  Lava = the engine (low-level, complex, powerful)
  synaptic_ml = the steering wheel (simple API on top of Lava)

Loihi 2 specs:
- 1 million neurons per chip
- ~30mW typical power (vs hundreds of watts for GPU)
- Asynchronous, event-driven execution
- On-chip STDP learning support
- 8.6 fJ per synaptic operation

Lava GitHub: https://github.com/lava-nc/lava
Lava docs:   https://lava-nc.org
"""

import numpy as np
from .base import Backend, BackendNotAvailableError


class Loihi2Backend(Backend):
    """
    Intel Loihi 2 neuromorphic chip backend.

    Uses Intel's Lava framework (open source) as the underlying engine.
    Install Lava with: pip install lava-nc

    For actual Loihi 2 hardware execution, join the Intel Neuromorphic
    Research Community (INRC): http://neuromorphic.intel.com

    Without hardware, Lava runs on CPU simulation — same results,
    no chip required for development.
    """

    MAX_NEURONS_PER_CHIP = 1_000_000
    MAX_SYNAPSES_PER_CHIP = 120_000_000
    TYPICAL_POWER_MW = 30.0
    ENERGY_PER_SYNAPTIC_OP_FJ = 8.6  # Intel published benchmark (2021)

    def __init__(self):
        self._check_lava()
        self.network = None
        self._lava_network = None
        self._ops_per_inference = 0

    def _check_lava(self):
        try:
            import lava.lib.dl.slayer as slayer  # noqa: F401
            self._lava_available = True
            self._lava_mode = "slayer"
        except ImportError:
            try:
                from lava.proc.lif.process import LIF  # noqa: F401
                self._lava_available = True
                self._lava_mode = "proc"
            except ImportError:
                self._lava_available = False
                self._lava_mode = None

    def _require_lava(self):
        if not self._lava_available:
            raise BackendNotAvailableError(
                "\n[synaptic_ml] Lava framework not found.\n\n"
                "Lava is Intel's open-source neuromorphic framework.\n"
                "Install it with:\n"
                "  pip install lava-nc\n\n"
                "For Loihi 2 hardware access, join the INRC:\n"
                "  http://neuromorphic.intel.com\n"
                "  Email: inrc_interest@intel.com\n\n"
                "Without hardware, Lava simulates on CPU.\n"
                "For now, use CPUBackend:\n"
                "  model.deploy(target='cpu')\n"
            )

    def load_network(self, network) -> None:
        """
        Compile SpikingNet to a Lava process network.

        Maps synaptic_ml layers to Lava processes:
          LIFLayer     -> lava.proc.lif.process.LIF
          DenseSynapse -> lava.proc.dense.process.Dense
          InputLayer   -> lava.proc.io.source.RingBuffer
        """
        self._require_lava()

        total_neurons = sum(network.layer_sizes)
        if total_neurons > self.MAX_NEURONS_PER_CHIP:
            raise ValueError(
                f"Network has {total_neurons} neurons but Loihi 2 supports "
                f"max {self.MAX_NEURONS_PER_CHIP:,} per chip."
            )

        from lava.proc.lif.process import LIF
        from lava.proc.dense.process import Dense

        self.network = network
        lava_layers = []
        lava_connections = []

        # Build Lava LIF processes for each hidden + output layer
        for i, size in enumerate(network.layer_sizes[1:], 1):
            lif = LIF(
                shape=(size,),
                du=0,           # no current decay (matches our LIF)
                dv=0,           # membrane decay handled by tau_m
                vth=1,          # normalized threshold
                bias_mant=0,
                bias_exp=0,
            )
            lava_layers.append(lif)

        # Connect layers with Dense synapses
        for i in range(len(lava_layers) - 1):
            layer = network.layers[i + 1]
            if hasattr(layer, "weights"):
                # Lava Dense expects weights in [post, pre] format
                w = layer.weights.T
                dense = Dense(weights=w)
                # Connect: pre.s_out -> dense.s_in -> post.a_in
                lava_layers[i].s_out.connect(dense.s_in)
                dense.a_out.connect(lava_layers[i + 1].a_in)
                lava_connections.append(dense)

        self._lava_network = {
            "layers": lava_layers,
            "connections": lava_connections,
        }

        print(f"[Loihi2/Lava] Compiled {total_neurons} neurons across "
              f"{len(lava_layers)} layers using Lava framework")
        print(f"[Loihi2/Lava] Mode: {'Hardware' if self._lava_mode == 'hw' else 'CPU simulation'}")

    def run(self, input_spikes: np.ndarray, time_steps: int = None) -> np.ndarray:
        """
        Run inference using Lava's execution runtime.

        On CPU: simulates Loihi 2 dynamics in software.
        On hardware: executes directly on Loihi 2 chip.
        """
        self._require_lava()

        if self._lava_network is None:
            raise RuntimeError("No network loaded. Call load_network() first.")

        from lava.proc.io.source import RingBuffer
        from lava.proc.io.sink import RingBuffer as SinkBuffer
        from lava.magma.core.run_configs import Loihi2SimCfg
        from lava.magma.core.run_conditions import RunSteps

        time_steps = time_steps or self.network.time_steps

        # Input source: inject spike trains
        # input_spikes shape: (time_steps, n_input)
        inp_data = input_spikes.T.astype(np.int32)  # Lava wants (n, T)
        source = RingBuffer(data=inp_data)

        # Output sink: collect output spikes
        n_out = self.network.layer_sizes[-1]
        sink = SinkBuffer(shape=(n_out,), buffer=time_steps)

        layers = self._lava_network["layers"]

        # Connect source to first layer, last layer to sink
        from lava.proc.dense.process import Dense
        w_in = self.network.layers[1].weights.T
        in_dense = Dense(weights=w_in)
        source.s_out.connect(in_dense.s_in)
        in_dense.a_out.connect(layers[0].a_in)
        layers[-1].s_out.connect(sink.a_in)

        # Run simulation
        run_cfg = Loihi2SimCfg()
        layers[0].run(condition=RunSteps(num_steps=time_steps), run_cfg=run_cfg)

        # Read output
        out_spikes = sink.data.get()  # shape: (n_out, time_steps)
        layers[0].stop()

        self._ops_per_inference = int(np.sum(input_spikes)) * self.network.layer_sizes[-1]

        # Return mean firing rate per output neuron
        return out_spikes.mean(axis=1).astype(np.float32)

    def get_energy_estimate(self) -> dict:
        energy_nJ = self._ops_per_inference * self.ENERGY_PER_SYNAPTIC_OP_FJ * 1e-6
        return {
            "ops_per_inference": self._ops_per_inference,
            "energy_nJ": energy_nJ,
            "power_mW": self.TYPICAL_POWER_MW,
            "chip": "Intel Loihi 2 (via Lava)",
        }

    def get_hardware_info(self) -> str:
        if self._lava_available:
            lava_status = f"Available (mode: {self._lava_mode})"
        else:
            lava_status = "Not installed — run: pip install lava-nc"

        return (
            f"  Backend:     Intel Loihi 2 (via Lava framework)\n"
            f"  Lava:        {lava_status}\n"
            f"  Lava docs:   https://lava-nc.org\n"
            f"  Max neurons: {self.MAX_NEURONS_PER_CHIP:,} per chip\n"
            f"  Power:       ~{self.TYPICAL_POWER_MW}mW typical\n"
            f"  Efficiency:  {self.ENERGY_PER_SYNAPTIC_OP_FJ} fJ per synaptic op\n"
            f"  Hardware:    Join INRC at neuromorphic.intel.com"
        )

    def __repr__(self) -> str:
        status = "available" if self._lava_available else "not installed (pip install lava-nc)"
        return f"Loihi2Backend(lava={status})"
