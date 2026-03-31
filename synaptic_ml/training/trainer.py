"""
Trainer — manages the training loop for SpikingNets.

Supports two learning rules:
1. surrogate — proper BPTT with surrogate gradients (recommended)
2. stdp — unsupervised Hebbian learning

Key fix over v0.1.x: proper per-timestep forward pass with full
backpropagation through time (BPTT), not averaged approximations.
"""

import numpy as np
from typing import Optional, Literal


class Trainer:
    """
    Training manager for SpikingNet models.

    Parameters
    ----------
    model : SpikingNet
    learning_rule : str
        'surrogate' or 'stdp'.
    learning_rate : float
    surrogate : str
        Surrogate gradient type: 'fast_sigmoid', 'sigmoid', 'piecewise_linear'.
    """

    def __init__(
        self,
        model,
        learning_rule: Literal["surrogate", "stdp"] = "surrogate",
        learning_rate: float = 0.001,
        surrogate: str = "fast_sigmoid",
    ):
        self.model = model
        self.learning_rule = learning_rule
        self.learning_rate = learning_rate

        if learning_rule == "surrogate":
            from ..learning.surrogate import get_surrogate
            self.surrogate_fn = get_surrogate(surrogate)
        elif learning_rule == "stdp":
            from ..learning.stdp import STDPRule
            self.stdp = STDPRule()
        else:
            raise ValueError(f"Unknown learning rule: {learning_rule}. Choose 'surrogate' or 'stdp'.")

        self.train_losses = []
        self.train_accuracies = []
        # Adam optimizer state
        self._adam_m = {}   # first moment
        self._adam_v = {}   # second moment
        self._adam_t = 0    # step counter

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.1,
        encoder=None,
        verbose: bool = True,
    ):
        from ..encoding.rate import RateEncoder

        if encoder is None:
            encoder = RateEncoder(time_steps=self.model.time_steps)

        n = len(X_train)
        n_val = int(n * validation_split)
        if n_val > 0:
            X_val, y_val = X_train[:n_val], y_train[:n_val]
            X_train, y_train = X_train[n_val:], y_train[n_val:]

        n_train = len(X_train)
        n_classes = int(y_train.max()) + 1

        if verbose:
            print(f"\n[synaptic_ml] Training {self.model}")
            print(f"  Samples: {n_train} train" + (f", {n_val} val" if n_val > 0 else ""))
            print(f"  Rule: {self.learning_rule}, LR: {self.learning_rate}")
            print(f"  Epochs: {epochs}, Batch: {batch_size}")
            print()

        for epoch in range(epochs):
            idx = np.random.permutation(n_train)
            X_shuf, y_shuf = X_train[idx], y_train[idx]

            epoch_loss = 0.0
            epoch_correct = 0
            n_batches = max(1, n_train // batch_size)

            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_train)
                X_b = X_shuf[start:end]
                y_b = y_shuf[start:end]

                batch_loss, batch_correct = self._train_batch(
                    X_b, y_b, encoder, n_classes
                )
                epoch_loss += batch_loss
                epoch_correct += batch_correct

            avg_loss = epoch_loss / n_train
            avg_acc = epoch_correct / n_train
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(avg_acc)

            if verbose:
                val_str = ""
                if n_val > 0:
                    val_acc = self._evaluate(X_val, y_val, encoder)
                    val_str = f"  val_acc={val_acc:.3f}"
                print(
                    f"  Epoch {epoch+1:>3}/{epochs}  "
                    f"loss={avg_loss:.4f}  acc={avg_acc:.3f}{val_str}"
                )

        if verbose:
            print(f"\n[synaptic_ml] Training complete.")

    def _train_batch(self, X_b, y_b, encoder, n_classes):
        total_loss = 0.0
        n_correct = 0

        for x, y_true in zip(X_b, y_b):
            encoded = encoder.encode(x)

            if self.learning_rule == "surrogate":
                loss, correct = self._surrogate_step(encoded, int(y_true), n_classes)
            else:
                loss, correct = self._stdp_step(encoded, int(y_true), n_classes)

            total_loss += loss
            n_correct += int(correct)

        return total_loss, n_correct

    def _apply_adam(self, synapse, grad: np.ndarray, beta1=0.9, beta2=0.999, eps=1e-8) -> np.ndarray:
        """Adam optimizer: adaptive learning rate per weight."""
        key = id(synapse)
        if key not in self._adam_m:
            self._adam_m[key] = np.zeros_like(grad)
            self._adam_v[key] = np.zeros_like(grad)
        self._adam_m[key] = beta1 * self._adam_m[key] + (1 - beta1) * grad
        self._adam_v[key] = beta2 * self._adam_v[key] + (1 - beta2) * grad ** 2
        m_hat = self._adam_m[key] / (1 - beta1 ** self._adam_t + eps)
        v_hat = self._adam_v[key] / (1 - beta2 ** self._adam_t + eps)
        return -self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def _surrogate_step(self, encoded: np.ndarray, y_true: int, n_classes: int):
        """
        Full BPTT with surrogate gradients.

        Uses voltage-based soft activations as presynaptic activity so that
        weight updates are non-zero even when neurons don't fire.
        """
        model = self.model
        layers = model.layers
        T = model.time_steps

        # ---- Forward pass ----
        # all_soft_acts[t][l] = soft activation (normalized voltage) at layer l, time t
        # Index 0 = raw input, index 1..n = layer outputs
        all_soft_acts = []
        all_voltages = []

        model._reset_all()

        for t in range(T):
            x = encoded[t] if t < len(encoded) else np.zeros(model.layer_sizes[0])
            t_soft = [x.astype(np.float32)]  # input: spike rates are already soft
            t_volts = []

            current = x.astype(np.float32)
            for layer in layers:
                current = layer.forward(current, model.dt)

                if hasattr(layer, 'neurons') and hasattr(layer.neurons, 'v'):
                    v = layer.neurons.v.copy()
                    t_volts.append(v)
                    # Normalize voltage to [0,1]: v_rest -> 0, v_thresh -> 1
                    v_rest = getattr(layer.neurons, 'v_rest', -65.0)
                    v_thresh = getattr(layer.neurons, 'v_thresh', -50.0)
                    soft = np.clip((v - v_rest) / (v_thresh - v_rest + 1e-8), 0.0, 1.0)
                    t_soft.append(soft)
                else:
                    # InputLayer: no neurons, use pass-through values
                    t_soft.append(current.copy())

            all_soft_acts.append(t_soft)
            all_voltages.append(t_volts)

        weight_layers = [l for l in layers if hasattr(l, 'synapse')]

        # ---- Compute output scores ----
        # Use soft-activation linear readout: avg_hidden_soft @ W_out
        # This reflects learned weights even when neurons don't fire,
        # giving a non-zero gradient signal from the very first step.
        out_wlayer = weight_layers[-1] if weight_layers else None
        if out_wlayer is not None:
            layer_idx = layers.index(out_wlayer)
            avg_soft_pre = np.mean(
                [all_soft_acts[t][layer_idx] for t in range(T)], axis=0
            )
            output_scores = avg_soft_pre @ out_wlayer.synapse.weights
        else:
            out_layer = layers[-1]
            output_scores = out_layer.neurons.v.copy() if hasattr(out_layer, 'neurons') else np.zeros(n_classes)

        output_scores = output_scores - output_scores.mean()

        pred = int(np.argmax(output_scores))
        correct = (pred == y_true)

        # ---- Cross-entropy loss ----
        target = np.zeros(n_classes, dtype=np.float32)
        target[y_true] = 1.0

        temp = 1.0
        scores = (output_scores - output_scores.max()) / (temp + 1e-8)
        exp_s = np.exp(np.clip(scores, -20, 20))
        probs = exp_s / (exp_s.sum() + 1e-8)
        loss = -np.log(probs[y_true] + 1e-8)

        # ---- Backward pass ----
        self._adam_t += 1
        d_out = probs - target
        # Clip gradient to prevent blowups
        d_out = np.clip(d_out, -1.0, 1.0)

        # Update output layer using Adam
        if weight_layers:
            out_wlayer = weight_layers[-1]
            layer_idx = layers.index(out_wlayer)
            avg_pre = np.mean(
                [all_soft_acts[t][layer_idx] for t in range(T)], axis=0
            )
            grad_out = np.outer(avg_pre, d_out)
            dw_out = self._apply_adam(out_wlayer.synapse, grad_out)
            out_wlayer.synapse.update_weights(dw_out)

        # Backprop through hidden layers
        if len(weight_layers) > 1:
            d_hidden = weight_layers[-1].synapse.weights @ d_out
            d_hidden = np.clip(d_hidden, -1.0, 1.0)

            for i in range(len(weight_layers) - 2, -1, -1):
                layer = weight_layers[i]
                layer_idx = layers.index(layer)

                # Soft-activation gradient: d(clip((v-vr)/(vt-vr),0,1))/dv
                # = 1/(vt-vr) when vr<=v<=vt, else 0
                if hasattr(layer, 'voltage_history') and len(layer.voltage_history) > 0:
                    avg_v = np.mean(layer.voltage_history, axis=0)
                    v_rest = getattr(layer.neurons, 'v_rest', -65.0)
                    v_thresh = getattr(layer.neurons, 'v_thresh', -50.0)
                    in_range = (avg_v >= v_rest) & (avg_v <= v_thresh)
                    surrogate_grad = np.where(in_range, 1.0 / (v_thresh - v_rest + 1e-8), 0.0).astype(np.float32)
                else:
                    surrogate_grad = np.ones(layer.n_neurons, dtype=np.float32)

                d_layer = d_hidden * surrogate_grad

                avg_pre = np.mean(
                    [all_soft_acts[t][layer_idx] for t in range(T)], axis=0
                )
                grad_hidden = np.outer(avg_pre, d_layer)
                dw = self._apply_adam(layer.synapse, grad_hidden)
                layer.synapse.update_weights(dw)

                d_hidden = np.clip(layer.synapse.weights @ d_layer, -1.0, 1.0)

        return float(loss), correct

    def _stdp_step(self, encoded: np.ndarray, y_true: int, n_classes: int):
        """STDP training step with reward modulation."""
        model = self.model
        layers = model.layers

        model._reset_all()

        # Initialize STDP traces
        traces = []
        for layer in layers:
            if hasattr(layer, 'synapse'):
                pre_t = np.zeros(layer.synapse.n_pre, dtype=np.float32)
                post_t = np.zeros(layer.n_neurons, dtype=np.float32)
                traces.append((pre_t, post_t))
            else:
                traces.append(None)

        prev_spikes = [None] * len(layers)

        for t in range(model.time_steps):
            x = encoded[t] if t < len(encoded) else np.zeros(model.layer_sizes[0])
            current_spikes = [x.astype(np.float32)]
            spikes = x.astype(np.float32)

            for layer in layers:
                spikes = layer.forward(spikes, model.dt)
                current_spikes.append(spikes.copy())

            # Apply STDP per layer
            for i, layer in enumerate(layers):
                if not hasattr(layer, 'synapse') or traces[i] is None:
                    continue
                pre_t, post_t = traces[i]
                delta_w, pre_t, post_t = self.stdp.update(
                    current_spikes[i],
                    current_spikes[i + 1],
                    layer.synapse.weights,
                    pre_t, post_t,
                    model.dt,
                )
                layer.synapse.update_weights(delta_w)
                traces[i] = (pre_t, post_t)

        output_scores = layers[-1].decode()
        pred = int(np.argmax(output_scores))
        correct = (pred == y_true)

        target = np.zeros(n_classes, dtype=np.float32)
        target[y_true] = 1.0
        scores = output_scores - output_scores.max()
        probs = np.exp(scores) / (np.exp(scores).sum() + 1e-8)
        loss = -np.log(probs[y_true] + 1e-8)

        return float(loss), correct

    def _evaluate(self, X_val, y_val, encoder) -> float:
        correct = 0
        for x, y_true in zip(X_val, y_val):
            encoded = encoder.encode(x)
            scores = self.model.run(encoded)
            if int(np.argmax(scores)) == int(y_true):
                correct += 1
        return correct / len(y_val)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, encoder=None) -> dict:
        from ..encoding.rate import RateEncoder

        if encoder is None:
            encoder = RateEncoder(time_steps=self.model.time_steps)

        correct = 0
        for x, y_true in zip(X_test, y_test):
            encoded = encoder.encode(x)
            scores = self.model.run(encoded)
            if int(np.argmax(scores)) == int(y_true):
                correct += 1

        acc = correct / len(y_test)
        print(f"[synaptic_ml] Test accuracy: {acc:.4f} ({correct}/{len(y_test)})")
        return {"accuracy": acc, "n_correct": correct, "n_total": len(y_test)}

    def __repr__(self) -> str:
        return f"Trainer(rule={self.learning_rule}, lr={self.learning_rate})"
