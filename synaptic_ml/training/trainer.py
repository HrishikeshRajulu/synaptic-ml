"""
Trainer — manages the training loop for SpikingNets.

Supports two learning rules:
1. surrogate — backprop with surrogate gradients (fast, accurate, recommended)
2. stdp — unsupervised Hebbian learning (no labels needed, biologically inspired)
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
        """
        Train the model.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, n_features), values in [0, 1]
        y_train : np.ndarray, shape (N,), integer class labels
        epochs : int
        batch_size : int
        validation_split : float
        encoder : Encoder, optional. Default: RateEncoder.
        verbose : bool
        """
        from ..encoding.rate import RateEncoder

        if encoder is None:
            encoder = RateEncoder(time_steps=self.model.time_steps)

        # Split validation
        n = len(X_train)
        n_val = int(n * validation_split)
        if n_val > 0:
            X_val, y_val = X_train[:n_val], y_train[:n_val]
            X_train, y_train = X_train[n_val:], y_train[n_val:]

        n_train = len(X_train)
        n_classes = len(np.unique(y_train))

        if verbose:
            print(f"\n[synaptic_ml] Training {self.model}")
            print(f"  Samples: {n_train} train" + (f", {n_val} val" if n_val > 0 else ""))
            print(f"  Rule: {self.learning_rule}, LR: {self.learning_rate}")
            print(f"  Epochs: {epochs}, Batch: {batch_size}")
            print()

        for epoch in range(epochs):
            # Shuffle
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

                batch_loss, batch_correct = self._train_batch(X_b, y_b, encoder, n_classes)
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
        """Train one batch, return (total_loss, n_correct)."""
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

    def _surrogate_step(self, encoded: np.ndarray, y_true: int, n_classes: int):
        """
        One surrogate gradient training step.

        Uses BPTT (Backpropagation Through Time) with surrogate gradients
        for the spike nonlinearity.
        """
        model = self.model
        layers = model.layers

        # Forward pass — collect all activations
        self.model._reset_all()
        all_spikes = []  # list of [time_steps, layer] arrays

        for t in range(model.time_steps):
            x = encoded[t] if t < len(encoded) else np.zeros(model.layer_sizes[0])
            step_spikes = [x]
            spikes = x
            for layer in layers:
                spikes = layer.forward(spikes, model.dt)
                step_spikes.append(spikes.copy())
            all_spikes.append(step_spikes)

        # Decode output
        output_scores = layers[-1].decode()
        pred = int(np.argmax(output_scores))
        correct = (pred == y_true)

        # Compute cross-entropy loss on rate-coded output
        # Target: one-hot
        target = np.zeros(n_classes, dtype=np.float32)
        target[y_true] = 1.0

        # Softmax
        scores = output_scores - output_scores.max()
        exp_s = np.exp(scores)
        probs = exp_s / (exp_s.sum() + 1e-8)
        loss = -np.log(probs[y_true] + 1e-8)

        # Backprop through output layer
        d_out = probs - target  # gradient of CE+softmax

        # Update output layer weights (simplified BPTT)
        output_layer = layers[-1]
        if hasattr(output_layer, "synapse"):
            # Average input spikes to output layer over time
            avg_pre = np.mean([all_spikes[t][-2] for t in range(model.time_steps)], axis=0)
            dw = -self.learning_rate * np.outer(avg_pre, d_out)
            output_layer.synapse.update_weights(dw)

        # Backprop through hidden layers
        d_hidden = output_layer.weights @ d_out
        for i in range(len(layers) - 2, 0, -1):
            layer = layers[i]
            if not hasattr(layer, "synapse"):
                continue

            # Surrogate gradient through spike nonlinearity
            if hasattr(layer, "voltage_history") and len(layer.voltage_history) > 0:
                avg_v = np.mean(layer.voltage_history, axis=0)
                thresh = getattr(layer.neurons, "v_thresh", -50.0)
                surrogate_grad = self.surrogate_fn.backward(avg_v, thresh)
            else:
                surrogate_grad = np.ones(layer.n_neurons, dtype=np.float32)

            d_layer = d_hidden * surrogate_grad

            avg_pre = np.mean([all_spikes[t][i - 1] for t in range(model.time_steps)], axis=0)
            dw = -self.learning_rate * np.outer(avg_pre, d_layer)
            layer.synapse.update_weights(dw)

            d_hidden = layer.weights @ d_layer

        return float(loss), correct

    def _stdp_step(self, encoded: np.ndarray, y_true: int, n_classes: int):
        """
        One STDP training step (unsupervised + class-specific modulation).
        """
        model = self.model
        layers = model.layers

        model._reset_all()

        # Initialize STDP traces for each connection
        traces = []
        for i in range(1, len(layers)):
            if hasattr(layers[i], "synapse"):
                pre_t, post_t = self.stdp.init_traces(
                    layers[i].synapse.n_pre, layers[i].n_neurons
                )
                traces.append((pre_t, post_t))
            else:
                traces.append(None)

        for t in range(model.time_steps):
            x = encoded[t] if t < len(encoded) else np.zeros(model.layer_sizes[0])
            spikes = [x]
            current = x
            for layer in layers:
                current = layer.forward(current, model.dt)
                spikes.append(current.copy())

            # Apply STDP updates
            trace_idx = 0
            for i in range(1, len(layers)):
                layer = layers[i]
                if not hasattr(layer, "synapse") or traces[i - 1] is None:
                    continue
                pre_t, post_t = traces[i - 1]
                delta_w, pre_t, post_t = self.stdp.update(
                    spikes[i - 1], spikes[i], layer.synapse.weights,
                    pre_t, post_t, model.dt
                )
                layer.synapse.update_weights(delta_w)
                traces[i - 1] = (pre_t, post_t)

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
        """
        Evaluate on test data.

        Returns dict with 'accuracy', 'n_correct', 'n_total'.
        """
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
