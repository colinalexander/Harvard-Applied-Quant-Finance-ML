from __future__ import annotations

from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def set_seed(seed: int) -> None:
    """Set NumPy global RNG for reproducibility in callers."""

    np.random.seed(seed)


def softmax(x: np.ndarray) -> np.ndarray:
    """Computes a numerically stable softmax over the last dimension.

    Args:
        x: Input array of shape (batch_size, num_classes).

    Returns:
        Array of probabilities with the same shape as x. Each row sums to 1.
    """
    x_stable = x - x.max(axis=1, keepdims=True)
    scores = np.exp(x_stable)
    row_sums = scores.sum(axis=1, keepdims=True)
    return scores / row_sums


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Computes the sigmoid activation function elementwise.

    Args:
        x: Input array.

    Returns:
        Array with the same shape as x, with sigmoid applied elementwise.
    """
    clip_value = 500.0
    x_clipped = np.clip(x, a_min=-clip_value, a_max=clip_value)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def softmax_cross_entropy(
    logits: np.ndarray,
    one_hot_labels: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Computes softmax probabilities and mean cross-entropy loss.

    This mirrors the PS4 MNIST implementation:
    - applies a numerically stable softmax
    - computes cross-entropy against one-hot labels

    Args:
        logits: Unnormalized scores of shape (batch_size, num_classes).
        one_hot_labels: One-hot encoded targets of shape
            (batch_size, num_classes).
        eps: Small constant added for numerical stability in log.

    Returns:
        Tuple consisting of:
            probs: Softmax probabilities of shape (batch_size, num_classes).
            loss: Mean cross-entropy loss over the batch.
    """
    probs = softmax(logits)
    log_probs = np.log(probs + eps)
    loss = -np.sum(one_hot_labels * log_probs) / logits.shape[0]
    return probs, float(loss)


@dataclass
class LayerCache:
    A_prev: np.ndarray
    Z: np.ndarray | None = None  # None for output layer.
    mask: np.ndarray | None = None  # Dropout mask (only for hidden layers).


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float


@dataclass
class NNParams:
    # Architecture.
    input_dim: int
    output_dim: int = 1
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.1

    # Optimization.
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256

    # Training schedule.
    max_epochs: int = 500
    patience: int = 10
    early_stopping_tolerance: float = 1e-6

    # Reproducibility.
    seed: int = 0


class NumpyNN:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        output_dim: int = 1,
        seed: int = 0,
    ):
        # Validate arguments.
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")

        self.dropout = dropout
        self.survival_prob = 1.0 - dropout

        # Normalize for consistency.
        hidden_dims = tuple(hidden_dims)

        # Architecture definition for introspection.
        dims = [input_dim, *hidden_dims, output_dim]
        layer_shapes = list(zip(dims[:-1], dims[1:], strict=False))
        self._layer_shapes = tuple(layer_shapes)
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dims = hidden_dims

        rng = np.random.default_rng(seed)
        dtype = np.float32

        self.weights = [
            rng.normal(0.0, scale=np.sqrt(2.0 / m), size=(m, n)).astype(dtype)
            for m, n in layer_shapes
        ]

        self.biases = [np.zeros(n, dtype=dtype) for _, n in layer_shapes]

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        return self._hidden_dims

    @property
    def layer_shapes(self) -> tuple[tuple[int, int], ...]:
        return self._layer_shapes

    def forward(
        self,
        X: np.ndarray,
        training: bool = False,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, list[LayerCache]]:
        """Runs a forward pass through the network.

        Args:
            X: Input feature matrix of shape (batch_size, input_dim).
            training: Whether to apply dropout. If True, dropout masks are sampled.
            rng: Random number generator used for dropout. Must be provided when
                training with positive dropout.

        Returns:
            Tuple consisting of:
                outputs: Network predictions. Shape is (batch_size,) when the
                    output dimension is 1, otherwise (batch_size, output_dim).
                caches: List of LayerCache objects containing intermediate values
                    needed for backpropagation (A_prev, Z, mask).
        """
        if training and self.dropout > 0.0 and rng is None:
            raise ValueError("rng must be provided when training with dropout.")

        # Ensure dtype matches weights
        A = X.astype(np.float32, copy=False)

        caches: list[LayerCache] = []
        num_layers = len(self.weights)

        # Hidden layers: Linear -> ReLU -> (optional dropout)
        for idx in range(num_layers - 1):
            W = self.weights[idx]
            b = self.biases[idx]

            Z = A @ W + b  # (batch, hidden_dim)
            A_next = relu(Z)

            mask = None
            if training and self.dropout > 0.0:
                mask = rng.random(A_next.shape) < self.survival_prob
                A_next = A_next * mask / self.survival_prob

            caches.append(LayerCache(A_prev=A, Z=Z, mask=mask))
            A = A_next

        # Output layer (linear only)
        W_out = self.weights[-1]
        b_out = self.biases[-1]
        Z_out = A @ W_out + b_out  # (batch, output_dim)

        caches.append(LayerCache(A_prev=A))  # no Z, no mask

        # If scalar output, return shape (batch,) to match training loop
        if Z_out.shape[1] == 1:
            outputs = Z_out[:, 0]
        else:
            outputs = Z_out

        return outputs, caches

    def backward(
        self,
        caches: list[LayerCache],
        dLoss: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Backpropagates the loss gradient through the network.

        Args:
            caches: List of LayerCache objects produced by forward().
            dLoss: Gradient of the loss with respect to the network outputs.
                Shape is (batch_size,) when output_dim == 1, otherwise
                (batch_size, output_dim).

        Returns:
            Tuple consisting of:
                grads_w: List of gradients for each weight matrix
                    (same shapes as self.weights).
                grads_b: List of gradients for each bias vector
                    (same shapes as self.biases).
        """
        # Ensure dLoss is 2D: (batch_size, output_dim)
        dA = dLoss[:, None] if dLoss.ndim == 1 else dLoss

        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # ----- Output layer -----
        last_cache = caches[-1]
        A_prev = last_cache.A_prev

        grads_w[-1] = A_prev.T @ dA
        grads_b[-1] = np.sum(dA, axis=0)
        dA_prev = dA @ self.weights[-1].T

        # ----- Hidden layers -----
        num_layers = len(self.weights)

        for layer_idx in reversed(range(num_layers - 1)):
            cache = caches[layer_idx]
            Z = cache.Z
            A_prev = cache.A_prev
            mask = cache.mask

            dZ = dA_prev * (Z > 0)

            if mask is not None:
                dZ = dZ * mask / self.survival_prob

            grads_w[layer_idx] = A_prev.T @ dZ
            grads_b[layer_idx] = np.sum(dZ, axis=0)

            dA_prev = dZ @ self.weights[layer_idx].T

        return grads_w, grads_b

    def apply_gradients(
        self,
        grads_w: list[np.ndarray],
        grads_b: list[np.ndarray],
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        """Applies SGD updates to weights and biases.

        Args:
            grads_w: Gradients of each weight matrix.
            grads_b: Gradients of each bias vector.
            learning_rate: Step size for the update.
            weight_decay: L2 regularization coefficient.
        """
        # Precompute update terms using comprehensions (readable + compact).
        weight_updates = [
            learning_rate * (grad_w + weight_decay * w)
            for grad_w, w in zip(grads_w, self.weights, strict=False)
        ]

        bias_updates = [learning_rate * grad_b for grad_b in grads_b]

        # Apply updates in-place.
        for idx in range(len(self.weights)):
            self.weights[idx] -= weight_updates[idx]
            self.biases[idx] -= bias_updates[idx]

    def get_state(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Returns a copy of the current network parameters."""
        return [w.copy() for w in self.weights], [b.copy() for b in self.biases]

    def load_state(self, state: tuple[list[np.ndarray], list[np.ndarray]]) -> None:
        """Loads network parameters from a saved state."""
        weights, biases = state
        self.weights = [w.copy() for w in weights]
        self.biases = [b.copy() for b in biases]

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Returns softmax probabilities for classification.

        Assumes the network's output layer produces logits of shape
        (batch_size, num_classes).
        """
        logits, _ = self.forward(X, training=False)
        if logits.ndim == 1:
            raise ValueError(
                "predict_proba expects multi-dimensional logits; "
                "got shape (batch_size,)."
            )
        return softmax(logits)

    def predict_classes(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Returns predicted class indices for classification."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def _iterate_minibatches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Generator[tuple[np.ndarray, np.ndarray]]:
    """Generates shuffled mini-batches from the dataset.

    Args:
        X: Feature matrix of shape (num_samples, num_features).
        y: Target array of shape (num_samples,) or (num_samples, output_dim).
        batch_size: Number of samples per mini-batch.
        rng: Random number generator used to shuffle sample indices.

    Yields:
        Tuples (X_batch, y_batch), each containing a mini-batch drawn
        from the shuffled dataset.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")

    n = len(X)
    idxs = rng.permutation(n)

    for start in range(0, n, batch_size):
        batch_idx = idxs[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: NNParams,
) -> tuple[NumpyNN, list[EpochRecord]]:
    """Trains a NumpyNN model using mini-batch SGD with early stopping.

    Args:
        X_train: Training features of shape (n_train, input_dim).
        y_train: Training targets of shape (n_train,) or (n_train, output_dim).
        X_val: Validation features of shape (n_val, input_dim).
        y_val: Validation targets of shape (n_val,) or (n_val, output_dim).
        params: Hyperparameters and architecture configuration.

    Returns:
        Tuple consisting of:
            model: Trained NumpyNN model restored to the best validation epoch.
            history: List of EpochRecord objects, one per epoch.
    """
    # Master RNG for shuffling and dropout.
    rng = np.random.default_rng(params.seed)

    # Initialize model.
    model = NumpyNN(
        input_dim=params.input_dim,
        hidden_dims=params.hidden_dims,
        dropout=params.dropout,
        output_dim=params.output_dim,
        seed=params.seed,
    )

    history: list[EpochRecord] = []
    best_state = model.get_state()
    best_val = float("inf")
    epochs_no_improve = 0
    early_stopping_tolerance = params.early_stopping_tolerance

    def mse(pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((pred - target) ** 2))

    for epoch in range(1, params.max_epochs + 1):
        train_losses: list[float] = []

        # Training loop over mini-batches.
        for Xb, yb in _iterate_minibatches(X_train, y_train, params.batch_size, rng):
            preds, caches = model.forward(Xb, training=True, rng=rng)
            diff = preds - yb  # (batch_size,)

            loss = mse(preds, yb)
            train_losses.append(loss)

            # Gradient of MSE: L = mean((pred - y)^2)
            dLoss = (2.0 / len(Xb)) * diff  # (batch_size,)
            grads_w, grads_b = model.backward(caches, dLoss)
            model.apply_gradients(
                grads_w, grads_b, params.learning_rate, params.weight_decay
            )

        # Validation loss (no dropout).
        val_preds, _ = model.forward(X_val, training=False)
        val_loss = mse(val_preds, y_val)

        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=float(np.mean(train_losses)),
                val_loss=val_loss,
            )
        )

        # Early stopping based on validation loss.
        if val_loss < best_val - early_stopping_tolerance:
            best_val = val_loss
            best_state = model.get_state()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= params.patience:
                break

    # Restore best model.
    model.load_state(best_state)
    return model, history


def predict(model: NumpyNN, X: np.ndarray) -> np.ndarray:
    """Runs the model in inference mode and returns predictions.

    Args:
        model: Trained NumpyNN model.
        X: Input feature matrix of shape (num_samples, input_dim).

    Returns:
        Array of predictions. Shape is (num_samples,) when output_dim == 1,
        otherwise (num_samples, output_dim).
    """
    return model.forward(X, training=False)[0]


def classification_accuracy(
    probs: np.ndarray,
    one_hot_labels: np.ndarray,
) -> float:
    """Computes classification accuracy from softmax probabilities.

    Args:
        probs: Predicted class probabilities of shape
            (num_samples, num_classes).
        one_hot_labels: One-hot encoded true labels of the same shape.

    Returns:
        Accuracy as a float in [0, 1].
    """
    pred_classes = np.argmax(probs, axis=1)
    true_classes = np.argmax(one_hot_labels, axis=1)
    return float((pred_classes == true_classes).mean())


# --------------------------------------------------------------------------- #
# Capacity sweeps and rolling evaluation
# --------------------------------------------------------------------------- #

architectures: dict[str, tuple[int, ...]] = {
    "small": (32, 16),
    "medium": (64, 32),
    "large": (128, 64, 32),
}


@dataclass
class Fold:
    """Container for a single rolling-window split."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def count_parameters(model: NumpyNN) -> int:
    """Total number of trainable parameters in the network."""

    n_weights = sum(w.size for w in model.weights)
    n_biases = sum(b.size for b in model.biases)
    return int(n_weights + n_biases)


def compute_spearman_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman rank IC with NaN safety."""

    if len(y_pred) < 2 or len(y_true) < 2:
        return np.nan
    corr, _ = stats.spearmanr(y_pred, y_true, nan_policy="omit")
    return float(corr)


def r2_oos(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Out-of-sample R^2."""

    if len(y_true) == 0:
        return np.nan
    residual = np.sum((y_true - y_pred) ** 2)
    baseline = np.sum((y_true - np.mean(y_true)) ** 2)
    if baseline == 0:
        return np.nan
    return 1.0 - float(residual / baseline)


def newey_west_tstat(series: np.ndarray, lag: int = 6) -> float:
    """
    Compute the Newey–West HAC t-statistic for a 1-D array of IC values.
    """

    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 3:
        return np.nan

    lag = min(lag, T - 1)
    x_centered = x - x.mean()

    gamma0 = np.mean(x_centered * x_centered)
    hac = gamma0
    for lag_idx in range(1, lag + 1):
        weight = 1.0 - lag_idx / (lag + 1)
        cov = np.mean(x_centered[lag_idx:] * x_centered[:-lag_idx])
        hac += 2.0 * weight * cov

    if hac <= 0:
        return np.nan

    se = np.sqrt(hac / T)
    if se == 0.0:
        return np.nan
    return float(x.mean() / se)


def run_nn_capacity_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    base_params: NNParams,
) -> dict[str, dict[str, object]]:
    """Train small/medium/large MLPs for a single fold."""

    results: dict[str, dict[str, object]] = {}
    for name, dims in architectures.items():
        params = replace(base_params, hidden_dims=dims)
        model, history = train_model(X_train, y_train, X_val, y_val, params)
        results[name] = {"model": model, "history": history}
    return results


def evaluate_nn_capacity_across_folds(
    folds: Iterable[Fold],
    base_params: NNParams,
) -> pd.DataFrame:
    """Run the capacity sweep over rolling folds and aggregate metrics."""

    records: list[dict[str, object]] = []

    for name, dims in architectures.items():
        all_ics: list[float] = []
        all_preds: list[np.ndarray] = []
        all_true: list[np.ndarray] = []
        param_count = None

        for fold in folds:
            params = replace(base_params, hidden_dims=dims)
            model, _ = train_model(
                fold.X_train, fold.y_train, fold.X_val, fold.y_val, params
            )
            if param_count is None:
                param_count = count_parameters(model)

            y_hat = predict(model, fold.X_test)
            ic = compute_spearman_ic(y_hat, fold.y_test)

            all_ics.append(ic)
            all_preds.append(np.asarray(y_hat).ravel())
            all_true.append(np.asarray(fold.y_test).ravel())

        y_true_all = np.concatenate(all_true) if all_true else np.array([])
        y_pred_all = np.concatenate(all_preds) if all_preds else np.array([])

        records.append(
            {
                "architecture": name,
                "hidden_dims": dims,
                "num_params": param_count,
                "mean_ic": float(np.nanmean(all_ics)) if all_ics else np.nan,
                "nw_tstat": newey_west_tstat(np.array(all_ics, dtype=float)),
                "r2_oos": r2_oos(y_true_all, y_pred_all),
            }
        )

    return pd.DataFrame.from_records(records)
