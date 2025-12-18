"""Configuration constants for Harvard model runners

(aligned with Stanford defaults).
"""

from __future__ import annotations

# Ridge grid (parity with documented λ set)
RIDGE_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

# Random Forest grid (parity with src.experiments.run_random_forest)
RF_N_ESTIMATORS = [100]
RF_MAX_DEPTH = [5, 10, None]
RF_MIN_SAMPLES_SPLIT = [10, 20]
RF_MAX_FEATURES = ["sqrt", "log2"]

# Neural network defaults (match Stanford NumpyNN settings)
NN_DEFAULTS = {
    "hidden_dims": (64, 32),
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 256,
    "max_epochs": 500,
    "patience": 10,
    "seed": 42,
}
