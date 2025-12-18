"""Harvard walk-forward pipeline utilities."""

from __future__ import annotations

import logging
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning
from sklearn.preprocessing import StandardScaler

from src.models.baselines import fit_ridge
from src.models.neural_network import NNParams, train_model
from src.models.neural_network import predict as nn_predict
from src.models.random_forest import RFParams, fit_random_forest

from .config import (
    NN_DEFAULTS,
    RF_MAX_DEPTH,
    RF_MAX_FEATURES,
    RF_MIN_SAMPLES_SPLIT,
    RF_N_ESTIMATORS,
    RIDGE_ALPHAS,
)
from .features import FactorPanel

logger = logging.getLogger(__name__)

_EXCESS_RETURN_TARGET_RE = re.compile(r"^ExcessReturn_t\+(\d+)$")


def make_splits(
    dates: pd.Series,
    train_days: int = 1134,
    val_days: int = 126,
    purge_days: int = 21,
    test_days: int = 21,
    step_days: int = 21,
) -> list[dict]:
    """Generate rolling walk-forward splits on trading-day counts.

    Splits are defined on unique trading dates, then mapped back to panel row indices.
    """
    dts = pd.to_datetime(dates)

    # 1) Canonical trading calendar (unique sorted dates)
    cal = pd.Index(dts.unique()).sort_values()
    window = train_days + val_days + purge_days + test_days

    # 2) Map each trading date to row indices in the original panel
    # Using groupby(indices) is fast and returns positional indices
    date_to_rows = dts.groupby(dts).indices  # dict[Timestamp] -> ndarray row positions

    def rows_for(date_index: pd.Index) -> np.ndarray:
        return np.concatenate([date_to_rows[d] for d in date_index])

    splits: list[dict] = []
    for i in range(0, len(cal) - window + 1, step_days):
        train_dates = cal[i : i + train_days]
        val_dates = cal[i + train_days : i + train_days + val_days]
        purge_dates = cal[
            i + train_days + val_days : i + train_days + val_days + purge_days
        ]
        test_dates = cal[i + train_days + val_days + purge_days : i + window]

        splits.append(
            {
                "train_idx": rows_for(train_dates),
                "val_idx": rows_for(val_dates),
                "test_idx": rows_for(test_dates),
                "meta": {
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    "val_start": val_dates[0],
                    "val_end": val_dates[-1],
                    "purge_start": purge_dates[0],
                    "purge_end": purge_dates[-1],
                    "test_start": test_dates[0],
                    "test_end": test_dates[-1],
                },
            }
        )
    return splits


def _safe_predict(model, X_test: pd.DataFrame) -> np.ndarray:
    """Predict with guard against failures.

    Args:
        model: Fitted model exposing predict().
        X_test: Test feature matrix.

    Returns:
        Prediction array; NaNs if prediction fails.
    """
    try:
        preds = model.predict(X_test)
        if isinstance(preds, list):
            preds = np.asarray(preds)
        return np.asarray(preds, dtype=float)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Prediction failed: %s", exc)
        return np.full(len(X_test), np.nan, dtype=float)


def load_data(path: str | Path = "data/factor_panel_daily.parquet") -> FactorPanel:
    """Load the prepared factor panel for Harvard analysis.

    Args:
        path: Parquet path containing Date, PortfolioPair, features, and target.

    Returns:
        FactorPanel with attached summary helper.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Factor panel not found at {p}")
    df = pd.read_parquet(p)
    return FactorPanel(df)


def _ic_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman IC with NaN safety."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        ic = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return np.nan if not np.isfinite(ic) else float(ic)


def _filter_finite_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rows where y is finite (dropping aligned X rows)."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(y_arr)
    return X_arr[mask], y_arr[mask]


def compute_forward_excess_return(
    panel: pd.DataFrame,
    *,
    portfolio_col: str,
    date_col: str = "Date",
    excess_return_col: str = "ExcessReturn",
    risk_free_col: str = "RF",
    forward_return_trading_days: int = 21,
) -> pd.Series:
    """Compute a forward compounded excess return target over N trading days.

    For each portfolio and date t, this computes the N-trading-day forward
    excess return:

        ((∏_{i=1..N} (1 + R_{t+i})) / (∏_{i=1..N} (1 + RF_{t+i}))) - 1

    where R is the portfolio gross return. If only an excess-return column is available,
    gross returns are inferred as (ExcessReturn + RF) per day.
    """
    if forward_return_trading_days <= 0:
        raise ValueError("forward_return_trading_days must be positive.")

    required = {portfolio_col, date_col, excess_return_col, risk_free_col}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise KeyError(
            f"compute_forward_excess_return: missing required columns: {missing}"
        )

    # Sort within portfolio to ensure rolling windows are chronological.
    work = panel[[portfolio_col, date_col, excess_return_col, risk_free_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work = work.sort_values([portfolio_col, date_col], kind="mergesort")

    portfolio_gross_return = work[excess_return_col] + work[risk_free_col]
    log_gross_return = np.log1p(portfolio_gross_return)
    log_risk_free_return = np.log1p(work[risk_free_col])

    def _forward_log_sum(series: pd.Series) -> pd.Series:
        return (
            series.rolling(
                window=forward_return_trading_days,
                min_periods=forward_return_trading_days,
            )
            .sum()
            .shift(-forward_return_trading_days)
        )

    forward_log_gross = log_gross_return.groupby(work[portfolio_col]).transform(
        _forward_log_sum
    )
    forward_log_rf = log_risk_free_return.groupby(work[portfolio_col]).transform(
        _forward_log_sum
    )
    forward_excess_return = np.expm1(forward_log_gross - forward_log_rf)

    # Restore original panel row ordering.
    target = pd.Series(forward_excess_return.to_numpy(), index=work.index, name=None)
    return target.reindex(panel.index)


def _resolve_target_series(
    panel: pd.DataFrame,
    *,
    portfolio_col: str,
    target_col: str,
    date_col: str = "Date",
) -> pd.Series:
    """Return the target series, computing forward targets if needed."""
    if target_col in panel:
        return panel[target_col]

    match = _EXCESS_RETURN_TARGET_RE.match(target_col)
    if not match:
        raise KeyError(
            f"Target column '{target_col}' not found and is not a supported "
            "computed target."
        )

    forward_return_trading_days = int(match.group(1))
    return compute_forward_excess_return(
        panel,
        portfolio_col=portfolio_col,
        date_col=date_col,
        forward_return_trading_days=forward_return_trading_days,
    )


def run_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Ridge with alpha grid search on validation, parity with Stanford defaults."""
    X_test = np.asarray(X_test)
    X_train, y_train = _filter_finite_xy(X_train, y_train)
    X_val, y_val = _filter_finite_xy(X_val, y_val)
    if len(y_train) < 2:
        logger.warning(
            "Ridge: insufficient finite training targets; returning NaN predictions."
        )
        return np.full(len(X_test), np.nan, dtype=float)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xval = scaler.transform(X_val)
    ytr = y_train
    yva = y_val

    best_alpha = None
    best_ic = -np.inf
    if len(y_val) >= 2:
        for alpha in RIDGE_ALPHAS:
            model = fit_ridge(Xtr, ytr, alpha=alpha)
            preds_val = model.predict(Xval)
            ic = _ic_spearman(yva, preds_val)
            if np.isnan(ic):
                continue
            if ic > best_ic:
                best_ic = ic
                best_alpha = alpha

    if best_alpha is None:
        best_alpha = RIDGE_ALPHAS[0]

    # Refit on train+val with best alpha and predict test.
    X_fit = X_train
    y_fit = y_train
    if len(y_val) >= 2:
        X_fit = np.vstack([X_fit, X_val])
        y_fit = np.concatenate([y_fit, y_val])
    scaler_final = StandardScaler()
    Xf = scaler_final.fit_transform(X_fit)
    Xte = scaler_final.transform(X_test)
    model_final = fit_ridge(Xf, y_fit, alpha=best_alpha)
    return _safe_predict(model_final, Xte)


def run_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    rf_n_jobs: int = -1,
) -> np.ndarray:
    """Random Forest grid search on validation (parity with Stanford defaults)."""
    best_cfg = None
    best_ic = -np.inf

    Xtr, ytr = _filter_finite_xy(X_train, y_train)
    Xva, yva = _filter_finite_xy(X_val, y_val)
    if len(ytr) < 2:
        logger.warning(
            "RandomForest: insufficient finite training targets; returning NaN "
            "predictions."
        )
        return np.full(len(X_test), np.nan, dtype=float)

    if len(yva) >= 2:
        for ne in RF_N_ESTIMATORS:
            for depth in RF_MAX_DEPTH:
                for mss in RF_MIN_SAMPLES_SPLIT:
                    for mf in RF_MAX_FEATURES:
                        params = RFParams(
                            n_estimators=ne,
                            max_depth=depth,
                            min_samples_split=mss,
                            max_features=mf,
                            random_state=42,
                            n_jobs=rf_n_jobs,
                        )
                        model = fit_random_forest(Xtr, ytr, params)
                        preds_val = model.predict(Xva)
                        ic = _ic_spearman(yva, preds_val)
                        if np.isnan(ic):
                            continue
                        if ic > best_ic:
                            best_ic = ic
                            best_cfg = params

    if best_cfg is None:
        best_cfg = RFParams()

    X_fit = np.vstack([Xtr, Xva])
    y_fit = np.concatenate([ytr, yva])
    model_final = fit_random_forest(X_fit, y_fit, best_cfg)
    return _safe_predict(model_final, np.asarray(X_test))


def run_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Train NumpyNN with val-based early stopping (parity with Stanford defaults)."""
    X_test = np.asarray(X_test)
    X_train, y_train = _filter_finite_xy(X_train, y_train)
    X_val, y_val = _filter_finite_xy(X_val, y_val)
    if len(y_train) < 2 or len(y_val) < 2:
        logger.warning(
            "NeuralNetwork: insufficient finite train/val targets; returning "
            "NaN predictions."
        )
        return np.full(len(X_test), np.nan, dtype=float)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    Xte = scaler.transform(X_test)
    ytr = y_train
    yva = y_val

    params = NNParams(
        input_dim=Xtr.shape[1],
        output_dim=1,
        hidden_dims=NN_DEFAULTS["hidden_dims"],
        dropout=NN_DEFAULTS["dropout"],
        learning_rate=NN_DEFAULTS["learning_rate"],
        weight_decay=NN_DEFAULTS["weight_decay"],
        batch_size=NN_DEFAULTS["batch_size"],
        max_epochs=NN_DEFAULTS["max_epochs"],
        patience=NN_DEFAULTS["patience"],
        seed=NN_DEFAULTS["seed"],
    )
    try:
        model, _ = train_model(Xtr, ytr, Xva, yva, params)
        preds = nn_predict(model, Xte)
        return np.asarray(preds, dtype=float).ravel()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Neural network fit failed: %s", exc)
        return np.full(len(X_test), np.nan, dtype=float)


def run_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    *,
    val_dates: pd.Series | np.ndarray | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """XGBoost helper with val-IC grid search and refit on train+val."""
    try:
        from xgboost import XGBRegressor  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "run_xgboost requires xgboost. Install with `pip install xgboost`."
        ) from exc

    X_test = np.asarray(X_test)
    Xtr, ytr = _filter_finite_xy(X_train, y_train)

    # Validation with aligned dates (drop non-finite y)
    yva_raw = np.asarray(y_val, dtype=float).reshape(-1)
    Xva_raw = np.asarray(X_val)
    dates_va = (
        np.asarray(val_dates) if val_dates is not None else np.arange(len(yva_raw))
    )
    m_va = np.isfinite(yva_raw)
    Xva = Xva_raw[m_va]
    yva = yva_raw[m_va]
    dates_va = dates_va[m_va]

    if len(ytr) < 2:
        logger.warning(
            "XGBoost: insufficient finite training targets; returning NaN predictions."
        )
        return np.full(len(X_test), np.nan, dtype=float)

    base_defaults = {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": 4,
        "tree_method": "hist",
    }
    if params:
        base_defaults.update(params)

    param_grid: list[dict] = []
    for md in [2, 3]:
        for lr in [0.03, 0.05]:
            for ne in [300, 600]:
                for mcw in [5, 10]:
                    for rl in [1.0, 5.0]:
                        param_grid.append(
                            {
                                **base_defaults,
                                "max_depth": md,
                                "learning_rate": lr,
                                "n_estimators": ne,
                                "min_child_weight": mcw,
                                "reg_lambda": rl,
                            }
                        )
    if not param_grid:
        param_grid = [base_defaults]

    best_params = None
    best_ic = -np.inf
    for cfg in param_grid:
        model = XGBRegressor(**cfg)
        model.fit(Xtr, ytr, verbose=False)
        preds_val = model.predict(Xva) if len(yva) else np.full(len(yva), np.nan)
        val_df = pd.DataFrame(
            {
                "Date": dates_va,
                "Actual": yva,
                "pred_xgb": preds_val,
            }
        )
        ic_val_df = _spearman_ic_by_date(
            val_df, date_col="Date", actual_col="Actual", pred_col="pred_xgb"
        )
        mean_ic = ic_val_df["IC"].mean(skipna=True)
        if np.isnan(mean_ic):
            continue
        if mean_ic > best_ic:
            best_ic = mean_ic
            best_params = cfg

    if best_params is None:
        best_params = param_grid[0]

    # Refit on train+val with best params
    X_fit = Xtr if len(yva) == 0 else np.vstack([Xtr, Xva])
    y_fit = ytr if len(yva) == 0 else np.concatenate([ytr, yva])
    model_final = XGBRegressor(**best_params)
    model_final.fit(X_fit, y_fit, verbose=False)
    return _safe_predict(model_final, np.asarray(X_test))


@dataclass
class XGBoostArtifacts:
    """Outputs from the optional boosted-tree extension."""

    predictions: pd.DataFrame
    ic_by_date: pd.DataFrame | None
    ic_rolling: pd.DataFrame | None
    summary_table: pd.DataFrame | None
    fold_ic: pd.DataFrame | None
    feature_importance: pd.DataFrame | None
    ic_series: pd.DataFrame | None  # legacy alias for ic_by_date
    model_params: dict
    backend: str


_XGB_TARGET_CANDIDATES = [
    "Actual",
    "y",
    "target",
    "ExcessReturn_t+1",
    "ExcessReturn_t+21",
]


def _infer_xgb_columns(
    features: pd.DataFrame,
    *,
    date_col: str,
    portfolio_col: str | None,
    target_col: str | None,
) -> tuple[str, str, list[str]]:
    """Infer portfolio, target, and feature columns for the XGB extension."""
    if date_col not in features.columns:
        raise KeyError(f"run_xgboost_extension: features missing '{date_col}' column.")

    port_col = portfolio_col
    if port_col is None:
        if "PortfolioPair" in features.columns:
            port_col = "PortfolioPair"
        elif "Portfolio" in features.columns:
            port_col = "Portfolio"
    if port_col is None or port_col not in features.columns:
        raise KeyError(
            "run_xgboost_extension: features must contain 'PortfolioPair' or "
            "'Portfolio'."
        )

    tgt_col = target_col
    if tgt_col is None:
        tgt_col = next(
            (c for c in _XGB_TARGET_CANDIDATES if c in features.columns), None
        )
    if tgt_col is None or tgt_col not in features.columns:
        raise KeyError(
            "run_xgboost_extension: target column not found. "
            f"Provide target_col or include one of {', '.join(_XGB_TARGET_CANDIDATES)}."
        )

    future_return_cols = [
        c for c in features.columns if _EXCESS_RETURN_TARGET_RE.match(c)
    ]
    drop = {
        date_col,
        port_col,
        tgt_col,
        "Source",
        "fold",
        "ExcessReturn",
        "Return",
    } | set(future_return_cols)
    feature_cols = [
        c for c in features.columns if c not in drop and not c.startswith("pred_")
    ]
    if not feature_cols:
        raise ValueError(
            "run_xgboost_extension: no feature columns found after exclusions."
        )

    return port_col, tgt_col, feature_cols


def _spearman_ic_by_date(
    df: pd.DataFrame,
    *,
    date_col: str = "Date",
    actual_col: str = "Actual",
    pred_col: str = "pred_xgb",
) -> pd.DataFrame:
    """Compute cross-sectional Spearman IC per date."""
    need = [date_col, actual_col, pred_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"_spearman_ic_by_date: missing columns {missing}")
    work = df[[date_col, actual_col, pred_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    records: list[dict] = []
    for d, g in work.groupby(date_col, sort=True):
        ic_val = _ic_spearman(
            np.asarray(g[actual_col], dtype=float),
            np.asarray(g[pred_col], dtype=float),
        )
        records.append({"Date": pd.to_datetime(d), "IC": ic_val})
    return pd.DataFrame(records).sort_values("Date").reset_index(drop=True)


def run_xgboost_extension(
    *,
    splits: list[dict],
    X: pd.DataFrame | np.ndarray | None = None,
    y: pd.Series | np.ndarray | None = None,
    features: pd.DataFrame | None = None,
    target_col: str = "ExcessReturn_t+21",
    date_col: str = "Date",
    portfolio_col: str = "PortfolioPair",
    params: dict | None = None,
    window: int = 126,
) -> XGBoostArtifacts:
    """Fit XGBoost using the provided walk-forward splits with val-IC model selection.

    Mirrors the main pipeline: strict column inference, no early stopping,
    val IC selection, refit on train+val, predict test, and compute daily
    cross-sectional IC.
    """
    # Require xgboost explicitly (no silent fallback)
    try:
        from xgboost import XGBRegressor  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "run_xgboost_extension requires xgboost. Install with "
            "`pip install xgboost`."
        ) from exc

    # Resolve feature matrix / target vector
    dates = None
    ports = None
    if X is None or y is None:
        if features is None:
            raise ValueError(
                "run_xgboost_extension: provide X/y or a features DataFrame."
            )
        port_col, tgt_col, feat_cols = _infer_xgb_columns(
            features,
            date_col=date_col,
            portfolio_col=portfolio_col,
            target_col=target_col
            if (target_col and target_col in features.columns)
            else None,
        )
        dates = pd.to_datetime(features[date_col]) if date_col in features else None
        ports = features[port_col] if port_col in features else None
        X = features[feat_cols]
        y = features[tgt_col]
    else:
        port_col = portfolio_col
        tgt_col = target_col
        feat_cols = (
            list(X.columns)
            if isinstance(X, pd.DataFrame)
            else [f"x{i}" for i in range(np.asarray(X).shape[1])]
        )

    X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    feature_names = list(X.columns)
    X_arr = X.to_numpy()

    if len(X_arr) != len(y_arr):
        raise ValueError(
            "run_xgboost_extension: X and y length mismatch "
            f"({len(X_arr)} vs {len(y_arr)})."
        )

    backend = "xgboost"
    base_defaults = {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": 4,
        "tree_method": "hist",
    }
    if params:
        base_defaults.update(params)

    param_grid: list[dict] = []
    for md in [2, 3]:
        for lr in [0.03, 0.05]:
            for ne in [300, 600]:
                for mcw in [5, 10]:
                    for rl in [1.0, 5.0]:
                        param_grid.append(
                            {
                                **base_defaults,
                                "max_depth": md,
                                "learning_rate": lr,
                                "n_estimators": ne,
                                "min_child_weight": mcw,
                                "reg_lambda": rl,
                            }
                        )
    if not param_grid:
        param_grid = [base_defaults]

    pred_blocks: list[pd.DataFrame] = []
    fold_ic_rows: list[dict] = []
    importances: list[np.ndarray] = []

    for fold_id, split in enumerate(splits, start=1):
        tr = np.asarray(split["train_idx"], dtype=int)
        va = np.asarray(split["val_idx"], dtype=int)
        te = np.asarray(split["test_idx"], dtype=int)

        X_tr, y_tr = _filter_finite_xy(X_arr[tr], y_arr[tr])
        y_va_raw = y_arr[va]
        finite_mask = np.isfinite(y_va_raw)
        X_va = X_arr[va][finite_mask]
        y_va = y_va_raw[finite_mask]
        val_dates = (
            np.asarray(dates)[va][finite_mask]
            if dates is not None
            else np.arange(len(y_va))
        )
        X_te = X_arr[te]
        y_te = y_arr[te]

        if len(y_tr) < 10:
            logger.warning(
                "XGBoost fold %s: insufficient finite training targets; "
                "returning NaN predictions.",
                fold_id,
            )
            preds = np.full(len(X_te), np.nan, dtype=float)
            fi = None
        else:
            best_ic = -np.inf
            best_params = None

            for cfg in param_grid:
                model = XGBRegressor(**cfg)
                model.fit(X_tr, y_tr, verbose=False)
                preds_val = (
                    model.predict(X_va) if len(y_va) else np.full(len(X_va), np.nan)
                )
                val_df = pd.DataFrame(
                    {
                        "Date": val_dates,
                        "Actual": y_va,
                        "pred_xgb": preds_val,
                    }
                )
                ic_val_df = _spearman_ic_by_date(
                    val_df, date_col="Date", actual_col="Actual", pred_col="pred_xgb"
                )
                mean_ic = ic_val_df["IC"].mean(skipna=True)
                fold_ic_rows.append(
                    {"fold": fold_id, "mean_val_ic": mean_ic, "params": cfg}
                )
                if np.isnan(mean_ic):
                    continue
                if mean_ic > best_ic:
                    best_ic = mean_ic
                    best_params = cfg

            if best_params is None:
                best_params = base_defaults

            # Refit on train+val with best params, then predict test
            model = XGBRegressor(**best_params)
            X_fit = X_tr if len(y_va) == 0 else np.vstack([X_tr, X_va])
            y_fit = y_tr if len(y_va) == 0 else np.concatenate([y_tr, y_va])
            model.fit(X_fit, y_fit, verbose=False)
            preds = model.predict(X_te)
            fi = getattr(model, "feature_importances_", None)

        block = pd.DataFrame(
            {
                "Actual": y_te,
                "pred_xgb": np.asarray(preds, dtype=float),
                "fold": fold_id,
            }
        )
        if dates is not None:
            block["Date"] = np.asarray(dates)[te]
        if ports is not None:
            block["PortfolioPair"] = np.asarray(ports)[te]

        ordered_cols = [
            c
            for c in ("Date", "PortfolioPair", "Actual", "pred_xgb", "fold")
            if c in block
        ]
        block = block[ordered_cols]
        pred_blocks.append(block)

        if fi is not None and len(fi) == len(feature_names):
            importances.append(np.asarray(fi, dtype=float))

    predictions = (
        pd.concat(pred_blocks, ignore_index=True)
        if pred_blocks
        else pd.DataFrame(columns=["Actual", "pred_xgb", "fold"])
    )

    if {"Date", "PortfolioPair"}.issubset(predictions.columns):
        dup = predictions.groupby(["Date", "PortfolioPair"]).size()
        if dup.max() > 1:
            raise ValueError(
                "run_xgboost_extension: duplicate (Date, PortfolioPair) rows "
                f"detected (max={dup.max()})."
            )

    ic_by_date = None
    ic_series = None
    if "Date" in predictions.columns and len(predictions):
        ic_by_date = _spearman_ic_by_date(
            predictions, date_col="Date", actual_col="Actual", pred_col="pred_xgb"
        )
        ic_by_date["IC_rolling"] = (
            ic_by_date["IC"].rolling(window=window, min_periods=window).mean()
        )
        ic_series = ic_by_date.rename(
            columns={"IC": "ic", "IC_rolling": "ic_roll"}
        )  # alias for compatibility

    fi_df = None
    if importances:
        fi_mean = np.nanmean(np.vstack(importances), axis=0)
        fi_df = (
            pd.DataFrame({"feature": feature_names, "importance": fi_mean})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    summary_table = None
    if ic_by_date is not None:
        summary_table = pd.DataFrame(
            [{"model": "XGBoost", "mean_ic": float(ic_by_date["IC"].mean(skipna=True))}]
        )
        mean_ic_val = float(ic_by_date["IC"].mean(skipna=True))
        if abs(mean_ic_val) > 0.05:
            logger.warning(
                "run_xgboost_extension: mean IC %.4f is unusually large; check "
                "for leakage.",
                mean_ic_val,
            )

    return XGBoostArtifacts(
        predictions=predictions,
        ic_by_date=ic_by_date,
        ic_rolling=ic_by_date[["Date", "IC_rolling"]].rename(
            columns={"IC_rolling": "XGBoost"}
        )
        if ic_by_date is not None
        else None,
        summary_table=summary_table,
        fold_ic=pd.DataFrame(fold_ic_rows) if fold_ic_rows else None,
        feature_importance=fi_df,
        ic_series=ic_series,
        model_params=base_defaults,
        backend=backend,
    )


@dataclass
class WalkForwardArtifacts:
    """Outputs from the Harvard walk-forward run."""

    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame


def run_walkforward_experiment(
    panel: pd.DataFrame,
    splits: list[dict],
    *,
    portfolio_col: str,
    target_col: str = "ExcessReturn_t+21",
    date_col: str = "Date",
    progress: bool = True,
    include_xgb: bool = False,
) -> WalkForwardArtifacts:
    """Run walk-forward training/prediction and return test-window predictions.

    Args:
        panel: Full panel DataFrame containing date_col, portfolio_col, features,
            and target_col.
        splits: Output of make_splits(...), containing train/val/test indices
            (row positions).
        portfolio_col: Name of the portfolio identifier column (required).
        target_col: Name of the target column (default: ExcessReturn_t+21). If the
            column is not present, forward targets of the form
            `ExcessReturn_t+{N}` are computed from `ExcessReturn` and `RF`.
        date_col: Name of the date column (default: Date).

    Returns:
        WalkForwardArtifacts with:
          - predictions: columns [Date, <portfolio_col renamed to PortfolioPair>,
                                 Actual, pred_ridge, pred_rf, pred_nn, fold]
          - fold_metrics: fold-level Spearman IC per model (descriptive)
    """
    # --- Validate schema (fail fast; no guessing) ---
    required = [date_col, portfolio_col]
    missing = [c for c in required if c not in panel]
    if missing:
        raise KeyError(
            f"run_walkforward_experiment: missing required columns: {missing}. "
            f"Available columns: {list(panel)}"
        )

    df = panel  # alias
    dates = pd.to_datetime(df[date_col])
    y_series = _resolve_target_series(
        df, portfolio_col=portfolio_col, target_col=target_col, date_col=date_col
    )

    # Feature columns: exclude identifiers/target and known non-features if present.
    future_return_cols = [c for c in df.columns if _EXCESS_RETURN_TARGET_RE.match(c)]
    exclude = {date_col, portfolio_col, "Source", "ExcessReturn"} | set(
        future_return_cols
    )
    feat_cols = [c for c in df if c not in exclude]
    if not feat_cols:
        raise ValueError(
            "run_walkforward_experiment: no feature columns found after exclusions. "
            f"Excluded={sorted(exclude)}"
        )

    # Slice only what we need to keep memory tight
    X_all = df[feat_cols]
    y_all = y_series
    p_all = df[portfolio_col]

    pred_blocks: list[pd.DataFrame] = []
    metric_rows: list[dict] = []

    # --- Progress bar over folds ---
    iterator = enumerate(splits, start=1)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(
            iterator,
            total=len(splits),
            desc="Walk-forward (fit/predict)",
            leave=True,  # IMPORTANT for notebooks: leave=True so you can see it.
        )

    for fold_id, split in iterator:
        tr = np.asarray(split["train_idx"], dtype=int)
        va = np.asarray(split["val_idx"], dtype=int)
        te = np.asarray(split["test_idx"], dtype=int)

        X_tr, y_tr = X_all.iloc[tr], y_all.iloc[tr]
        X_va, y_va = X_all.iloc[va], y_all.iloc[va]
        X_te, y_te = X_all.iloc[te], y_all.iloc[te].to_numpy(dtype=float)

        pred_ridge = run_ridge(X_tr, y_tr, X_va, y_va, X_te)
        pred_rf = run_random_forest(X_tr, y_tr, X_va, y_va, X_te)
        pred_nn = run_neural_network(X_tr, y_tr, X_va, y_va, X_te)
        pred_xgb = None
        if include_xgb:
            pred_xgb = run_xgboost(
                X_tr, y_tr, X_va, y_va, X_te, val_dates=dates.iloc[va]
            )

        block = pd.DataFrame(
            {
                "Date": dates.iloc[te].to_numpy(),
                "PortfolioPair": p_all.iloc[te].to_numpy(),  # normalized for downstream
                "Actual": y_te,
                "pred_ridge": np.asarray(pred_ridge, dtype=float),
                "pred_rf": np.asarray(pred_rf, dtype=float),
                "pred_nn": np.asarray(pred_nn, dtype=float),
                "fold": fold_id,
            }
        )
        if include_xgb and pred_xgb is not None:
            block["pred_xgb"] = np.asarray(pred_xgb, dtype=float)
        pred_blocks.append(block)

        metric_rows.append(
            {"fold": fold_id, "model": "ridge", "ic": _ic_spearman(y_te, pred_ridge)}
        )
        metric_rows.append(
            {"fold": fold_id, "model": "rf", "ic": _ic_spearman(y_te, pred_rf)}
        )
        metric_rows.append(
            {"fold": fold_id, "model": "nn", "ic": _ic_spearman(y_te, pred_nn)}
        )
        if include_xgb and pred_xgb is not None:
            metric_rows.append(
                {"fold": fold_id, "model": "xgb", "ic": _ic_spearman(y_te, pred_xgb)}
            )

    predictions = (
        pd.concat(pred_blocks, ignore_index=True)
        .sort_values(["Date", "PortfolioPair"], kind="mergesort")
        .reset_index(drop=True)
    )
    fold_metrics = pd.DataFrame(metric_rows)

    # Optional tightening
    predictions["PortfolioPair"] = predictions["PortfolioPair"].astype("category")
    predictions["fold"] = predictions["fold"].astype("int16")

    return WalkForwardArtifacts(predictions=predictions, fold_metrics=fold_metrics)


# --------------------------------------------------------------------------- #
# Parallel runner (process pool with shared globals)
# --------------------------------------------------------------------------- #
_X_GLOBAL: np.ndarray | None = None
_Y_GLOBAL: np.ndarray | None = None
_DATES_GLOBAL: np.ndarray | None = None
_PORTS_GLOBAL: np.ndarray | None = None
_RF_JOBS: int = -1
_INCLUDE_XGB: bool = False


def _init_globals(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    ports: np.ndarray,
    rf_n_jobs: int,
    include_xgb: bool,
) -> None:
    """Initializer for worker processes to set module-level globals."""
    global _X_GLOBAL, _Y_GLOBAL, _DATES_GLOBAL, _PORTS_GLOBAL, _RF_JOBS, _INCLUDE_XGB
    _X_GLOBAL = X
    _Y_GLOBAL = y
    _DATES_GLOBAL = dates
    _PORTS_GLOBAL = ports
    _RF_JOBS = rf_n_jobs
    _INCLUDE_XGB = include_xgb


def _run_fold(args: tuple[int, dict]) -> tuple[pd.DataFrame, dict]:
    """Worker to run a single fold using module-level globals."""
    fold_id, split = args
    X = _X_GLOBAL
    y = _Y_GLOBAL
    dates = _DATES_GLOBAL
    ports = _PORTS_GLOBAL
    if X is None or y is None or dates is None or ports is None:
        raise RuntimeError("Global arrays not initialized in worker.")

    tr = np.asarray(split["train_idx"], dtype=int)
    va = np.asarray(split["val_idx"], dtype=int)
    te = np.asarray(split["test_idx"], dtype=int)

    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]

    pred_ridge = run_ridge(X_tr, y_tr, X_va, y_va, X_te)
    pred_rf = run_random_forest(X_tr, y_tr, X_va, y_va, X_te, rf_n_jobs=_RF_JOBS)
    pred_nn = run_neural_network(X_tr, y_tr, X_va, y_va, X_te)
    pred_xgb = None
    if _INCLUDE_XGB:
        pred_xgb = run_xgboost(
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_te,
            val_dates=dates[va],
            params={"n_jobs": 1},
        )

    block = pd.DataFrame(
        {
            "Date": dates[te],
            "PortfolioPair": ports[te],
            "Actual": y_te,
            "pred_ridge": np.asarray(pred_ridge, dtype=float),
            "pred_rf": np.asarray(pred_rf, dtype=float),
            "pred_nn": np.asarray(pred_nn, dtype=float),
            "fold": fold_id,
        }
    )
    if pred_xgb is not None:
        block["pred_xgb"] = np.asarray(pred_xgb, dtype=float)

    metrics = [
        {"fold": fold_id, "model": "ridge", "ic": _ic_spearman(y_te, pred_ridge)},
        {"fold": fold_id, "model": "rf", "ic": _ic_spearman(y_te, pred_rf)},
        {"fold": fold_id, "model": "nn", "ic": _ic_spearman(y_te, pred_nn)},
    ]
    if pred_xgb is not None:
        metrics.append(
            {"fold": fold_id, "model": "xgb", "ic": _ic_spearman(y_te, pred_xgb)}
        )
    return block, metrics


def run_walkforward_experiment_parallel(
    panel: pd.DataFrame,
    splits: list[dict],
    *,
    portfolio_col: str,
    target_col: str = "ExcessReturn_t+21",
    date_col: str = "Date",
    max_workers: int = 4,
    rf_n_jobs: int = 1,
    progress: bool = True,
    include_xgb: bool = False,
) -> WalkForwardArtifacts:
    """Parallel walk-forward runner using a process pool and shared numpy arrays.

    Args:
        panel: Full panel DataFrame.
        splits: Output of make_splits(...).
        portfolio_col: Portfolio identifier column.
        target_col: Target column name.
        date_col: Date column name.
        max_workers: Number of worker processes.
        rf_n_jobs: n_jobs passed into RandomForest when parallelizing folds.
        progress: Show tqdm over folds.
    """
    required = [date_col, portfolio_col]
    missing = [c for c in required if c not in panel]
    if missing:
        raise KeyError(f"run_walkforward_experiment_parallel: missing {missing}")

    y_series = _resolve_target_series(
        panel, portfolio_col=portfolio_col, target_col=target_col, date_col=date_col
    )

    future_return_cols = [c for c in panel.columns if _EXCESS_RETURN_TARGET_RE.match(c)]
    exclude = {date_col, portfolio_col, "Source", "ExcessReturn"} | set(
        future_return_cols
    )
    feat_cols = [c for c in panel if c not in exclude]
    if not feat_cols:
        raise ValueError(
            "run_walkforward_experiment_parallel: no feature columns found."
        )

    X_np = panel[feat_cols].to_numpy()
    y_np = y_series.to_numpy()
    dates_np = pd.to_datetime(panel[date_col]).to_numpy()
    ports_np = panel[portfolio_col].to_numpy()

    tasks = list(enumerate(splits, start=1))
    pred_blocks: list[pd.DataFrame] = []
    metric_rows: list[dict] = []

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_globals,
        initargs=(X_np, y_np, dates_np, ports_np, rf_n_jobs, include_xgb),
    ) as pool:
        futures = [pool.submit(_run_fold, args) for args in tasks]
        if progress:
            from tqdm import tqdm

            futures_iter = tqdm(
                futures, total=len(futures), desc="Walk-forward (parallel)", leave=True
            )
        else:
            futures_iter = futures
        for fut in futures_iter:
            block, metrics = fut.result()
            pred_blocks.append(block)
            metric_rows.extend(metrics)

    predictions = (
        pd.concat(pred_blocks, ignore_index=True)
        .sort_values(["Date", "PortfolioPair"], kind="mergesort")
        .reset_index(drop=True)
    )
    fold_metrics = pd.DataFrame(metric_rows)
    predictions["PortfolioPair"] = predictions["PortfolioPair"].astype("category")
    predictions["fold"] = predictions["fold"].astype("int16")
    return WalkForwardArtifacts(predictions=predictions, fold_metrics=fold_metrics)
