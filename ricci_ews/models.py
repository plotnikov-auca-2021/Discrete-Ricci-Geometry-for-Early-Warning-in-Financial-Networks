"""
models.py
This module provides:

  1) A simple FEATURE GROUPING convention:
       - 'baseline':   non-curvature features (market/topology/eigenmode),
                       i.e. columns starting with any of:
                           'market_', 'topo_', 'eigen_'
       - 'curvature':  curvature-based features (with/without EWMA),
                       i.e. columns starting with:
                           'corr_', 'prec_'
       - 'all':        union of curvature and baseline features.

  2) Model specs and factory functions for scikit-learn models:
       - Logistic regression with L2 penalty (balanced classes)
       - Logistic regression with L1 penalty (optional)
       - Random forest classifier

  3) Training and evaluation helpers:
       - fit_model(..):       train a model from a ModelSpec
       - predict_proba(..):   probability predictions
       - evaluate_model(..):  ROC AUC, PR AUC, Brier score, log loss, etc.

This module does NOT implement any walk-forward logic; that will live in
pipeline.py. Here we focus on model definitions and core ML utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

from . import config


# ----------------------------------------------------------------------
# Feature grouping helpers
# ----------------------------------------------------------------------


BASELINE_PREFIXES: Tuple[str, ...] = ("market_", "topo_", "eigen_")
CURVATURE_PREFIXES: Tuple[str, ...] = ("corr_", "prec_")


def select_feature_columns(
    X: pd.DataFrame,
    feature_group: str = "all",
) -> List[str]:
    """
    Select columns from X according to feature_group.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (rows = dates, columns = features).
    feature_group : {"baseline", "curvature", "all"}
        - "baseline": columns starting with 'market_', 'topo_', 'eigen_'
        - "curvature": columns starting with 'corr_', 'prec_'
        - "all": union of baseline + curvature

    Returns
    -------
    cols : list of str
        Column names to use for this feature group.
    """
    all_cols = list(X.columns)

    def _has_prefix(col: str, prefixes: Tuple[str, ...]) -> bool:
        return any(col.startswith(p) for p in prefixes)

    if feature_group == "baseline":
        cols = [c for c in all_cols if _has_prefix(c, BASELINE_PREFIXES)]
    elif feature_group == "curvature":
        cols = [c for c in all_cols if _has_prefix(c, CURVATURE_PREFIXES)]
    elif feature_group == "all":
        baseline_cols = [c for c in all_cols if _has_prefix(c, BASELINE_PREFIXES)]
        curvature_cols = [c for c in all_cols if _has_prefix(c, CURVATURE_PREFIXES)]
        cols = sorted(set(baseline_cols) | set(curvature_cols))
    else:
        raise ValueError(
            f"Unknown feature_group='{feature_group}'. "
            f"Expected one of ['baseline', 'curvature', 'all']."
        )

    return cols


# ----------------------------------------------------------------------
# Model specifications
# ----------------------------------------------------------------------


@dataclass
class ModelSpec:
    """
    Specification for a predictive model.

    Attributes
    ----------
    name : str
        Human-readable model name (used as key in results).
    base_estimator : {"logit_l2", "logit_l1", "rf"}
        Type of underlying estimator.
    feature_group : {"baseline", "curvature", "all"}
        Which feature subset to use.
    params : dict
        Hyperparameters for the underlying estimator.
    """

    name: str
    base_estimator: str
    feature_group: str = "all"
    params: Dict[str, Any] | None = None


def get_default_model_specs() -> List[ModelSpec]:
    """
    Default set of model specifications.
    """
    rs = getattr(config, "RANDOM_STATE", 42)

    specs: List[ModelSpec] = [
        ModelSpec(
            name="logit_baseline",
            base_estimator="logit_l2",
            feature_group="baseline",
            params={
                "C": 1.0,
                "max_iter": 500,
                "class_weight": "balanced",
            },
        ),
        ModelSpec(
            name="logit_full",
            base_estimator="logit_l2",
            feature_group="all",
            params={
                "C": 1.0,
                "max_iter": 500,
                "class_weight": "balanced",
            },
        ),
        ModelSpec(
            name="rf_full",
            base_estimator="rf",
            feature_group="all",
            params={
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "class_weight": "balanced",
                "random_state": rs,
                "n_jobs": -1,
            },
        ),
    ]
    return specs


# ----------------------------------------------------------------------
# Model factories
# ----------------------------------------------------------------------


def make_sklearn_estimator(spec: ModelSpec) -> BaseEstimator:
    """
    Build an unfitted scikit-learn estimator (possibly a Pipeline)
    from a ModelSpec.
    """
    base = spec.base_estimator.lower()
    params = dict(spec.params or {})
    rs = getattr(config, "RANDOM_STATE", 42)

    if base == "logit_l2":
        C = params.pop("C", 1.0)
        max_iter = params.pop("max_iter", 500)
        class_weight = params.pop("class_weight", "balanced")

        logit = LogisticRegression(
            penalty="l2",
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver="lbfgs",
            n_jobs=-1,
            random_state=rs,
            **params,
        )
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", logit),
            ]
        )

    elif base == "logit_l1":
        C = params.pop("C", 1.0)
        max_iter = params.pop("max_iter", 1000)
        class_weight = params.pop("class_weight", "balanced")

        logit = LogisticRegression(
            penalty="l1",
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver="saga",
            n_jobs=-1,
            random_state=rs,
            **params,
        )
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", logit),
            ]
        )

    elif base == "rf":
        model = RandomForestClassifier(**params)

    else:
        raise ValueError(f"Unknown base_estimator='{spec.base_estimator}'.")

    return model


# ----------------------------------------------------------------------
# Training & prediction helpers
# ----------------------------------------------------------------------


@dataclass
class FittedModel:
    """
    Container for a fitted model.

    Attributes
    ----------
    spec : ModelSpec
        Model specification used for training.
    feature_columns : list of str
        Columns in X used to train this model.
    estimator : BaseEstimator
        Fitted scikit-learn object (Pipeline or bare estimator).
    """

    spec: ModelSpec
    feature_columns: List[str]
    estimator: BaseEstimator

    # ---- key fix: provide decision_function passthrough or fallback ----
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Provide a score usable for ranking-based metrics.
        Prefer estimator.decision_function if available; otherwise fall back
        to probabilities for class 1.
        """
        est = self.estimator
        if hasattr(est, "decision_function"):
            s = est.decision_function(X)
            return np.asarray(s, dtype=float).ravel()

        # Fall back to probability "score"
        if hasattr(est, "predict_proba"):
            p = est.predict_proba(X)
            if p.ndim == 2 and p.shape[1] >= 2:
                return np.asarray(p[:, 1], dtype=float).ravel()
            return np.asarray(p, dtype=float).ravel()

        # Last resort: class labels as score
        if hasattr(est, "predict"):
            return np.asarray(est.predict(X), dtype=float).ravel()

        raise AttributeError("Underlying estimator has neither decision_function nor predict_proba nor predict.")


def fit_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
) -> FittedModel:
    """
    Fit a model according to a ModelSpec.
    """
    cols = select_feature_columns(X_train, spec.feature_group)
    if not cols:
        raise ValueError(
            f"No columns selected for feature_group='{spec.feature_group}'. "
            "Check your feature naming conventions."
        )

    X_sub = X_train[cols]
    y_arr = np.asarray(y_train, dtype=float)

    est = make_sklearn_estimator(spec)
    est.fit(X_sub, y_arr)

    return FittedModel(
        spec=spec,
        feature_columns=cols,
        estimator=est,
    )


def predict_proba(
    fitted: FittedModel,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Predict P(y=1 | X) for a fitted model.
    """
    cols = fitted.feature_columns
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(
            f"Missing columns in X for prediction: {missing}. "
            "Ensure feature construction is consistent between train/test."
        )

    X_sub = X[cols]
    est = fitted.estimator

    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X_sub)[:, 1]
    else:
        # fallback: decision scores -> sigmoid
        scores = est.decision_function(X_sub)
        p = 1.0 / (1.0 + np.exp(-scores))

    return np.asarray(p, dtype=float)


# ----------------------------------------------------------------------
# Evaluation utilities
# ----------------------------------------------------------------------


def _safe_metric(fn, y_true, y_score, default=np.nan) -> float:
    """
    Safely compute a metric; if something fails (e.g. only one class
    present), return default instead of raising.
    """
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return float(default)


def evaluate_model(
    fitted: FittedModel,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Safe evaluation for rare-event labels.

    IMPORTANT FIXES vs previous version:
      - Uses our own `predict_proba(fitted, ...)` so feature column selection is correct.
      - Does NOT require `decision_function` at all (but FittedModel now supports it anyway).
      - Computes log_loss even for single-class y_test using labels=[0,1].
      - Computes ROC AUC / PR AUC only when both classes are present.
      - Returns consistent metric keys used by the pipeline (accuracy, precision, recall, f1, brier, logloss, auc, ap).
    """
    y = np.asarray(y_test).astype(int)

    # Ensure we use the correct feature subset
    cols = fitted.feature_columns
    missing = [c for c in cols if c not in X_test.columns]
    if missing:
        raise ValueError(
            f"Missing columns in X_test for evaluation: {missing}. "
            "Ensure feature construction is consistent between train/test."
        )
    X_sub = X_test[cols]

    # Probabilities for the positive class
    p = predict_proba(fitted, X_sub)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1 - 1e-12)

    y_pred = (p >= threshold).astype(int)

    out: Dict[str, float] = {}
    out["n_samples"] = int(len(y))
    out["pos_rate"] = float(np.mean(y)) if len(y) else np.nan

    out["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    out["brier"] = float(brier_score_loss(y, p))

    out["accuracy"] = float(accuracy_score(y, y_pred))
    out["precision"] = float(precision_score(y, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y, y_pred, zero_division=0))

    # AUC metrics only if both classes present
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
        out["ap"] = float(average_precision_score(y, p))
    else:
        out["auc"] = np.nan
        out["ap"] = np.nan

    out["error"] = ""
    return out
