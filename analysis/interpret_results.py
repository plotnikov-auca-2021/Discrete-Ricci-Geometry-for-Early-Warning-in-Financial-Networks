# analysis/interpret_results.py
"""
Post-pipeline analysis for Discrete Ricci Geometry Early-Warning project.

Expected inputs (in project_root/outputs):
  - model_metrics.csv
  - model_predictions.csv
  - features_and_labels.csv
  - mechanistic_probes.csv   (optional but recommended)

Outputs (in project_root/analysis):
  - tables (CSV)
  - plots (PNG)
  - subfolders: pr_curves/, capture_curves/, timeseries/, mechanistic_probes/
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, average_precision_score


# -----------------------------
# Project root + fixed I/O dirs
# -----------------------------

def project_root() -> Path:
    """
    Resolve project root robustly regardless of current working directory.
    This file lives at: <root>/analysis/interpret_results.py
    So root is parents[1].
    """
    return Path(__file__).resolve().parents[1]


# -----------------------------
# Helpers: parsing + IO
# -----------------------------

VOL_RE = re.compile(r"^y_vol_top(?P<pct>\d+)_h(?P<h>\d+)$", re.IGNORECASE)
DD_RE = re.compile(r"^y_dd_h(?P<h>\d+)_thr(?P<thr>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class LabelInfo:
    label_name: str
    event_type: str  # "vol" or "dd"
    horizon_days: int
    vol_top_pct_int: Optional[int] = None
    dd_thr_int: Optional[int] = None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates or [])


def parse_label_info(label_name: str) -> Optional[LabelInfo]:
    m = VOL_RE.match(label_name)
    if m:
        return LabelInfo(
            label_name=label_name,
            event_type="vol",
            horizon_days=int(m.group("h")),
            vol_top_pct_int=int(m.group("pct")),
            dd_thr_int=None,
        )
    m = DD_RE.match(label_name)
    if m:
        return LabelInfo(
            label_name=label_name,
            event_type="dd",
            horizon_days=int(m.group("h")),
            vol_top_pct_int=None,
            dd_thr_int=int(m.group("thr")),
        )
    return None


def clamp_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1.0 - eps)


# -----------------------------
# Human-readable naming helpers
# -----------------------------

MODEL_NAME_MAP: Dict[str, str] = {
    "logit_baseline": "Logistic Regression (Baseline features)",
    "logit_full": "Logistic Regression (Baseline + Ricci curvature)",
    "rf_full": "Random Forest (Baseline + Ricci curvature)",
}


def pretty_model_name(name: str) -> str:
    s = str(name)
    return MODEL_NAME_MAP.get(s, s)


def pretty_event_type(event_type: str) -> str:
    if event_type == "vol":
        return "High Volatility Event"
    if event_type == "dd":
        return "Drawdown Event"
    return str(event_type)


def pretty_label_name(label_name: str) -> str:
    """
    Convert internal label keys to a human-readable description.
    Examples:
      y_vol_top3_h5   -> "High volatility event (top 3% of rolling volatility), horizon = 5 trading days"
      y_dd_h10_thr7   -> "Drawdown event (threshold = 7%), horizon = 10 trading days"
    """
    li = parse_label_info(str(label_name))
    if not li:
        return str(label_name)

    if li.event_type == "vol":
        return (
            f"High volatility event (top {li.vol_top_pct_int}% of rolling volatility), "
            f"prediction horizon = {li.horizon_days} days"
        )
    return (
        f"Drawdown event (drawdown threshold = {li.dd_thr_int}%), "
        f"prediction horizon = {li.horizon_days} days"
    )


def label_short_id(label_name: str) -> str:
    """
    Short, still readable id for filenames and compact axes.
    """
    li = parse_label_info(str(label_name))
    if not li:
        return str(label_name)
    if li.event_type == "vol":
        return f"Volatility top {li.vol_top_pct_int}%, horizon {li.horizon_days}d"
    return f"Drawdown ≥{li.dd_thr_int}%, horizon {li.horizon_days}d"

def label_shorter_id(label_name: str) -> str:
    """
    Even shortert, still readable id for filenames and compact axes.
    """
    li = parse_label_info(str(label_name))
    if not li:
        return str(label_name)
    if li.event_type == "vol":
        return f"Vol. {li.vol_top_pct_int}%, h {li.horizon_days}d"
    return f"Dd ≥{li.dd_thr_int}%, h {li.horizon_days}d"


def pretty_metric_name(metric_col: str) -> str:
    m = str(metric_col)
    if m in {"auc", "roc_auc"}:
        return "ROC AUC"
    if m in {"ap", "pr_auc"}:
        return "Average Precision (PR AUC)"
    if m in {"log_loss", "logloss"}:
        return "Log Loss"
    if m == "brier":
        return "Brier Score"
    return m


def pretty_probe_name(col: str) -> str:
    """
    Best-effort probe name prettifier.
    Assumes probes columns are numeric and may include corr_/prec_ prefixes.
    """
    s = str(col)

    # Graph type prefix
    graph_prefix = ""
    if s.startswith("corr_"):
        graph_prefix = "Correlation graph"
        s = s[len("corr_"):]
    elif s.startswith("prec_"):
        graph_prefix = "Precision graph"
        s = s[len("prec_"):]

    # Replace common tokens
    replacements = {
        "mfpt": "Mean First Passage Time",
        "commute": "Commute Time",
        "mixing": "Mixing Time",
        "diffusion": "Diffusion",
        "spectral_gap": "Spectral Gap",
        "gap": "Gap",
        "laplacian": "Laplacian",
        "diameter": "Graph Diameter",
        "avg": "Average",
        "mean": "Mean",
        "median": "Median",
        "std": "Standard Deviation",
        "var": "Variance",
    }

    # token-based transform
    toks = s.replace("__", "_").split("_")
    toks2 = []
    for t in toks:
        t_low = t.lower()
        toks2.append(replacements.get(t_low, t))
    name = " ".join(toks2).strip()

    # Title-case but keep acronyms readable
    name = " ".join([w if w.isupper() else w.capitalize() for w in name.split()])

    if graph_prefix:
        return f"{graph_prefix}: {name}"
    return name


def human_threshold_axis(event_type: str, value: float) -> str:
    """
    For heatmap x-axis tick labels.
    """
    if pd.isna(value):
        return ""
    if event_type == "vol":
        return f"Top {int(value)}%"
    return f"{int(value)}%"


# -----------------------------
# Tables: Best models + deltas
# -----------------------------

def best_models_table(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()

    # normalize AUC/AP naming
    if "roc_auc" in df.columns and "auc" not in df.columns:
        df["auc"] = df["roc_auc"]
    if "pr_auc" in df.columns and "ap" not in df.columns:
        df["ap"] = df["pr_auc"]
    if "auc" not in df.columns:
        df["auc"] = np.nan
    if "ap" not in df.columns:
        df["ap"] = np.nan

    df["rank_score"] = df["auc"]

    rows = []
    for label, g in df.groupby("label_name", dropna=False):
        g2 = g.copy()
        if g2["rank_score"].isna().all():
            g2["rank_score"] = g2["ap"]
        g2 = g2.sort_values(["rank_score", "ap"], ascending=False, na_position="last")
        rows.append(g2.iloc[0])

    out = pd.DataFrame(rows)

    infos = [parse_label_info(x) for x in out["label_name"].astype(str)]
    out["event_type_parsed"] = [i.event_type if i else "" for i in infos]
    out["horizon_days_parsed"] = [i.horizon_days if i else np.nan for i in infos]
    out["vol_top_pct_int_parsed"] = [i.vol_top_pct_int if i else np.nan for i in infos]
    out["dd_thr_int_parsed"] = [i.dd_thr_int if i else np.nan for i in infos]

    # Human-readable helpers
    out["label_readable"] = out["label_name"].astype(str).map(pretty_label_name)
    out["model_readable"] = out["model_name"].astype(str).map(pretty_model_name)
    out["event_type_readable"] = out["event_type_parsed"].astype(str).map(pretty_event_type)

    keep = [
        "label_name", "label_readable",
        "event_type_parsed", "event_type_readable",
        "horizon_days_parsed", "vol_top_pct_int_parsed", "dd_thr_int_parsed",
        "model_name", "model_readable", "fit_status",
        "n_train", "n_test",
        "pos_rate", "auc", "ap", "precision", "recall", "f1",
        "log_loss", "logloss", "brier", "error",
        "split_mode", "split_cutoff_date", "total_pos", "train_pos", "test_pos",
        "y_train_unique", "y_test_unique",
    ]
    cols = [c for c in keep if c in out.columns]
    out = out[cols].sort_values(
        ["event_type_parsed", "vol_top_pct_int_parsed", "dd_thr_int_parsed", "horizon_days_parsed", "auc"],
        ascending=[True, True, True, True, False],
        na_position="last",
    ).reset_index(drop=True)
    return out


def delta_auc_table(metrics: pd.DataFrame,
                    model_a: str = "logit_full",
                    model_b: str = "logit_baseline") -> pd.DataFrame:
    df = metrics.copy()
    if "roc_auc" in df.columns and "auc" not in df.columns:
        df["auc"] = df["roc_auc"]
    if "auc" not in df.columns:
        df["auc"] = np.nan

    pivot = df.pivot_table(index="label_name", columns="model_name", values="auc", aggfunc="mean")
    if model_a not in pivot.columns:
        pivot[model_a] = np.nan
    if model_b not in pivot.columns:
        pivot[model_b] = np.nan

    out = pd.DataFrame({
        "label_name": pivot.index,
        f"auc_{model_a}": pivot[model_a].values,
        f"auc_{model_b}": pivot[model_b].values,
        "delta_auc": (pivot[model_a] - pivot[model_b]).values,
    }).reset_index(drop=True)

    infos = [parse_label_info(x) for x in out["label_name"].astype(str)]
    out["event_type"] = [i.event_type if i else "" for i in infos]
    out["horizon_days"] = [i.horizon_days if i else np.nan for i in infos]
    out["vol_top_pct_int"] = [i.vol_top_pct_int if i else np.nan for i in infos]
    out["dd_thr_int"] = [i.dd_thr_int if i else np.nan for i in infos]

    out["label_readable"] = out["label_name"].astype(str).map(pretty_label_name)
    out["model_a_readable"] = pretty_model_name(model_a)
    out["model_b_readable"] = pretty_model_name(model_b)

    out = out.sort_values(
        ["event_type", "vol_top_pct_int", "dd_thr_int", "horizon_days", "delta_auc"],
        ascending=[True, True, True, True, False],
        na_position="last",
    ).reset_index(drop=True)
    return out


# -----------------------------
# Plots: heatmaps + deltas
# -----------------------------

def heatmap_auc(metrics: pd.DataFrame, out_dir: Path, event_type: str) -> None:
    df = metrics.copy()
    if "roc_auc" in df.columns and "auc" not in df.columns:
        df["auc"] = df["roc_auc"]
    if "auc" not in df.columns:
        df["auc"] = np.nan

    df = df[df["label_name"].notna()].copy()
    infos = df["label_name"].astype(str).map(parse_label_info)
    df["event_type_parsed"] = infos.map(lambda x: x.event_type if x else None)
    df["horizon"] = infos.map(lambda x: x.horizon_days if x else np.nan)
    df["top_pct_int"] = infos.map(lambda x: x.vol_top_pct_int if x else np.nan)
    df["thr_int"] = infos.map(lambda x: x.dd_thr_int if x else np.nan)

    df = df[df["event_type_parsed"] == event_type].copy()
    if df.empty:
        print(f"[warn] No rows for event_type={event_type} in metrics; skipping heatmaps.")
        return

    key_col = "top_pct_int" if event_type == "vol" else "thr_int"
    models = sorted(df["model_name"].dropna().unique().tolist())

    for m in models:
        sub = df[df["model_name"] == m].copy()
        pv = sub.pivot_table(index="horizon", columns=key_col, values="auc", aggfunc="mean")
        pv = pv.sort_index(axis=0).sort_index(axis=1)

        fig = plt.figure(figsize=(10, 4.5))
        ax = plt.gca()
        im = ax.imshow(pv.values, aspect="auto")

        title_event = pretty_event_type(event_type)
        title_model = pretty_model_name(m)
        ax.set_title(f"Model performance (ROC AUC)\n{title_event} • {title_model}")

        ax.set_xlabel("Event definition")
        ax.set_ylabel("Prediction horizon (days)")

        ax.set_xticks(range(pv.shape[1]))
        ax.set_xticklabels([human_threshold_axis(event_type, x) for x in pv.columns], rotation=0)

        ax.set_yticks(range(pv.shape[0]))
        ax.set_yticklabels([str(int(x)) if pd.notna(x) else "" for x in pv.index])

        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                val = pv.values[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("ROC AUC", rotation=90)

        fig.tight_layout()
        fname = f"auc_heatmap__{event_type}__{m}.png"
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)


def barplot_delta_auc(delta_df: pd.DataFrame, out_dir: Path,
                      model_a: str = "logit_full",
                      model_b: str = "logit_baseline") -> None:
    df = delta_df.copy()
    df = df[df["delta_auc"].notna()].copy()
    if df.empty:
        print("[warn] No delta_auc values; skipping delta plot.")
        return

    df["label_short"] = df["label_name"].astype(str).map(label_shorter_id)

    title_a = pretty_model_name(model_a)
    title_b = pretty_model_name(model_b)

    for et in ["vol", "dd"]:
        sub = df[df["event_type"] == et].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("delta_auc", ascending=False)

        fig = plt.figure(figsize=(12, max(4, 0.35 * len(sub))))
        ax = plt.gca()
        ax.barh(sub["label_short"], sub["delta_auc"])
        ax.axvline(0.0)

        ax.set_title(
            "Incremental value of Ricci curvature features\n"
            f"Δ(ROC AUC) = {title_a} − {title_b}\n"
            f"Event type: {pretty_event_type(et)}"
        )
        ax.set_xlabel("Δ ROC AUC (positive = curvature improves performance)")
        ax.set_ylabel("Event label")

        fig.tight_layout()
        fig.savefig(out_dir / f"delta_auc_bar__{et}.png", dpi=150)
        plt.close(fig)


# -----------------------------
# PR curves + Capture curves
# -----------------------------

def pr_curves_overlay(preds: pd.DataFrame, out_dir: Path, label_name: str) -> None:
    sub = preds[preds["label_name"] == label_name].copy()
    sub = sub.dropna(subset=["y_true", "p_hat", "model_name", "date"])
    if sub.empty:
        return

    y = sub["y_true"].astype(int).values
    if len(np.unique(y)) < 2:
        return

    fig = plt.figure(figsize=(7.5, 5.5))
    ax = plt.gca()

    for model_name, g in sub.groupby("model_name"):
        g2 = g.sort_values("date")
        y_true = g2["y_true"].astype(int).values
        p = clamp_probs(g2["p_hat"].values)
        if len(np.unique(y_true)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true, p)
        ap = average_precision_score(y_true, p)
        ax.plot(rec, prec, label=f"{pretty_model_name(model_name)} (AP={ap:.3f})")

    ax.set_title(f"Precision–Recall Curves\n{pretty_label_name(label_name)}")
    ax.set_xlabel("Recall (fraction of events correctly identified)")
    ax.set_ylabel("Precision (fraction of predicted events that are true events)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    fig.savefig(out_dir / f"pr_curve__{label_name}.png", dpi=150)
    plt.close(fig)


def capture_curve(preds: pd.DataFrame, out_dir: Path, label_name: str,
                  k_percents: Optional[List[int]] = None) -> None:
    if k_percents is None:
        k_percents = [1, 2, 5, 10, 15, 20, 30, 40, 50]

    sub = preds[preds["label_name"] == label_name].copy()
    sub = sub.dropna(subset=["y_true", "p_hat", "model_name"])
    if sub.empty:
        return

    fig = plt.figure(figsize=(7.5, 5.5))
    ax = plt.gca()

    for model_name, g in sub.groupby("model_name"):
        g2 = g.sort_values("p_hat", ascending=False)
        y = g2["y_true"].astype(int).values
        total_pos = int(y.sum())
        if total_pos == 0:
            continue

        recalls = []
        xs = []
        n = len(g2)
        for k in k_percents:
            k_n = max(1, int(np.ceil(n * (k / 100.0))))
            y_top = y[:k_n]
            recall = float(y_top.sum()) / float(total_pos)
            xs.append(k)
            recalls.append(recall)

        ax.plot(xs, recalls, marker="o", label=pretty_model_name(model_name))

    ax.set_title(f"Event Capture Curve\n{pretty_label_name(label_name)}")
    ax.set_xlabel("Top k% of days ranked by predicted risk (p̂)")
    ax.set_ylabel("Recall (fraction of all events captured within top k% days)")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    fig.savefig(out_dir / f"capture_curve__{label_name}.png", dpi=150)
    plt.close(fig)


# -----------------------------
# Time-series plots: p_hat + events
# -----------------------------

def timeseries_risk_plot(preds: pd.DataFrame, out_dir: Path, label_name: str,
                         max_models: int = 5) -> None:
    sub = preds[preds["label_name"] == label_name].copy()
    sub = sub.dropna(subset=["date", "y_true", "p_hat", "model_name"])
    if sub.empty:
        return

    models = sorted(sub["model_name"].unique().tolist())[:max_models]
    for model_name in models:
        g = sub[sub["model_name"] == model_name].sort_values("date")
        dates = pd.to_datetime(g["date"])
        p = clamp_probs(g["p_hat"].values)
        y = g["y_true"].astype(int).values

        fig = plt.figure(figsize=(12.5, 4.3))
        ax = plt.gca()
        ax.plot(dates, p, label="Predicted risk score (p̂)")

        event_dates = dates[y == 1]
        if len(event_dates) > 0:
            ax.scatter(
                event_dates,
                np.full(len(event_dates), 1.0),
                marker="x",
                label="Observed event date",
            )

        ax.set_title(
            "Predicted risk through time\n"
            f"{pretty_label_name(label_name)}\n"
            f"Model: {pretty_model_name(model_name)}"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted risk score (p̂)")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        fig.savefig(out_dir / f"timeseries__{label_name}__{model_name}.png", dpi=150)
        plt.close(fig)


# -----------------------------
# Mechanistic probes: boxplots + scatter
# -----------------------------

def mechanistic_probe_columns(probes: pd.DataFrame) -> List[str]:
    if probes is None or probes.empty:
        return []
    meta = {"date", "window_start", "window_end", "n_days", "n_assets"}
    cols = []
    for c in probes.columns:
        if c in meta:
            continue
        if pd.api.types.is_numeric_dtype(probes[c]):
            cols.append(c)
    return cols


def probe_boxplots(merged: pd.DataFrame, out_dir: Path,
                   label_name: str, model_name: str,
                   probe_cols: List[str], max_probes: int = 12) -> None:
    if merged.empty:
        return
    probe_cols = probe_cols[:max_probes]

    y = merged["y_true"].astype(int).values
    if len(np.unique(y)) < 2:
        return

    for c in probe_cols:
        if c not in merged.columns:
            continue
        vals0 = merged.loc[merged["y_true"] == 0, c].dropna().values
        vals1 = merged.loc[merged["y_true"] == 1, c].dropna().values
        if len(vals0) < 5 or len(vals1) < 2:
            continue

        fig = plt.figure(figsize=(7.0, 4.5))
        ax = plt.gca()
        ax.boxplot([vals0, vals1], tick_labels=["Non-event days", "Event days"])
        ax.set_title(
            "Mechanistic probe distribution\n"
            f"Probe: {pretty_probe_name(c)}\n"
            f"{pretty_label_name(label_name)}\n"
            f"Model: {pretty_model_name(model_name)}"
        )
        ax.set_ylabel(pretty_probe_name(c))
        fig.tight_layout()
        fig.savefig(out_dir / f"probe_box__{c}.png", dpi=150)
        plt.close(fig)


def probe_scatter_vs_risk(merged: pd.DataFrame, out_dir: Path,
                          label_name: str, model_name: str,
                          probe_cols: List[str], max_probes: int = 12) -> None:
    if merged.empty:
        return
    probe_cols = probe_cols[:max_probes]

    for c in probe_cols:
        if c not in merged.columns:
            continue
        x = merged[c].astype(float).values
        y = merged["p_hat"].astype(float).values
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 30:
            continue

        fig = plt.figure(figsize=(7.0, 4.5))
        ax = plt.gca()
        ax.scatter(x[m], y[m], s=10)
        ax.set_title(
            "Mechanistic probe vs predicted risk\n"
            f"Probe: {pretty_probe_name(c)}\n"
            f"{pretty_label_name(label_name)}\n"
            f"Model: {pretty_model_name(model_name)}"
        )
        ax.set_xlabel(pretty_probe_name(c))
        ax.set_ylabel("Predicted risk score (p̂)")
        fig.tight_layout()
        fig.savefig(out_dir / f"probe_scatter__{c}.png", dpi=150)
        plt.close(fig)


# -----------------------------
# Main analysis orchestration
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-labels", type=int, default=50, help="Limit number of labels to plot")
    parser.add_argument("--max-probes", type=int, default=12, help="Max probe columns to plot per label/model")
    args = parser.parse_args()

    root = project_root()
    in_dir = root / "outputs"
    out_dir = root / "analysis"
    ensure_dir(out_dir)

    metrics_path = in_dir / "model_metrics.csv"
    preds_path = in_dir / "model_predictions.csv"
    feats_path = in_dir / "features_and_labels.csv"
    probes_path = in_dir / "mechanistic_probes.csv"

    metrics = safe_read_csv(metrics_path)
    preds = safe_read_csv(preds_path, parse_dates=["date"])
    _ = safe_read_csv(feats_path, parse_dates=["date"])  # loaded for completeness

    # explicit probes loading (no DataFrame truthiness)
    if probes_path.exists():
        probes = safe_read_csv(probes_path, parse_dates=["date"])
    else:
        probes = pd.DataFrame()

    # normalize naming
    if "roc_auc" in metrics.columns and "auc" not in metrics.columns:
        metrics["auc"] = metrics["roc_auc"]
    if "pr_auc" in metrics.columns and "ap" not in metrics.columns:
        metrics["ap"] = metrics["pr_auc"]
    if "auc" not in metrics.columns:
        metrics["auc"] = np.nan
    if "ap" not in metrics.columns:
        metrics["ap"] = np.nan

    # 1) Best model tables
    best_df = best_models_table(metrics)
    best_df.to_csv(out_dir / "best_models_by_label.csv", index=False)

    delta_df = delta_auc_table(metrics, model_a="logit_full", model_b="logit_baseline")
    delta_df.to_csv(out_dir / "delta_auc_logit_full_minus_baseline.csv", index=False)

    compact_auc = metrics.pivot_table(index="label_name", columns="model_name", values="auc", aggfunc="mean")
    compact_ap = metrics.pivot_table(index="label_name", columns="model_name", values="ap", aggfunc="mean")
    compact_auc.to_csv(out_dir / "auc_pivot.csv")
    compact_ap.to_csv(out_dir / "ap_pivot.csv")

    # Add a readable version too
    metrics_readable = metrics.copy()
    if "label_name" in metrics_readable.columns:
        metrics_readable["label_readable"] = metrics_readable["label_name"].astype(str).map(pretty_label_name)
    if "model_name" in metrics_readable.columns:
        metrics_readable["model_readable"] = metrics_readable["model_name"].astype(str).map(pretty_model_name)
    metrics_readable.to_csv(out_dir / "model_metrics_with_readable_names.csv", index=False)

    # 2) Heatmaps (AUC)
    heatmap_auc(metrics, out_dir, event_type="dd")
    heatmap_auc(metrics, out_dir, event_type="vol")

    # 3) Bar plot: delta AUC (curvature contribution)
    barplot_delta_auc(delta_df, out_dir, model_a="logit_full", model_b="logit_baseline")

    # 4) PR curves + capture curves + timeseries
    labels = sorted(preds["label_name"].dropna().unique().tolist())
    labels = labels[: max(1, min(len(labels), args.max_labels))]

    pr_dir = out_dir / "pr_curves"
    cap_dir = out_dir / "capture_curves"
    ts_dir = out_dir / "timeseries"
    ensure_dir(pr_dir)
    ensure_dir(cap_dir)
    ensure_dir(ts_dir)

    for lbl in labels:
        pr_curves_overlay(preds, pr_dir, lbl)
        capture_curve(preds, cap_dir, lbl, k_percents=[1, 2, 5, 10, 15, 20, 30, 40, 50])
        timeseries_risk_plot(preds, ts_dir, lbl, max_models=5)

    # 5) Mechanistic probe visuals
    if probes is not None and not probes.empty:
        probe_dir = out_dir / "mechanistic_probes"
        ensure_dir(probe_dir)

        probes2 = probes.copy()
        probes2["date"] = pd.to_datetime(probes2["date"])
        probe_cols = mechanistic_probe_columns(probes2)

        best_map = {str(r["label_name"]): str(r["model_name"]) for _, r in best_df.iterrows()}

        for lbl in labels:
            if lbl not in best_map:
                continue
            model_name = best_map[lbl]

            sub_pred = preds[(preds["label_name"] == lbl) & (preds["model_name"] == model_name)].copy()
            sub_pred = sub_pred.dropna(subset=["date", "y_true", "p_hat"])
            if sub_pred.empty:
                continue

            sub_pred["date"] = pd.to_datetime(sub_pred["date"])
            merged = pd.merge(
                sub_pred[["date", "y_true", "p_hat"]],
                probes2[["date"] + probe_cols],
                on="date",
                how="inner",
            )
            if merged.empty:
                continue

            # subfolder name stays stable (machine-friendly)
            sub_dir = probe_dir / f"{lbl}__{model_name}"
            ensure_dir(sub_dir)

            probe_boxplots(merged, sub_dir, lbl, model_name, probe_cols, max_probes=args.max_probes)
            probe_scatter_vs_risk(merged, sub_dir, lbl, model_name, probe_cols, max_probes=args.max_probes)

        # Optional probe correlation table
        rows = []
        for lbl in labels:
            if lbl not in best_map:
                continue
            model_name = best_map[lbl]
            sub_pred = preds[(preds["label_name"] == lbl) & (preds["model_name"] == model_name)].copy()
            sub_pred = sub_pred.dropna(subset=["date", "y_true", "p_hat"])
            if sub_pred.empty:
                continue
            sub_pred["date"] = pd.to_datetime(sub_pred["date"])
            merged = pd.merge(sub_pred[["date", "p_hat"]], probes2[["date"] + probe_cols], on="date", how="inner")
            if merged.empty:
                continue

            for c in probe_cols:
                x = merged[c].astype(float).values
                yv = merged["p_hat"].astype(float).values
                m = np.isfinite(x) & np.isfinite(yv)
                if m.sum() < 30:
                    continue
                corr = np.corrcoef(x[m], yv[m])[0, 1]
                rows.append({
                    "label_name": lbl,
                    "label_readable": pretty_label_name(lbl),
                    "model_name": model_name,
                    "model_readable": pretty_model_name(model_name),
                    "probe": c,
                    "probe_readable": pretty_probe_name(c),
                    "corr_probe_vs_p_hat": float(corr),
                })

        if rows:
            pd.DataFrame(rows).sort_values(
                ["label_name", "corr_probe_vs_p_hat"],
                ascending=[True, False]
            ).to_csv(out_dir / "probe_vs_risk_correlations.csv", index=False)

    print(f"[analysis] input:  {in_dir}")
    print(f"[analysis] output: {out_dir}")
    print("[analysis] done.")


if __name__ == "__main__":
    main()
