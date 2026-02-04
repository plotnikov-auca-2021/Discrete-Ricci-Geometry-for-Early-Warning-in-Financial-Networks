"""
Controls
--------
1) window_stride_days
   - Step between successive window endpoints (in trading days).
   - 1  → daily windows
   - 5  → weekly windows (every 5th trading day), etc.

2) max_years_of_data
   - Limit for how many *years* of data we use from the beginning of
     the available sample.
   - Example: max_years_of_data = 2:
       - If your index series starts at 2000-01-03, the pipeline will
         only use dates up to ~2002-01-03 for building windows.
   - Set to None to use the full sample.

3) n_workers
   - Number of parallel workers for heavy per-window computations.
   - If you use joblib:
       - 1  → no parallelism (single process)
       - 2  → two workers
       - -1 → all available cores.

The pipeline will:
  - Load and pre-process S&P 500 data (via ricci_ews.*)
  - Restrict the index to the first `max_years_of_data` years
  - Slide a rolling window with stride `window_stride_days`
  - Optionally parallelize per-window computations with `n_workers`
  - Output intermediate CSVs for each step.
"""

# pipeline/pipeline_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os


@dataclass
class PipelineConfig:
    # --- Data slicing ---
    warmup_years: int = 1
    analysis_years: int = 5  # you said you'll decide; keep default = 1

    # --- Rolling windows ---
    window_stride_days: int = 1  # stride=1 means daily endpoints

    # --- Parallelism ---
    n_workers: int = 2  # increase on desktop/cloud

    # --- Curvature controls ---
    compute_orc: bool = True  # OFF by default (FRC-only pipeline)

    # --- Label parameters (ALL adjustable) ---
    vol_top_pct: float = 0.03
    dd_tail_pcts: Tuple[float, ...] = (0.03, 0.05, 0.07)
    label_horizons: Tuple[int, ...] = (5, 10, 20)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    return default if v == "" else int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    return default if v == "" else float(v)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if v == "":
        return default
    return v in ("1", "true", "yes", "y", "on")


def _env_tuple_float(name: str, default: Tuple[float, ...]) -> Tuple[float, ...]:
    v = os.getenv(name, "").strip()
    if v == "":
        return default
    parts = [p.strip() for p in v.split(",") if p.strip() != ""]
    return tuple(float(p) for p in parts)


def _env_tuple_int(name: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    v = os.getenv(name, "").strip()
    if v == "":
        return default
    parts = [p.strip() for p in v.split(",") if p.strip() != ""]
    return tuple(int(p) for p in parts)


def get_pipeline_config() -> PipelineConfig:
    """
    Returns PipelineConfig with optional environment overrides.

    Example overrides:
      PIPE_WARMUP_YEARS=1
      PIPE_ANALYSIS_YEARS=3
      PIPE_STRIDE=5
      PIPE_NWORKERS=8
      PIPE_COMPUTE_ORC=1
      PIPE_VOL_TOP_PCT=0.05
      PIPE_DD_TAIL_PCTS=0.02,0.05,0.10
      PIPE_LABEL_HORIZONS=5,10,15
    """
    cfg = PipelineConfig(
        warmup_years=_env_int("PIPE_WARMUP_YEARS", 1),
        analysis_years=_env_int("PIPE_ANALYSIS_YEARS", 5),
        window_stride_days=_env_int("PIPE_STRIDE", 1),
        n_workers=_env_int("PIPE_NWORKERS", 2),
        compute_orc=_env_bool("PIPE_COMPUTE_ORC", True),
        vol_top_pct=_env_float("PIPE_VOL_TOP_PCT", 0.03),
        dd_tail_pcts=_env_tuple_float("PIPE_DD_TAIL_PCTS", (0.03, 0.05, 0.07)),
        label_horizons=_env_tuple_int("PIPE_LABEL_HORIZONS", (5, 10, 20)),
    )
    return cfg