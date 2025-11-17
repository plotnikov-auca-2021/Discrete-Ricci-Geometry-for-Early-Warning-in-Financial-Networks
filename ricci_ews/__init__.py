"""
ricci_ews: Early-warning signals on S&P 500 via discrete Ricci curvature.

This package contains all core modules:
- config: experiment configuration and hyperparameters
- data_io: loading and basic preprocessing of S&P 500 data
- universe: construction of the fixed asset universe
- returns: return computation and cleaning
- windows: rolling-window generation
- graphs_correlation / graphs_precision / backbones: graph construction
- curvature: Ollivier–Ricci and Forman–Ricci curvature
- features: feature engineering from curvature and graph statistics
- labels: construction of target variables from index series
- probes: mechanistic probes (diffusion, robustness, etc.)
- models: ML models and evaluation helpers
- pipeline: end-to-end walk-forward orchestration
- utils: small shared helpers

Most functions are written so that they can be unit-tested in isolation.
"""
