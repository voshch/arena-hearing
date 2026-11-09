"""Ego-noise whitener the call front-end applies before the phase transform."""

from __future__ import annotations

import numpy as np

DEFAULT_LOADING = 1e-3
EIGENVALUE_FLOOR = 1e-12


def whitener(cov: np.ndarray, loading: float = DEFAULT_LOADING) -> np.ndarray:
    """Per bin R^(-1/2) of the covariance with loading * mean eigenvalue added to its diagonal."""
    cov = np.asarray(cov, dtype=np.complex128)
    channels = cov.shape[-1]
    trace = np.einsum("fii->f", cov).real
    loaded = cov + (loading * trace / channels)[:, None, None] * np.eye(channels)
    values, vectors = np.linalg.eigh(loaded)
    return np.einsum("fij,fj,fkj->fik", vectors, 1.0 / np.sqrt(np.maximum(values, EIGENVALUE_FLOOR)), np.conj(vectors))
