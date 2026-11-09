"""Steered response power of a planar far-field array with the phase transform and an optional per-bin whitener."""

from __future__ import annotations

import functools

import numpy as np

SPEED_OF_SOUND_MPS = 343.0
DECIMATION = 8
AZIMUTHS = 720


def fft_length(window_samples: int) -> int:
    return 1 << int(np.ceil(np.log2(window_samples)))


def band_bins(sample_rate_hz: int, n_fft: int, low_hz: float, high_hz: float) -> np.ndarray:
    """rfft bins strictly inside the band, trimmed to a multiple of DECIMATION."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate_hz)
    bins = np.flatnonzero((freqs > low_hz) & (freqs < high_hz))
    return bins[: len(bins) // DECIMATION * DECIMATION]


@functools.lru_cache(maxsize=8)
def _steering(mics: bytes, sample_rate_hz: int, n_fft: int, low_hz: float, high_hz: float) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    positions = np.frombuffer(mics, dtype=np.float64).reshape(-1, 3)
    bins = band_bins(sample_rate_hz, n_fft, low_hz, high_hz)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate_hz)[bins].reshape(-1, DECIMATION).mean(axis=1)
    azimuths = np.linspace(-np.pi, np.pi, AZIMUTHS, endpoint=False)
    directions = np.stack([np.cos(azimuths), np.sin(azimuths), np.zeros_like(azimuths)], axis=1)
    delays = -(directions @ positions.T) / SPEED_OF_SOUND_MPS
    steering = [np.exp(2j * np.pi * np.outer(delays[:, i] - delays[:, j], freqs)) for i in range(len(positions)) for j in range(i + 1, len(positions))]
    return bins, azimuths, steering


def srp_map(segment: np.ndarray, sample_rate_hz: int, mics: np.ndarray, low_hz: float, high_hz: float, whitener: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The azimuth grid and the steered response power over it, for (samples, channels) audio and a (band bins, channels, channels) whitener."""
    n_fft = fft_length(len(segment))
    spectrum = np.fft.rfft(segment * np.hanning(len(segment))[:, None], n_fft, axis=0)
    bins, azimuths, steering = _steering(np.ascontiguousarray(mics, dtype=np.float64).tobytes(), sample_rate_hz, n_fft, low_hz, high_hz)
    spectrum = spectrum[bins]
    if whitener is not None:
        spectrum = np.einsum("fij,fj->fi", whitener, spectrum)
    unit = spectrum / (np.abs(spectrum) + 1e-12)
    power = np.zeros(AZIMUTHS)
    pair = 0
    for i in range(segment.shape[1]):
        for j in range(i + 1, segment.shape[1]):
            cross = (unit[:, i] * np.conj(unit[:, j])).reshape(-1, DECIMATION).mean(axis=1)
            power += np.real(steering[pair] @ cross)
            pair += 1
    return azimuths, power
