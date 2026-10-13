"""SALSA-Lite features, the SELDnet graph and the multi-ACCDOA helpers on inputs with a known answer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from arena_hearing import dcase

FS = 16000
C = 343.0


def _sines(freqs_hz: tuple[float, ...], delays_s: tuple[float, ...], seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    return np.stack([sum(np.sin(2.0 * np.pi * f * (t - d)) for f in freqs_hz) for d in delays_s], axis=1) * 0.2


def test_params_derive_the_feature_sequence_from_the_label_hop() -> None:
    assert dcase.PARAMS["feature_sequence_length"] == 250
    assert dcase.PARAMS["t_pool_size"] == [5, 1, 1]


def test_salsalite_bins_follow_from_the_frequency_limits() -> None:
    feat = dcase.SalsaLiteFeatures()
    assert (feat.hop_len, feat.label_hop_len, feat.win_len, feat.nfft) == (320, 1600, 640, 1024)
    assert (feat.lower_bin, feat.upper_bin, feat.cutoff_bin) == (3, 128, 480)
    assert feat.nb_mel_bins == 477
    assert feat.delta == pytest.approx(2.0 * math.pi * FS / (1024 * C))
    assert feat.freq_vector.shape == (1, 513, 1)
    assert feat.freq_vector[0, 0, 0] == 1
    assert feat.freq_vector[0, 7, 0] == 7


def test_salsalite_lower_bin_never_reaches_dc() -> None:
    feat = dcase.SalsaLiteFeatures({**dcase.PARAMS, "fmin_doa_salsalite": 1})
    assert feat.lower_bin == 1


def test_salsalite_refuses_a_doa_band_above_the_spectrogram_cutoff() -> None:
    with pytest.raises(ValueError, match="exceeds spectrogram cutoff"):
        dcase.SalsaLiteFeatures({**dcase.PARAMS, "fmax_doa_salsalite": 4000, "fmax_spectra_salsalite": 3000})


def test_spectrogram_is_frames_by_bins_by_channels_truncated_to_the_requested_frames() -> None:
    pytest.importorskip("librosa")
    feat = dcase.SalsaLiteFeatures()
    audio = _sines((500.0,), (0.0, 0.0, 0.0, 0.0), 0.5)
    spect = feat.spectrogram(audio, 20)
    assert spect.shape == (20, 513, 4)
    peak = int(np.argmax(np.abs(spect[10, :, 0])))
    assert peak == 32


def test_salsalite_phase_is_the_negative_path_difference_to_the_reference_mic() -> None:
    pytest.importorskip("librosa")
    feat = dcase.SalsaLiteFeatures()
    path_m = (0.0, 0.02, -0.03, 0.05)
    bins = (32, 64, 96)
    audio = _sines(tuple(k * FS / feat.nfft for k in bins), tuple(d / C for d in path_m), 1.0)
    out = feat.salsalite(feat.spectrogram(audio, 50))
    nb = feat.nb_mel_bins
    assert out.shape == (50, 7 * nb)
    phase = out[10:40, 4 * nb :].reshape(30, 3, nb)
    for ch in range(3):
        for k in bins:
            np.testing.assert_allclose(phase[:, ch, k - feat.lower_bin], -path_m[ch + 1], atol=1e-4)
        assert not phase[:, ch, feat.upper_bin :].any()


def test_salsalite_power_is_the_db_spectrum_and_doubling_amplitude_adds_six_db() -> None:
    pytest.importorskip("librosa")
    feat = dcase.SalsaLiteFeatures()
    audio = _sines((1000.0,), (0.0, 0.0, 0.0, 0.0), 0.5)
    spect = feat.spectrogram(audio, 20)
    quiet = feat.salsalite(spect)
    loud = feat.salsalite(feat.spectrogram(2.0 * audio, 20))
    nb = feat.nb_mel_bins
    k = 64 - feat.lower_bin
    expected = 10.0 * np.log10(np.abs(spect[:, 64, 0]) ** 2)
    np.testing.assert_allclose(quiet[:, k], expected, rtol=1e-6)
    np.testing.assert_allclose(loud[:, k] - quiet[:, k], 20.0 * math.log10(2.0), atol=1e-6)
    np.testing.assert_allclose(loud[:, 4 * nb :], quiet[:, 4 * nb :], atol=1e-9)


def _small_params(nb_fnn_layers: int) -> dict:
    return {**dcase.PARAMS, "nb_cnn2d_filt": 8, "rnn_size": 16, "nb_heads": 2, "fnn_size": 12, "nb_fnn_layers": nb_fnn_layers, "nb_rnn_layers": 1, "dropout_rate": 0.0}


@pytest.mark.parametrize("nb_fnn_layers", [0, 2])
def test_build_model_pools_time_to_label_frames_and_bounds_the_output(nb_fnn_layers: int) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    params = _small_params(nb_fnn_layers)
    model = dcase.build_model((2, 7, 50, 64), (2, 10, 18), params).eval()
    assert len(model.fnn_list) == nb_fnn_layers + 1
    assert model.fnn_list[-1].out_features == 18
    assert model.gru.input_size == 8 * 2
    with torch.inference_mode():
        out = model(torch.randn(2, 7, 50, 64) * 3.0)
    assert tuple(out.shape) == (2, 10, 18)
    assert float(out.abs().max()) < 1.0


def _accdoa(tracks: list[list[tuple[float, float, float]]]) -> np.ndarray:
    """One frame, tracks[t][c] the (x, y, z) of class c on track t."""
    nb = len(tracks[0])
    row = np.zeros(9 * nb)
    for t, classes in enumerate(tracks):
        for c, (x, y, z) in enumerate(classes):
            row[3 * t * nb + c] = x
            row[3 * t * nb + nb + c] = y
            row[3 * t * nb + 2 * nb + c] = z
    return row[None, None, :]


def test_multi_accdoa_labels_split_tracks_and_threshold_the_vector_norm() -> None:
    accdoa = _accdoa([[(0.6, 0.0, 0.0), (0.3, 0.3, 0.0)], [(0.0, 0.0, 0.0), (0.0, 0.0, 0.51)], [(0.4, 0.3, 0.0), (0.0, -0.2, 0.0)]])
    sed0, doa0, sed1, doa1, sed2, doa2 = dcase.get_multi_accdoa_labels(accdoa, 2)
    assert sed0[0, 0].tolist() == [True, False]
    assert sed1[0, 0].tolist() == [False, True]
    assert sed2[0, 0].tolist() == [False, False]
    np.testing.assert_array_equal(doa0[0, 0], [0.6, 0.3, 0.0, 0.3, 0.0, 0.0])
    np.testing.assert_array_equal(doa2[0, 0], [0.4, 0.0, 0.3, -0.2, 0.0, 0.0])
    assert doa1.shape == (1, 1, 6)


@pytest.mark.parametrize(
    ("a", "b", "degrees"),
    [
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0),
        ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), 90.0),
        ((1.0, 0.0, 0.0), (-3.0, 0.0, 0.0), 180.0),
        ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), 45.0),
        ((0.0, 0.0, 0.5), (0.0, 0.5, 0.0), 90.0),
        ((0.2, 0.2, 0.2), (0.9, 0.9, 0.9), 0.0),
    ],
)
def test_distance_between_cartesian_coordinates_is_the_angle_in_degrees(a: tuple[float, ...], b: tuple[float, ...], degrees: float) -> None:
    assert dcase.distance_between_cartesian_coordinates(*a, *b) == pytest.approx(degrees, abs=1e-2)


def _doa(nb: int, cls: int, xyz: tuple[float, float, float]) -> np.ndarray:
    row = np.zeros(3 * nb)
    row[cls], row[cls + nb], row[cls + 2 * nb] = xyz
    return row


@pytest.mark.parametrize(
    ("sed0", "sed1", "deg", "similar"),
    [(True, True, 10.0, 1), (True, True, 20.0, 0), (True, False, 0.0, 0), (False, True, 0.0, 0), (False, False, 0.0, 0)],
)
def test_determine_similar_location_needs_both_active_and_within_the_unify_angle(sed0: bool, sed1: bool, deg: float, similar: int) -> None:
    rad = math.radians(deg)
    a = _doa(2, 1, (1.0, 0.0, 0.0))
    b = _doa(2, 1, (math.cos(rad), math.sin(rad), 0.0))
    assert dcase.determine_similar_location(sed0, sed1, a, b, 1, 15.0, 2) == similar
