from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from arena_hearing.call import BEARING_WINDOW, BAND_HZ, FRAME, LOADING, SAMPLE_RATE_HZ, STATES, BlockStream, Detector, Onset, default_calibration, load_calibration
from arena_hearing.srp import band_bins, fft_length

FS = SAMPLE_RATE_HZ
NS_PER_SAMPLE = 62_500
MICS = np.asarray([[0.19, 0.135, 0.22], [0.19, -0.135, 0.22], [-0.19, 0.135, 0.22], [-0.19, -0.135, 0.22]])
SPEED_OF_SOUND = 343.0


def _arrival(signal: np.ndarray, azimuth: float) -> np.ndarray:
    """Far-field plane wave from azimuth on the four microphones, fractional delays in the frequency domain."""
    direction = np.asarray([math.cos(azimuth), math.sin(azimuth), 0.0])
    delays = -(MICS @ direction) / SPEED_OF_SOUND
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, 1.0 / FS)
    return np.stack([np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * d), signal.size) for d in delays], axis=1)


def _call(seconds: float, rng: np.random.Generator) -> np.ndarray:
    n = round(seconds * FS)
    t = np.arange(n) / FS
    voiced = sum(np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi)) / k for k, f0 in enumerate((220.0, 440.0, 660.0, 880.0, 1320.0, 1760.0), start=1))
    return voiced * (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t) ** 2)


def _run(detector: Detector, audio: np.ndarray, moving: bool, yaw: float = 0.0) -> list[Onset]:
    out = []
    for k in range(audio.shape[0] // FRAME):
        out.extend(detector.push(k * FRAME * NS_PER_SAMPLE, audio[k * FRAME : (k + 1) * FRAME], moving, yaw))
    return out


def _bearing_error_deg(heading: float, truth_deg: float) -> float:
    return abs((math.degrees(heading) - truth_deg + 180.0) % 360.0 - 180.0)


def _scene(azimuth_deg: float) -> np.ndarray:
    rng = np.random.default_rng(1)
    audio = rng.standard_normal((6 * FS, 4)) * 1e-3
    audio[3 * FS : 5 * FS] += _arrival(_call(2.0, rng) * 0.02, math.radians(azimuth_deg))
    return audio


def _identity_calibration(path: Path) -> Path:
    bins = fft_length(BEARING_WINDOW) // 2 + 1
    cov = np.broadcast_to(np.eye(4), (bins, 4, 4)).copy()
    np.savez(path, **{f"threshold__{state}": np.asarray(value) for state, value in (("moving", 7.5), ("stopped", 1.0))}, **{f"cov__{state}__{BEARING_WINDOW}": cov for state in STATES})
    return path


@pytest.mark.parametrize(("azimuth_deg", "yaw_deg"), [(60.0, 0.0), (-135.0, 0.0), (60.0, 90.0)])
def test_a_stopped_robot_detects_a_call_and_heads_for_it_in_the_map_frame(azimuth_deg: float, yaw_deg: float) -> None:
    detector = Detector(MICS, default_calibration())

    onsets = _run(detector, _scene(azimuth_deg), moving=False, yaw=math.radians(yaw_deg))

    assert len(onsets) >= 1
    first = onsets[0]
    assert abs(first.stamp_ns * 1e-9 - 3.0) < 0.2
    assert first.state == "stopped"
    assert first.windows >= 5
    assert first.peak_rms > 0.0
    assert _bearing_error_deg(first.azimuth_rad, azimuth_deg) < 5.0
    assert _bearing_error_deg(first.heading_rad, azimuth_deg + yaw_deg) < 5.0


def test_a_whitened_detector_keeps_the_bearing_of_a_call(tmp_path: Path) -> None:
    detector = Detector(MICS, load_calibration(_identity_calibration(tmp_path / "calib.npz")))

    onsets = _run(detector, _scene(60.0), moving=False)

    assert len(onsets) >= 1
    assert _bearing_error_deg(onsets[0].heading_rad, 60.0) < 5.0


def test_no_onset_on_noise_alone() -> None:
    rng = np.random.default_rng(2)
    detector = Detector(MICS, default_calibration())

    assert _run(detector, rng.standard_normal((8 * FS, 4)) * 1e-3, moving=False) == []


def test_the_motor_gate_mutes_a_moving_robot_after_a_drivetrain_jump() -> None:
    rng = np.random.default_rng(3)
    audio = rng.standard_normal((6 * FS, 4)) * 1e-3
    audio[3 * FS : round(4.2 * FS)] += _arrival(_call(1.2, rng) * 0.05, 0.5)
    gated = Detector(MICS, default_calibration())
    open_ = Detector(MICS, default_calibration())
    quiet = np.zeros((FRAME, 4))
    loud = np.full((FRAME, 4), 0.1)
    for k in range(audio.shape[0] // FRAME):
        stamp = k * FRAME * NS_PER_SAMPLE
        gated.push_motor(stamp, loud if k >= round(2.9 * FS / FRAME) else quiet)
        open_.push_motor(stamp, quiet)

    assert _run(gated, audio, moving=True) == []
    assert len(_run(open_, audio, moving=True)) >= 1


def test_calibration_file_gives_thresholds_and_a_whitener_per_state_on_the_band_bins(tmp_path: Path) -> None:
    calibration = load_calibration(_identity_calibration(tmp_path / "calib.npz"))

    assert calibration.thresholds == {"moving": 7.5, "stopped": 1.0}
    bins = band_bins(FS, fft_length(BEARING_WINDOW), *BAND_HZ)
    assert calibration.whiteners is not None
    for state in STATES:
        assert calibration.whiteners[state].shape == (len(bins), 4, 4)
        np.testing.assert_allclose(calibration.whiteners[state][0], np.eye(4) / math.sqrt(1.0 + LOADING), atol=1e-9)


def test_calibration_file_without_a_state_threshold_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "calib.npz"
    np.savez(path, threshold__moving=np.asarray(7.5))

    with pytest.raises(KeyError):
        load_calibration(path)


def test_block_stream_cuts_frames_of_any_length_into_stamped_blocks() -> None:
    stream = BlockStream(2)

    first = stream.push(0, np.ones((500, 2), dtype=np.float32))
    second = stream.push(500 * NS_PER_SAMPLE, np.ones((500, 2), dtype=np.float32))

    assert [stamp for stamp, _ in first] == [0]
    assert [stamp for stamp, _ in second] == [FRAME * NS_PER_SAMPLE, 2 * FRAME * NS_PER_SAMPLE]
    assert all(block.shape == (FRAME, 2) for _, block in first + second)
    assert stream.end_ns == 1000 * NS_PER_SAMPLE


def test_block_stream_fills_a_gap_between_frames_with_silence() -> None:
    stream = BlockStream(1)

    stream.push(0, np.ones((FRAME, 1), dtype=np.float32))
    blocks = stream.push(3 * FRAME * NS_PER_SAMPLE, np.ones((FRAME, 1), dtype=np.float32))

    assert [stamp for stamp, _ in blocks] == [FRAME * NS_PER_SAMPLE, 2 * FRAME * NS_PER_SAMPLE, 3 * FRAME * NS_PER_SAMPLE]
    assert [float(block.sum()) for _, block in blocks] == [0.0, 0.0, float(FRAME)]
