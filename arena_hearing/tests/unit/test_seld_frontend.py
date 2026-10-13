"""SeldFrontend and SeldStream on a seeded checkpoint of the real graph, decode on hand-built ACCDOA rows."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from arena_hearing import dcase
from arena_hearing.seld import FILLER, Detection, SeldStream, load_wav

CLASSES = ("footstep", "speech")
NB_BINS = 477
NB_CH = 7


@pytest.fixture(scope="module")
def model_files(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    torch = pytest.importorskip("torch")
    pytest.importorskip("librosa")
    root = tmp_path_factory.mktemp("seld")
    torch.manual_seed(7)
    model = dcase.build_model((1, NB_CH, 250, NB_BINS), (1, 50, 18))
    checkpoint = root / "model.h5"
    torch.save(model.state_dict(), checkpoint)
    np.savez(root / "scaler.npz", mean=np.zeros(NB_CH * NB_BINS), scale=np.ones(NB_CH * NB_BINS))
    np.savez(root / "shifted.npz", mean=np.ones(NB_CH * NB_BINS), scale=np.full(NB_CH * NB_BINS, 2.0))
    np.savez(root / "short.npz", mean=np.zeros(10), scale=np.ones(10))
    return checkpoint, root


def _frontend(model_files: tuple[Path, Path], *, det_threshold: float = 0.5, scaler: str = "scaler.npz", classes: tuple[str, ...] = CLASSES, torch_threads: int = 0):
    from arena_hearing.seld import SeldFrontend

    checkpoint, root = model_files
    return SeldFrontend(checkpoint, root / scaler, classes, device="cpu", det_threshold=det_threshold, torch_threads=torch_threads)


@pytest.fixture(scope="module")
def frontend(model_files: tuple[Path, Path]):
    return _frontend(model_files, torch_threads=2)


def _vec(az_deg: float, norm: float = 0.8, el_deg: float = 0.0) -> tuple[float, float, float]:
    az, el = math.radians(az_deg), math.radians(el_deg)
    return norm * math.cos(el) * math.cos(az), norm * math.cos(el) * math.sin(az), norm * math.sin(el)


def _row(tracks: list[tuple[float, float, float] | None], cls: int = 0) -> np.ndarray:
    """One ACCDOA row with class ``cls`` on each track at the given vector, every other entry zero."""
    row = np.zeros(18)
    for t, xyz in enumerate(tracks):
        if xyz is None:
            continue
        for axis in range(3):
            row[6 * t + 2 * axis + cls] = xyz[axis]
    return row


def _azimuths(dets: list[Detection]) -> list[float]:
    return [round(math.degrees(d.azimuth_rad), 6) for d in dets]


def test_frontend_derives_window_and_channel_counts(frontend) -> None:
    assert frontend.nb_classes == 2
    assert frontend.classes == CLASSES
    assert (frontend.nb_raw_ch, frontend.nb_ch, frontend.nb_mel_bins) == (4, 7, NB_BINS)
    assert (frontend.fs, frontend.hop_len, frontend.label_hop_len) == (16000, 320, 1600)
    assert (frontend.feature_sequence_length, frontend.label_sequence_length) == (250, 50)
    assert frontend.window_samples == 80000
    assert frontend.thresh_unify == 15.0
    assert not frontend.model.training


def test_frontend_refuses_a_class_list_of_the_wrong_length(model_files) -> None:
    with pytest.raises(ValueError, match="the model has 2 classes"):
        _frontend(model_files, classes=("footstep",))


def test_frontend_refuses_a_scaler_of_the_wrong_width(model_files) -> None:
    with pytest.raises(ValueError, match="scaler has 10 bins, features have 3339"):
        _frontend(model_files, scaler="short.npz")


def test_decode_keeps_three_far_apart_tracks_as_three_detections(frontend) -> None:
    dets = frontend.decode(_row([_vec(0.0), _vec(90.0), _vec(180.0)])[None, :])
    assert _azimuths(dets) == [0.0, 90.0, 180.0]
    assert all(d.kind == "footstep" and d.frame == 0 for d in dets)
    assert all(d.activity == pytest.approx(0.8) for d in dets)


def test_decode_without_activity_emits_nothing(frontend) -> None:
    assert frontend.decode(np.zeros((3, 18))) == []
    assert frontend.decode(_row([_vec(0.0, 0.4), _vec(90.0, 0.49), None])[None, :]) == []


@pytest.mark.parametrize(
    ("tracks", "expected"),
    [
        ([_vec(0.0), _vec(6.0), _vec(120.0)], [120.0, 3.0]),
        ([_vec(0.0), _vec(6.0), None], [3.0]),
        ([_vec(-90.0), _vec(40.0), _vec(46.0)], [-90.0, 43.0]),
        ([None, _vec(40.0), _vec(46.0)], [43.0]),
        ([_vec(170.0), _vec(0.0), _vec(176.0)], [0.0, 173.0]),
        ([_vec(170.0), None, _vec(176.0)], [173.0]),
    ],
)
def test_decode_merges_one_similar_pair_and_keeps_the_odd_track(frontend, tracks, expected) -> None:
    dets = frontend.decode(_row(tracks)[None, :])
    assert _azimuths(dets) == pytest.approx(expected, abs=1e-6)
    assert dets[-1].activity == pytest.approx(0.8 * math.cos(math.radians(3.0)))


def test_decode_merges_three_similar_tracks_into_their_mean(frontend) -> None:
    dets = frontend.decode(_row([_vec(10.0), _vec(14.0), _vec(18.0)], cls=1)[None, :])
    assert len(dets) == 1
    assert dets[0].kind == "speech"
    assert math.degrees(dets[0].azimuth_rad) == pytest.approx(14.0)


def test_decode_reports_elevation_and_activity_of_the_vector(frontend) -> None:
    dets = frontend.decode(_row([(0.0, -0.5, 0.5), None, None])[None, :])
    assert len(dets) == 1
    assert dets[0].azimuth_rad == pytest.approx(-math.pi / 2)
    assert dets[0].elevation_rad == pytest.approx(math.pi / 4)
    assert dets[0].activity == pytest.approx(0.5 * math.sqrt(2.0))


def test_decode_limited_to_frames_keeps_the_frame_index(frontend) -> None:
    accdoa = np.stack([_row([_vec(0.0), None, None]), _row([_vec(90.0), None, None]), _row([_vec(180.0), None, None])])
    dets = frontend.decode(accdoa, frames=range(1, 2))
    assert [(d.frame, round(math.degrees(d.azimuth_rad))) for d in dets] == [(1, 90)]
    assert [d.frame for d in frontend.decode(accdoa)] == [0, 1, 2]


@pytest.mark.parametrize(("threshold", "norm", "detected"), [(0.7, 0.6, False), (0.7, 0.75, True), (0.3, 0.4, True), (0.3, 0.25, False)])
def test_decode_with_a_custom_threshold_gates_on_the_vector_norm(model_files, threshold: float, norm: float, detected: bool) -> None:
    fe = _frontend(model_files, det_threshold=threshold)
    dets = fe.decode(_row([_vec(30.0, norm), None, None])[None, :])
    assert bool(dets) is detected


def _noise(seconds: float, channels: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((int(seconds * 16000), channels)) * 0.1


def test_features_drop_extra_channels_and_apply_the_scaler(frontend, model_files) -> None:
    audio = _noise(0.5, channels=5)
    feat = frontend.features(audio)
    assert feat.shape == (25, NB_CH * NB_BINS)
    np.testing.assert_array_equal(feat, frontend.features(audio[:, :4]))
    shifted = _frontend(model_files, scaler="shifted.npz").features(audio)
    np.testing.assert_allclose(shifted, (feat - 1.0) / 2.0)


def test_to_sequences_pads_the_tail_with_the_filler_and_splits_channels(frontend) -> None:
    feat = np.arange(300 * NB_CH * NB_BINS, dtype=np.float64).reshape(300, NB_CH * NB_BINS)
    seq = frontend._to_sequences(feat)
    assert seq.shape == (2, NB_CH, 250, NB_BINS)
    np.testing.assert_array_equal(seq[0, 3, 7], feat[7, 3 * NB_BINS : 4 * NB_BINS])
    np.testing.assert_array_equal(seq[1, 6, 49], feat[299, 6 * NB_BINS :])
    assert np.all(seq[1, :, 50:] == FILLER)
    assert frontend._to_sequences(feat[:250]).shape == (1, NB_CH, 250, NB_BINS)


def test_forward_returns_one_bounded_row_per_label_frame_deterministically(frontend) -> None:
    feat = np.random.default_rng(1).standard_normal((250, NB_CH * NB_BINS))
    out = frontend.forward(feat)
    assert out.shape == (50, 18)
    assert float(np.abs(out).max()) < 1.0
    np.testing.assert_array_equal(out, frontend.forward(feat))


def test_events_from_wav_decodes_only_the_label_frames_the_clip_covers(model_files, tmp_path: Path) -> None:
    import scipy.io.wavfile as wav

    fe = _frontend(model_files, det_threshold=0.01)
    pcm = (_noise(0.75, seed=3) * 32767).astype(np.int16)
    path = tmp_path / "clip.wav"
    wav.write(path, 16000, pcm)
    events = fe.events_from_wav(path)
    audio, _fs = load_wav(path)
    every = fe.decode(fe.forward(fe.features(audio)))
    assert events
    assert max(d.frame for d in every) >= 7
    assert events == [d for d in every if d.frame < 7]


def test_events_from_wav_refuses_another_sample_rate(frontend, tmp_path: Path) -> None:
    import scipy.io.wavfile as wav

    path = tmp_path / "clip.wav"
    wav.write(path, 8000, np.zeros((800, 4), dtype=np.int16))
    with pytest.raises(ValueError, match="wav sample rate 8000 != expected 16000"):
        frontend.events_from_wav(path)


def test_load_wav_scales_int16_and_adds_a_channel_axis_to_mono(tmp_path: Path) -> None:
    import scipy.io.wavfile as wav

    mono = tmp_path / "mono.wav"
    wav.write(mono, 16000, np.array([0, 16384, -32768, 32767], dtype=np.int16))
    audio, fs = load_wav(mono)
    assert fs == 16000
    assert audio.shape == (4, 1)
    np.testing.assert_array_equal(audio[:, 0], [0.0, 0.5, -1.0, 32767 / 32768])

    floats = tmp_path / "float.wav"
    wav.write(floats, 8000, np.array([[0.25, -2.0], [1.5, 0.0]], dtype=np.float32))
    audio, fs = load_wav(floats)
    assert fs == 8000
    assert audio.dtype == np.float64
    np.testing.assert_array_equal(audio, [[0.25, -2.0], [1.5, 0.0]])


@pytest.mark.parametrize("lookahead", [-1, 50])
def test_stream_refuses_a_lookahead_outside_one_label_sequence(frontend, lookahead: int) -> None:
    with pytest.raises(ValueError, match="lookahead_frames"):
        SeldStream(frontend, lookahead)


def test_stream_is_ready_after_one_label_hop_since_the_last_step(frontend) -> None:
    stream = SeldStream(frontend, lookahead_frames=5)
    assert not stream.ready()
    stream.push(np.zeros((1599, 4)))
    assert not stream.ready()
    stream.push(np.zeros((1, 4)))
    assert stream.ready()


def test_stream_step_decodes_the_lookahead_frame_of_the_zero_padded_window(model_files) -> None:
    fe = _frontend(model_files, det_threshold=0.01)
    stream = SeldStream(fe, lookahead_frames=5)
    block = _noise(0.2, seed=4)
    stream.push(block)
    dets, end, seg = stream.step()
    assert not stream.ready()
    window = np.concatenate([np.zeros((fe.window_samples - block.shape[0], 4)), block])
    assert dets == fe.decode(fe.forward(fe.features(window)), frames=range(44, 45))
    assert dets
    assert all(d.frame == 44 for d in dets)
    assert end == block.shape[0] - 5 * 1600
    np.testing.assert_array_equal(seg, window[43 * 1600 : 45 * 1600])


def test_stream_step_at_the_first_label_frame_returns_that_frame_alone(frontend) -> None:
    stream = SeldStream(frontend, lookahead_frames=49)
    block = _noise(0.1, seed=5)
    stream.push(block)
    _dets, end, seg = stream.step()
    assert end == 1600 - 49 * 1600
    assert seg.shape == (1600, 4)
    assert not seg.any()
