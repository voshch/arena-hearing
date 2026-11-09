"""Call detector: band flux with a drivetrain ego gate and a whitened SRP bearing accumulated over the call, no ROS."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import attrs
import numpy as np

from arena_hearing.srp import AZIMUTHS, band_bins, fft_length, srp_map
from arena_hearing.timeline import AudioTimeline
from arena_hearing.whitening import whitener

SAMPLE_RATE_HZ = 16000
FRAME = 320
FLUX_FFT = 1024
BAND_HZ = (300.0, 3400.0)
BACKGROUND_FRAMES = 100
REFRACTORY_FRAMES = 15
GATE_JUMP_DB = 10.0
GATE_HOLD_S = 1.5
BEARING_WINDOW = 4096
ACCUM_FRAMES = 75
ACCUM_STEP = 2
ACCUM_FRACTION = 0.5
SETTLE_FRAMES = 100
FINALIZE_LAG_FRAMES = 3
LOADING = 1e-3
EPS = 1e-30
STATES = ("moving", "stopped")
DEFAULT_THRESHOLDS = {"moving": 7.5, "stopped": 1.0}


@attrs.frozen
class Onset:
    """A detected call with its map-frame heading, the SRP maps summed over it and its loudest block rms."""

    stamp_ns: int
    state: str
    stat_db: float
    azimuth_rad: float
    heading_rad: float
    windows: int
    peak_rms: float


@attrs.define
class _Accumulator:
    onset_stamp_ns: int
    state: str
    stat_db: float
    start_frame: int
    power: np.ndarray
    azimuth_rad: float = math.nan
    windows: int = 0
    peak_rms: float = 0.0


@attrs.frozen(eq=False)
class Calibration:
    thresholds: Mapping[str, float]
    whiteners: Mapping[str, np.ndarray] | None


def default_calibration() -> Calibration:
    return Calibration(dict(DEFAULT_THRESHOLDS), None)


def load_calibration(path: Path) -> Calibration:
    """Thresholds and band whiteners of an npz with threshold__<state> and cov__<state>__4096 per state."""
    bins = band_bins(SAMPLE_RATE_HZ, fft_length(BEARING_WINDOW), *BAND_HZ)
    with np.load(path) as data:
        thresholds = {state: float(data[f"threshold__{state}"]) for state in STATES}
        whiteners = {state: whitener(np.asarray(data[f"cov__{state}__{BEARING_WINDOW}"])[bins], LOADING) for state in STATES}
    return Calibration(thresholds, whiteners)


class BlockStream:
    """Re-blocks stamped audio frames of any length into FRAME-sample blocks stamped by their first sample, gaps filled with silence."""

    def __init__(self, channels: int) -> None:
        self._timeline = AudioTimeline(SAMPLE_RATE_HZ, max_gap_samples=FRAME * 50)
        self._buf = np.zeros((0, channels), dtype=np.float32)
        self._channels = channels
        self._consumed = 0

    def push(self, stamp_ns: int, block: np.ndarray) -> list[tuple[int, np.ndarray]]:
        fill = self._timeline.observe(stamp_ns, block.shape[0])
        if fill:
            self._buf = np.concatenate([self._buf, np.zeros((fill, self._channels), dtype=np.float32)])
        self._buf = np.concatenate([self._buf, block.astype(np.float32, copy=False)])
        out: list[tuple[int, np.ndarray]] = []
        while self._buf.shape[0] >= FRAME:
            out.append((self._timeline.time_ns(self._consumed), self._buf[:FRAME]))
            self._buf = self._buf[FRAME:]
            self._consumed += FRAME
        return out

    @property
    def end_ns(self) -> int:
        return self._timeline.time_ns(self._timeline.samples)


class Detector:
    """Band flux over the median of the previous BACKGROUND_FRAMES, muted while the drivetrain level jumps, then ACCUM_FRAMES of whitened SRP maps summed in the map frame."""

    def __init__(self, mics: np.ndarray, calibration: Calibration) -> None:
        self.mics = np.asarray(mics, dtype=np.float64)
        self.channels = self.mics.shape[0]
        self.thresholds = dict(calibration.thresholds)
        self.whiteners = calibration.whiteners
        freqs = np.fft.rfftfreq(FLUX_FFT, 1.0 / SAMPLE_RATE_HZ)
        self._band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
        self._window = np.hanning(FLUX_FFT)[:, None]
        self._ring = np.zeros((BEARING_WINDOW, self.channels))
        self._prev_mags: np.ndarray | None = None
        self._flux: deque[float] = deque(maxlen=BACKGROUND_FRAMES)
        self._pending: deque[tuple[int, float, bool]] = deque()
        self._over: deque[bool] = deque(maxlen=REFRACTORY_FRAMES)
        self._stopped_frames = 0
        self._motor_prev_db: float | None = None
        self._gate_until_ns = -1
        self._motor_stamp_ns = -1
        self._acc: _Accumulator | None = None
        self.frames = 0

    def push_motor(self, stamp_ns: int, block: np.ndarray) -> None:
        level = 10.0 * math.log10(float(np.sum(np.square(block, dtype=np.float64))) + EPS)
        if self._motor_prev_db is not None and level - self._motor_prev_db > GATE_JUMP_DB:
            self._gate_until_ns = stamp_ns + round(GATE_HOLD_S * 1e9)
        self._motor_prev_db = level
        self._motor_stamp_ns = max(self._motor_stamp_ns, stamp_ns)

    def _flux_frame(self) -> float:
        window = self._ring[-FLUX_FFT:]
        mags = np.abs(np.fft.rfft(window * self._window, axis=0))[self._band]
        rise = 0.0 if self._prev_mags is None else float(np.maximum(mags - self._prev_mags, 0.0).sum())
        self._prev_mags = mags
        return rise

    def push(self, stamp_ns: int, block: np.ndarray, moving: bool, yaw: float) -> list[Onset]:
        """One FRAME of (FRAME, channels) audio with the robot's moving flag and map yaw. Returns the calls whose accumulation ended."""
        block = np.asarray(block, dtype=np.float64)
        self._ring = np.roll(self._ring, -block.shape[0], axis=0)
        self._ring[-block.shape[0] :] = block
        flux = self._flux_frame()
        stat = 10.0 * math.log10((flux + EPS) / (float(np.median(self._flux)) + EPS)) if len(self._flux) == BACKGROUND_FRAMES else math.nan
        self._flux.append(flux)
        self._pending.append((stamp_ns, stat, moving))
        self.frames += 1
        while self._pending and (len(self._pending) > FINALIZE_LAG_FRAMES or self._pending[0][0] <= self._motor_stamp_ns):
            candidate = self._finalize(*self._pending.popleft())
            if candidate is not None and self._acc is None:
                self._acc = _Accumulator(onset_stamp_ns=candidate[0], state=candidate[1], stat_db=candidate[2], start_frame=self.frames, power=np.zeros(AZIMUTHS))
        acc = self._acc
        if acc is None:
            return []
        acc.peak_rms = max(acc.peak_rms, float(np.sqrt(np.mean(np.square(block)))))
        state = "moving" if moving else "stopped"
        gated = moving and stamp_ns <= self._gate_until_ns
        if (self.frames - acc.start_frame) % ACCUM_STEP == 0 and not gated and math.isfinite(stat) and stat >= ACCUM_FRACTION * self.thresholds[state]:
            white = None if self.whiteners is None else self.whiteners[state]
            azimuths, power = srp_map(self._ring, SAMPLE_RATE_HZ, self.mics, BAND_HZ[0], BAND_HZ[1], white)
            if acc.windows == 0:
                acc.azimuth_rad = float(azimuths[int(np.argmax(power))])
            acc.power += np.roll(power, round(yaw / (2.0 * math.pi / AZIMUTHS)))
            acc.windows += 1
        if self.frames - acc.start_frame < ACCUM_FRAMES:
            return []
        self._acc = None
        heading = -math.pi + int(np.argmax(acc.power)) * 2.0 * math.pi / AZIMUTHS if acc.windows else math.nan
        return [Onset(stamp_ns=acc.onset_stamp_ns, state=acc.state, stat_db=acc.stat_db, azimuth_rad=acc.azimuth_rad, heading_rad=heading, windows=acc.windows, peak_rms=acc.peak_rms)]

    def _finalize(self, stamp_ns: int, stat: float, moving: bool) -> tuple[int, str, float] | None:
        self._stopped_frames = 0 if moving else self._stopped_frames + 1
        if moving:
            state = "moving"
            usable = stamp_ns > self._gate_until_ns
        else:
            state = "stopped"
            usable = self._stopped_frames >= SETTLE_FRAMES
        over = usable and math.isfinite(stat) and stat >= self.thresholds[state]
        candidate = (stamp_ns, state, stat) if over and not any(self._over) else None
        self._over.append(over)
        return candidate
