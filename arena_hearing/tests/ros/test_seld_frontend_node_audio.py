"""seld_frontend built in-process on the real checkpoint: frame validation, layout refusal, detections with the fitted bearing, gaps and heartbeats."""

from __future__ import annotations

import array
import asyncio
import math
import os
import uuid

import numpy as np
import pytest

_spin_until = pytest.importorskip("arena_rclpy_mixins.testing").spin_until

BLOCK = 320
NS = 1_000_000_000
SENSITIVITY = -26.0
START_NS = 50 * NS


@pytest.fixture(scope="module")
def node(rclpy_context, default_sounds):
    for module in ("torch", "arena_robots_msgs.msg", "arena_runtime_msgs.msg", "arena_runtime_msgs.msg"):
        pytest.importorskip(module)
    import rclpy
    from arena_rclpy_mixins import ArenaMixinNode

    from arena_hearing import weights
    from arena_hearing.seld_frontend_node import SeldFrontendNode

    if not os.environ.get("ARENA_DATA_DIR") or not all((weights.data_dir() / entry["dest"]).is_file() for entry in weights.manifest()):
        pytest.skip("SELD weights not fetched (ros2 run arena_hearing hearing_setup)")
    overrides = [rclpy.Parameter("seld.device", value="cpu")]
    namespace = f"/t_{uuid.uuid4().hex[:8]}"

    class _Scope(ArenaMixinNode):
        def __init__(self, name: str, **kwargs: object) -> None:
            super().__init__(name, namespace=namespace, parameter_overrides=overrides, **kwargs)

    scoped = type("SeldFrontendNode", (SeldFrontendNode, _Scope), {})

    async def build():
        return scoped()

    loop = asyncio.new_event_loop()
    built = loop.run_until_complete(build())
    try:
        yield built
    finally:
        built.destroy_node()
        loop.close()


def _fleet(prefix: str, *names: str):
    from arena_runtime_msgs.msg import RobotFleet, RobotState

    fleet = RobotFleet()
    for name in names:
        state = RobotState()
        state.descriptor.name = name
        state.descriptor.model = "jackal"
        state.descriptor.ns = f"/{prefix}/{name}"
        state.descriptor.frame = f"{prefix}/{name}"
        fleet.robots.append(state)
    return fleet


def _bind(node):
    name = f"r{uuid.uuid4().hex[:8]}"
    node._robots._cb_fleet(_fleet(f"p{uuid.uuid4().hex[:8]}", name))
    (robot,) = list(node._robots)
    return robot


def _frame(node, block: np.ndarray, stamp_ns: int, *, positions=None, names=None, sensitivity: float = SENSITIVITY, fs: int | None = None, frame_count: int | None = None):
    from arena_robots_msgs.msg import AudioFrame
    from geometry_msgs.msg import Point

    spec = node._array
    n, ch = block.shape
    msg = AudioFrame()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, NS)
    msg.header.frame_id = "array_link"
    msg.sample_rate = node._fe.fs if fs is None else fs
    msg.channel_count = ch
    msg.channel_names = list(spec.channel_names if names is None else names)
    msg.frame_count = n if frame_count is None else frame_count
    msg.interleaved = True
    msg.microphone_positions = [Point(x=x, y=y, z=z) for x, y, z in ([mic.position_m for mic in spec.mics] if positions is None else positions)]
    msg.sensitivity_dbfs_at_94_dbspl = sensitivity
    msg.data = array.array("f", np.ascontiguousarray(block, dtype=np.float32).tobytes())
    return msg


def _silence(node) -> np.ndarray:
    return np.zeros((BLOCK, node._array.channels), dtype=np.float32)


def _period_ns(node) -> int:
    return BLOCK * NS // node._fe.fs


def _footsteps(node, az_deg: float) -> np.ndarray:
    """Six footsteps from ``az_deg`` at 3 m with exact fractional delays, 4.5 s in all."""
    librosa = pytest.importorskip("librosa")
    soundfile = pytest.importorskip("soundfile")
    from arena_robots.audio import geometric_delays_s
    from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary

    fs = node._fe.fs
    path = next(variant.path for variant in SoundLibrary.default().default_asset("footstep").variants if variant.default)
    audio, source_fs = soundfile.read(str(path), dtype="float32", always_2d=True)
    clip = librosa.resample(audio[:, 0], orig_sr=source_fs, target_sr=fs) * 0.3
    total = int(4.5 * fs)
    mono = np.zeros(total)
    for k in range(6):
        start = int((0.4 + 0.5 * k) * fs)
        mono[start : start + clip.size] += clip[: total - start]
    az = math.radians(az_deg)
    delays = geometric_delays_s((3.0 * math.cos(az), 3.0 * math.sin(az), 0.0), tuple(mic.position_m for mic in node._array.mics))
    delays = delays - delays.min()
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(total, 1.0 / fs)
    return np.stack([np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * d), total) for d in delays], axis=1).astype(np.float32)


class _Probe:
    """Detections and heartbeats of one robot."""

    def __init__(self, robot) -> None:
        import rclpy
        from arena_rclpy_mixins import qos
        from arena_robots_msgs.msg import SoundDetection
        from arena_runtime_msgs.msg import LockstepHeartbeat

        self.node = rclpy.create_node(f"seld_probe_{uuid.uuid4().hex[:8]}")
        self.detections: list = []
        self.beats: list = []
        self.node.create_subscription(SoundDetection, robot.pub.topic_name, self.detections.append, qos.reliable(50))
        self.node.create_subscription(LockstepHeartbeat, robot.tick_pub.publisher.topic_name, self.beats.append, qos.reliable(100))
        _spin_until(rclpy, [self.node], lambda: robot.pub.get_subscription_count() > 0 and robot.tick_pub.publisher.get_subscription_count() > 0)

    def wait(self, predicate) -> None:
        import rclpy

        _spin_until(rclpy, [self.node], predicate)

    def close(self) -> None:
        self.node.destroy_node()


def _stamp_ns(msg) -> int:
    return msg.header.stamp.sec * NS + msg.header.stamp.nanosec


def test_frames_with_a_bad_rate_or_size_are_counted_and_dropped(node) -> None:
    robot = _bind(node)
    block = _silence(node)
    node._cb_audio(robot, _frame(node, block, START_NS, fs=0))
    node._cb_audio(robot, _frame(node, block, START_NS, frame_count=BLOCK + 1))
    assert robot.bad_frames == 2
    assert robot.layout is None
    assert robot.stream.samples_seen == 0


@pytest.mark.parametrize(
    ("given", "match"),
    [
        ({"sensitivity": 0.0}, "no microphone sensitivity"),
        ({"fs": 44100}, "44100 Hz, the SELDnet weights need"),
        ({"names": ("a", "b", "c", "d")}, "the SELDnet weights need"),
        ({"positions": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))}, "differ from the"),
    ],
)
def test_a_layout_the_weights_cannot_read_is_refused_and_never_reaches_the_stream(node, given: dict, match: str) -> None:
    robot = _bind(node)
    node._cb_audio(robot, _frame(node, _silence(node), START_NS, **given))
    assert match in robot.refusal
    assert robot.refused == 1
    assert robot.doa is None
    assert robot.stream.samples_seen == 0


def test_refused_layout_keeps_the_heartbeat_going_until_a_valid_layout_arrives(node) -> None:
    robot = _bind(node)
    probe = _Probe(robot)
    period = _period_ns(node)
    block = _silence(node)
    try:
        for k in range(3):
            node._cb_audio(robot, _frame(node, block, START_NS + k * period, sensitivity=0.0))
        assert robot.refused == 3
        assert robot.stream.samples_seen == 0
        node._cb_audio(robot, _frame(node, block, START_NS + 3 * period))
        assert robot.refused == 0
        assert robot.refusal == ""
        assert robot.stream.samples_seen == BLOCK
        probe.wait(lambda: len(probe.beats) >= 4)
    finally:
        probe.close()
    assert [_stamp_ns(beat) for beat in probe.beats] == [START_NS + (k + 1) * period for k in range(4)]
    assert all(beat.header.frame_id == "array_link" for beat in probe.beats)


def test_layout_change_rebuilds_the_bearing_fit(node) -> None:
    robot = _bind(node)
    block = _silence(node)
    period = _period_ns(node)
    node._cb_audio(robot, _frame(node, block, START_NS))
    first = robot.doa
    assert first is not None
    node._cb_audio(robot, _frame(node, block, START_NS + period))
    assert robot.doa is first
    node._cb_audio(robot, _frame(node, block, START_NS + 2 * period, sensitivity=-30.0))
    assert robot.doa is not first
    assert robot.sensitivity_dbfs_at_94_dbspl == -30.0


def test_stamp_gap_is_filled_with_silence_on_the_timeline(node) -> None:
    robot = _bind(node)
    block = _silence(node)
    period = _period_ns(node)
    gap_samples = round(period * node._fe.fs / NS)
    node._cb_audio(robot, _frame(node, block, START_NS))
    node._cb_audio(robot, _frame(node, block, START_NS + 2 * period))
    assert robot.timeline.gaps == 1
    assert robot.timeline.samples == 2 * BLOCK + gap_samples
    assert robot.stream.samples_seen == 2 * BLOCK + gap_samples


def test_channel_declares_a_hard_heartbeat_at_the_label_frame_period(node) -> None:
    from arena_hearing.fleet import HEARTBEAT_TYPE

    robot = _bind(node)
    channel = node._channel(robot)
    assert (channel.name, channel.topic, channel.type) == (f"hearing/{robot.name}", robot.tick_pub.publisher.topic_name, HEARTBEAT_TYPE)
    assert channel.topic.endswith("/lockstep/hearing")
    assert channel.period_s == pytest.approx(node._fe.label_hop_len / node._fe.fs)
    assert channel.hard


def test_footsteps_yield_detections_with_the_fitted_bearing(node) -> None:
    az_deg = 90.0
    robot = _bind(node)
    audio = _footsteps(node, az_deg)
    probe = _Probe(robot)
    try:
        for start in range(0, audio.shape[0] - BLOCK + 1, BLOCK):
            node._cb_audio(robot, _frame(node, audio[start : start + BLOCK], START_NS + start * NS // node._fe.fs))
        probe.wait(lambda: len(probe.detections) == robot.n_events)
    finally:
        probe.close()
    footsteps = [msg for msg in probe.detections if msg.kind == "footstep"]
    assert len(footsteps) >= 3
    assert all(msg.frontend == "seld" and msg.robot == robot.name and msg.header.frame_id == "array_link" for msg in footsteps)
    assert all(msg.event_id.startswith("seld:") for msg in footsteps)
    assert all(math.isfinite(msg.level_db) and 0.0 <= msg.confidence <= 1.0 for msg in footsteps)
    stamps = [_stamp_ns(msg) for msg in footsteps]
    hop_ns = node._fe.label_hop_len * NS // node._fe.fs
    assert min(stamps) >= START_NS - hop_ns // 2
    assert max(stamps) <= START_NS + audio.shape[0] * NS // node._fe.fs
    assert all((stamp - START_NS + hop_ns // 2) % hop_ns == 0 for stamp in stamps)
    fitted = [msg.azimuth_rad for msg in footsteps if math.isfinite(msg.azimuth_rad)]
    assert fitted
    errors = [abs((math.degrees(azimuth) - az_deg + 180.0) % 360.0 - 180.0) for azimuth in fitted]
    assert float(np.median(errors)) <= 5.0


def test_robot_leaving_the_fleet_releases_its_topics(node) -> None:
    prefix = f"p{uuid.uuid4().hex[:8]}"
    node._robots._cb_fleet(_fleet(prefix, "r0", "r1"))
    r0, r1 = list(node._robots)
    assert r0.sub.topic_name.endswith("/r0/audio/raw_array")
    node._robots._cb_fleet(_fleet(prefix, "r1"))
    assert list(node._robots) == [r1]
    publishers, subscriptions = list(node.publishers), list(node.subscriptions)
    assert r0.pub not in publishers
    assert r0.tick_pub.publisher not in publishers
    assert r0.sub not in subscriptions
    assert r1.pub in publishers
