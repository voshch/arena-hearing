"""srp_frontend built in-process: frame validation, layout refusal, the onset-to-detection path with a known bearing, gaps and heartbeats."""

from __future__ import annotations

import array
import asyncio
import math
import time
import uuid

import numpy as np
import pytest

FS = 16000
HOP = 1600
NS = 1_000_000_000
SENSITIVITY = -26.0
START_NS = 50 * NS


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_robots_msgs.msg")
    pytest.importorskip("arena_runtime_msgs.msg")
    pytest.importorskip("task_generator_msgs.msg")


@pytest.fixture
def loop():
    event_loop = asyncio.new_event_loop()
    try:
        yield event_loop
    finally:
        event_loop.close()


@pytest.fixture
def srp(rclpy_context, loop):
    import rclpy
    from arena_rclpy_mixins import ArenaMixinNode

    from arena_hearing.srp_frontend_node import SrpFrontendNode

    nodes = []

    def make(**params: object):
        overrides = [rclpy.Parameter(name, value=value) for name, value in params.items()]
        namespace = f"/t_{uuid.uuid4().hex[:8]}"

        class _Scope(ArenaMixinNode):
            def __init__(self, name: str, **kwargs: object) -> None:
                super().__init__(name, namespace=namespace, parameter_overrides=overrides, **kwargs)

        scoped = type("SrpFrontendNode", (SrpFrontendNode, _Scope), {})

        async def build():
            return scoped()

        node = loop.run_until_complete(build())
        nodes.append(node)
        return node

    yield make
    for node in nodes:
        node.destroy_node()


def _fleet(prefix: str, *names: str):
    from task_generator_msgs.msg import RobotFleet, RobotState

    fleet = RobotFleet()
    for name in names:
        state = RobotState()
        state.descriptor.name = name
        state.descriptor.model = "jackal"
        state.descriptor.ns = f"/{prefix}/{name}"
        state.descriptor.frame = f"{prefix}/{name}"
        fleet.robots.append(state)
    return fleet


def _bind(node, *names: str) -> list:
    node._robots._cb_fleet(_fleet(f"p{uuid.uuid4().hex[:8]}", *names))
    return list(node._robots)


def _positions() -> tuple[tuple[float, float, float], ...]:
    from arena_robots.audio import load_array_spec

    return tuple(mic.position_m for mic in load_array_spec("four_mic").mics)


def _frame(block: np.ndarray, stamp_ns: int, *, positions=None, sensitivity: float = SENSITIVITY, interleaved: bool = True, fs: int = FS, frame_count: int | None = None):
    from arena_robots_msgs.msg import AudioFrame
    from geometry_msgs.msg import Point

    n, ch = block.shape
    msg = AudioFrame()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, NS)
    msg.header.frame_id = "array_link"
    msg.sample_rate = fs
    msg.channel_count = ch
    msg.frame_count = n if frame_count is None else frame_count
    msg.interleaved = interleaved
    msg.microphone_positions = [Point(x=x, y=y, z=z) for x, y, z in (_positions() if positions is None else positions)]
    msg.sensitivity_dbfs_at_94_dbspl = sensitivity
    payload = block if interleaved else block.T
    msg.data = array.array("f", np.ascontiguousarray(payload, dtype=np.float32).tobytes())
    return msg


def _plane_wave(az_deg: float, total: int, start: int, length: int, amp: float, seed: int) -> np.ndarray:
    """4-channel Hann-tapered burst from ``az_deg`` at 3 m with exact fractional delays."""
    from arena_robots.audio import geometric_delays_s

    az = math.radians(az_deg)
    delays = geometric_delays_s((3.0 * math.cos(az), 3.0 * math.sin(az), 0.0), _positions())
    delays = delays - delays.min()
    mono = np.zeros(total)
    mono[start : start + length] = np.random.default_rng(seed).standard_normal(length) * np.hanning(length) * amp
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(total, 1.0 / FS)
    return np.stack([np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * d), total) for d in delays], axis=1)


def _scene(az_deg: float = 90.0, *, identical: bool = False) -> np.ndarray:
    """1.5 s of faint independent noise with a 50 ms burst inside hop 12."""
    total = 15 * HOP
    floor = np.random.default_rng(1).standard_normal((total, 4)) * 1e-3
    if identical:
        burst = np.zeros(total)
        burst[12 * HOP + 400 : 12 * HOP + 1200] = np.random.default_rng(2).standard_normal(800) * 0.3
        return (floor + burst[:, None]).astype(np.float32)
    return (floor + _plane_wave(az_deg, total, 12 * HOP + 400, 800, 0.3, seed=2)).astype(np.float32)


def _feed(node, robot, audio: np.ndarray, *, block: int = 320, interleaved: bool = True) -> None:
    for i in range(0, audio.shape[0], block):
        node._cb_audio(robot, _frame(audio[i : i + block], START_NS + i * NS // FS, interleaved=interleaved))


def _spin_until(rclpy, nodes, predicate, timeout_sec: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)
        if predicate():
            return
    raise AssertionError("timed out waiting for ROS round-trip")


class _Probe:
    """Detections and heartbeats of the given robots."""

    def __init__(self, robots: list) -> None:
        import rclpy
        from arena_rclpy_mixins import qos
        from arena_robots_msgs.msg import SoundDetection
        from arena_runtime_msgs.msg import LockstepHeartbeat

        self.node = rclpy.create_node(f"srp_probe_{uuid.uuid4().hex[:8]}")
        self.detections: dict[str, list] = {r.name: [] for r in robots}
        self.beats: dict[str, list] = {r.name: [] for r in robots}
        for r in robots:
            self.node.create_subscription(SoundDetection, r.pub.topic_name, self.detections[r.name].append, qos.reliable(50))
            self.node.create_subscription(LockstepHeartbeat, r.tick_pub.topic_name, self.beats[r.name].append, qos.reliable(100))
        _spin_until(rclpy, [self.node], lambda: all(r.pub.get_subscription_count() > 0 and r.tick_pub.get_subscription_count() > 0 for r in robots))

    def wait(self, predicate) -> None:
        import rclpy

        _spin_until(rclpy, [self.node], predicate)

    def close(self) -> None:
        self.node.destroy_node()


def _stamp_ns(msg) -> int:
    return msg.header.stamp.sec * NS + msg.header.stamp.nanosec


@pytest.mark.parametrize(("margin_db", "confidence"), [(-10.0, 0.0), (0.0, 0.0), (20.0 * math.log10(2.0), 0.5), (20.0, 0.9), (40.0, 0.99)])
def test_onset_confidence_squashes_the_margin_into_the_unit_interval(margin_db: float, confidence: float) -> None:
    from arena_hearing.srp_frontend_node import onset_confidence

    assert onset_confidence(margin_db) == pytest.approx(confidence)


def test_onset_confidence_rises_with_the_margin() -> None:
    from arena_hearing.srp_frontend_node import onset_confidence

    values = [onset_confidence(m) for m in np.linspace(0.0, 60.0, 61)]
    assert np.all(np.diff(values) > 0.0)
    assert values[-1] < 1.0


def test_pipeline_sizes_hop_onset_and_timeline_from_the_parameters(srp) -> None:
    node = srp(**{"srp.hop_s": 0.05, "srp.floor_window_s": 2.0})
    pipe = node._pipeline(FS, 4, _positions(), SENSITIVITY)
    assert pipe.hop_samples == 800
    assert pipe.timeline.max_gap_samples == 8000
    assert pipe.onset._floor_history.maxlen == 40
    assert pipe.buf.shape == (0, 4)
    assert pipe.prev_hop.shape == (800, 4)
    assert len(pipe.doa.pairs) == 6
    assert pipe.sensitivity_dbfs_at_94_dbspl == SENSITIVITY


@pytest.mark.parametrize(
    ("channels", "positions", "sensitivity", "match"),
    [
        (3, None, SENSITIVITY, "declares 4 microphone positions for 3 channels"),
        (4, None, 0.0, "no microphone sensitivity"),
        (2, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.3)), SENSITIVITY, "two apart in the plane"),
    ],
)
def test_pipeline_refuses_a_layout_without_bearing_or_level(srp, channels: int, positions, sensitivity: float, match: str) -> None:
    node = srp()
    with pytest.raises(ValueError, match=match):
        node._pipeline(FS, channels, _positions() if positions is None else positions, sensitivity)


def test_frames_with_a_bad_rate_or_size_are_counted_and_dropped(srp) -> None:
    node = srp()
    (robot,) = _bind(node, "r0")
    block = np.zeros((320, 4), dtype=np.float32)
    node._cb_audio(robot, _frame(block, START_NS, fs=0))
    node._cb_audio(robot, _frame(block, START_NS, frame_count=321))
    assert robot.bad_frames == 2
    assert robot.layout is None
    assert robot.pipeline is None


def test_refused_layout_keeps_the_heartbeat_going_until_a_valid_layout_arrives(srp) -> None:
    node = srp()
    (robot,) = _bind(node, "r0")
    probe = _Probe([robot])
    try:
        block = np.zeros((320, 4), dtype=np.float32)
        for k in range(3):
            node._cb_audio(robot, _frame(block, START_NS + k * 20_000_000, sensitivity=0.0))
        assert robot.pipeline is None
        assert robot.refused == 3
        assert robot.refusal == "frame declares no microphone sensitivity"
        node._diag()
        node._cb_audio(robot, _frame(block, START_NS + 60_000_000))
        assert robot.refused == 0
        assert robot.refusal == ""
        assert robot.pipeline is not None
        node._diag()
        probe.wait(lambda: len(probe.beats["r0"]) >= 4)
    finally:
        probe.close()
    beats = [_stamp_ns(b) for b in probe.beats["r0"]]
    assert beats == [START_NS + 20_000_000, START_NS + 40_000_000, START_NS + 60_000_000, START_NS + 80_000_000]
    assert all(b.header.frame_id == "array_link" for b in probe.beats["r0"])


def test_onset_burst_yields_one_detection_with_the_source_bearing_and_level(srp) -> None:
    from arena_robots.audio import dbfs_from_rms, dbfs_to_spl

    node = srp()
    r0, r1 = _bind(node, "r0", "r1")
    audio = _scene(90.0)
    probe = _Probe([r0, r1])
    try:
        _feed(node, r0, audio)
        _feed(node, r1, audio, interleaved=False)
        probe.wait(lambda: len(probe.detections["r0"]) >= 1 and len(probe.detections["r1"]) >= 1 and len(probe.beats["r1"]) >= 75)
    finally:
        probe.close()
    assert (r0.n_hops, r0.n_events, r0.n_no_bearing) == (15, 1, 0)
    (det,) = probe.detections["r0"]
    assert (det.robot, det.frontend, det.kind, det.event_id) == ("r0", "srp", "onset", "srp:13")
    assert det.header.frame_id == "array_link"
    assert _stamp_ns(det) == START_NS + 13 * HOP * NS // FS - 50_000_000
    assert abs(math.degrees(det.azimuth_rad) - 90.0) <= 5.0
    assert math.isnan(det.elevation_rad)
    hop = audio[12 * HOP : 13 * HOP]
    assert det.level_db == pytest.approx(dbfs_to_spl(dbfs_from_rms(float(np.sqrt(np.mean(hop**2)))), SENSITIVITY), abs=1e-3)
    assert 0.99 < det.confidence < 1.0
    (planar,) = probe.detections["r1"]
    assert (planar.azimuth_rad, planar.level_db, planar.event_id) == (det.azimuth_rad, det.level_db, det.event_id)
    assert _stamp_ns(probe.beats["r1"][-1]) == START_NS + audio.shape[0] * NS // FS
    np.testing.assert_array_equal(r0.pipeline.prev_hop, audio[14 * HOP :])


@pytest.mark.parametrize("az_deg", [0.0, 135.0, 270.0])
def test_onset_bearing_follows_the_source_around_the_array(srp, az_deg: float) -> None:
    node = srp()
    (robot,) = _bind(node, "r0")
    probe = _Probe([robot])
    try:
        _feed(node, robot, _scene(az_deg), block=1600)
        probe.wait(lambda: len(probe.detections["r0"]) >= 1)
    finally:
        probe.close()
    fitted = math.degrees(probe.detections["r0"][0].azimuth_rad)
    assert abs((fitted - az_deg + 180.0) % 360.0 - 180.0) <= 5.0


def test_onset_from_identical_channels_has_no_bearing(srp) -> None:
    node = srp()
    (robot,) = _bind(node, "r0")
    probe = _Probe([robot])
    try:
        _feed(node, robot, _scene(identical=True), block=1600)
        probe.wait(lambda: len(probe.detections["r0"]) >= 1)
    finally:
        probe.close()
    assert (robot.n_events, robot.n_no_bearing) == (1, 1)
    assert math.isnan(probe.detections["r0"][0].azimuth_rad)


def test_stamp_gap_is_filled_with_silence_on_the_timeline(srp) -> None:
    node = srp()
    (robot,) = _bind(node, "r0")
    block = np.full((320, 4), 0.01, dtype=np.float32)
    node._cb_audio(robot, _frame(block, START_NS))
    node._cb_audio(robot, _frame(block, START_NS + 70_000_000))
    pipe = robot.pipeline
    assert pipe.timeline.gaps == 1
    assert pipe.timeline.samples == 320 + 800 + 320
    assert pipe.buf.shape == (1440, 4)
    assert not pipe.buf[320:1120].any()
    assert np.all(pipe.buf[1120:] == np.float32(0.01))
    node._cb_audio(robot, _frame(block, START_NS + 90_000_000))
    assert robot.n_hops == 1
    assert pipe.consumed == HOP
    assert pipe.buf.shape == (160, 4)


def test_layout_change_rebuilds_the_pipeline(srp) -> None:
    node = srp()
    (robot,) = _bind(node, "r0")
    block = np.zeros((320, 4), dtype=np.float32)
    node._cb_audio(robot, _frame(block, START_NS))
    first = robot.pipeline
    node._cb_audio(robot, _frame(block, START_NS + 20_000_000))
    assert robot.pipeline is first
    node._cb_audio(robot, _frame(block, START_NS + 40_000_000, sensitivity=-30.0))
    assert robot.pipeline is not first
    assert robot.pipeline.sensitivity_dbfs_at_94_dbspl == -30.0
    assert robot.layout[3] == -30.0


def test_channel_declares_a_hard_heartbeat_at_the_hop_period(srp) -> None:
    from arena_hearing.fleet import HEARTBEAT_TYPE

    node = srp(**{"srp.hop_s": 0.08})
    (robot,) = _bind(node, "r0")
    channel = node._channel(robot)
    assert (channel.name, channel.topic, channel.type) == ("hearing/r0", robot.tick_pub.topic_name, HEARTBEAT_TYPE)
    assert channel.topic.endswith("/r0/lockstep/hearing")
    assert channel.period_s == pytest.approx(0.08)
    assert channel.hard


@pytest.mark.parametrize(("reliable", "policy"), [(True, "RELIABLE"), (False, "BEST_EFFORT")])
def test_audio_subscription_reliability_follows_the_parameter(srp, reliable: bool, policy: str) -> None:
    node = srp(**{"audio.reliable.enabled": reliable})
    assert node._audio_qos.reliability.name == policy


def test_robot_leaving_the_fleet_releases_its_topics(srp) -> None:
    node = srp()
    prefix = f"p{uuid.uuid4().hex[:8]}"
    node._robots._cb_fleet(_fleet(prefix, "r0", "r1"))
    r0, r1 = list(node._robots)
    assert r0.sub.topic_name.endswith("/r0/audio/raw_array")
    node._robots._cb_fleet(_fleet(prefix, "r1"))
    assert list(node._robots) == [r1]
    publishers, subscriptions = list(node.publishers), list(node.subscriptions)
    assert r0.pub not in publishers
    assert r0.tick_pub not in publishers
    assert r0.sub not in subscriptions
    assert r1.pub in publishers
