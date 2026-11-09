"""call_frontend built in-process: a call in raw array frames becomes a map-frame onset detection, an unusable stream is refused."""

from __future__ import annotations

import array
import asyncio
import math
import time
import uuid

import numpy as np
import pytest

FS = 16000
NS = 1_000_000_000
BLOCK = 320
SENSITIVITY = -26.0
START_NS = 50 * NS
SPEED_OF_SOUND = 343.0
MICS = ((0.19, 0.135, 0.22), (0.19, -0.135, 0.22), (-0.19, 0.135, 0.22), (-0.19, -0.135, 0.22))


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_robots_msgs.msg")
    pytest.importorskip("arena_runtime_msgs.msg")
    pytest.importorskip("arena_runtime_msgs.msg")


@pytest.fixture
def loop():
    event_loop = asyncio.new_event_loop()
    try:
        yield event_loop
    finally:
        event_loop.close()


@pytest.fixture
def call(rclpy_context, loop):
    import rclpy
    from arena_rclpy_mixins import ArenaMixinNode

    from arena_hearing.call_frontend_node import CallFrontendNode

    nodes = []

    def make(**params: object):
        overrides = [rclpy.Parameter(name, value=value) for name, value in params.items()]
        namespace = f"/t_{uuid.uuid4().hex[:8]}"

        class _Scope(ArenaMixinNode):
            def __init__(self, name: str, **kwargs: object) -> None:
                super().__init__(name, namespace=namespace, parameter_overrides=overrides, **kwargs)

        scoped = type("CallFrontendNode", (CallFrontendNode, _Scope), {})

        async def build():
            return scoped()

        node = loop.run_until_complete(build())
        nodes.append(node)
        return node

    yield make
    for node in nodes:
        node.destroy_node()


def _fleet(prefix: str, name: str):
    from arena_runtime_msgs.msg import RobotFleet, RobotState

    fleet = RobotFleet()
    state = RobotState()
    state.descriptor.name = name
    state.descriptor.model = "jackal"
    state.descriptor.ns = f"/{prefix}/{name}"
    state.descriptor.frame = f"{prefix}/{name}"
    fleet.robots.append(state)
    return fleet


def _bind(node):
    node._robots._cb_fleet(_fleet(f"p{uuid.uuid4().hex[:8]}", "r0"))
    (robot,) = list(node._robots)
    return robot


def _array_yaw(node, yaw: float) -> None:
    from geometry_msgs.msg import TransformStamped

    t = TransformStamped()
    t.header.frame_id = "map"
    t.child_frame_id = "array_link"
    t.transform.rotation.z = math.sin(yaw / 2.0)
    t.transform.rotation.w = math.cos(yaw / 2.0)
    node._tf_buffer.set_transform_static(t, "test")


def _frame(block: np.ndarray, stamp_ns: int, *, fs: int = FS):
    from arena_robots_msgs.msg import AudioFrame
    from geometry_msgs.msg import Point

    n, ch = block.shape
    msg = AudioFrame()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, NS)
    msg.header.frame_id = "array_link"
    msg.sample_rate = fs
    msg.channel_count = ch
    msg.frame_count = n
    msg.interleaved = True
    msg.microphone_positions = [Point(x=x, y=y, z=z) for x, y, z in MICS]
    msg.sensitivity_dbfs_at_94_dbspl = SENSITIVITY
    msg.data = array.array("f", np.ascontiguousarray(block, dtype=np.float32).tobytes())
    return msg


def _arrival(signal: np.ndarray, azimuth: float) -> np.ndarray:
    direction = np.asarray([math.cos(azimuth), math.sin(azimuth), 0.0])
    delays = -(np.asarray(MICS) @ direction) / SPEED_OF_SOUND
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, 1.0 / FS)
    return np.stack([np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * d), signal.size) for d in delays], axis=1)


def _scene(azimuth_deg: float) -> np.ndarray:
    rng = np.random.default_rng(1)
    audio = rng.standard_normal((6 * FS, 4)) * 1e-3
    t = np.arange(2 * FS) / FS
    voiced = sum(np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi)) / k for k, f0 in enumerate((220.0, 440.0, 660.0, 880.0, 1320.0, 1760.0), start=1))
    call = voiced * (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t) ** 2) * 0.02
    audio[3 * FS : 5 * FS] += _arrival(call, math.radians(azimuth_deg))
    return audio.astype(np.float32)


def _feed(node, robot, audio: np.ndarray, *, fs: int = FS) -> None:
    for i in range(0, audio.shape[0], BLOCK):
        node._cb_audio(robot, _frame(audio[i : i + BLOCK], START_NS + i * NS // fs, fs=fs))


def _spin_until(rclpy, nodes, predicate, timeout_sec: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)
        if predicate():
            return
    raise AssertionError("timed out waiting for ROS round-trip")


class _Probe:
    def __init__(self, robot) -> None:
        import rclpy
        from arena_rclpy_mixins import qos
        from arena_robots_msgs.msg import SoundDetection
        from arena_runtime_msgs.msg import LockstepHeartbeat

        self.node = rclpy.create_node(f"call_probe_{uuid.uuid4().hex[:8]}")
        self.detections: list = []
        self.beats: list = []
        self.node.create_subscription(SoundDetection, robot.pub.topic_name, self.detections.append, qos.reliable(50))
        self.node.create_subscription(LockstepHeartbeat, robot.tick_pub.topic_name, self.beats.append, qos.reliable(1000))
        _spin_until(rclpy, [self.node], lambda: robot.pub.get_subscription_count() > 0 and robot.tick_pub.get_subscription_count() > 0)

    def wait(self, predicate) -> None:
        import rclpy

        _spin_until(rclpy, [self.node], predicate)

    def close(self) -> None:
        self.node.destroy_node()


def _error_deg(heading: float, truth_deg: float) -> float:
    return abs((math.degrees(heading) - truth_deg + 180.0) % 360.0 - 180.0)


@pytest.mark.parametrize(("azimuth_deg", "yaw_deg"), [(60.0, 0.0), (60.0, 90.0)])
def test_a_call_becomes_a_map_frame_onset_detection_rotated_by_the_array_yaw(call, azimuth_deg: float, yaw_deg: float) -> None:
    node = call()
    robot = _bind(node)
    _array_yaw(node, math.radians(yaw_deg))
    probe = _Probe(robot)
    try:
        _feed(node, robot, _scene(azimuth_deg))
        probe.wait(lambda: bool(probe.detections))
        first = probe.detections[0]
    finally:
        probe.close()
    assert (first.frontend, first.kind, first.header.frame_id, first.robot) == ("call", "onset", "map", "r0")
    assert first.event_id.startswith("call:")
    assert _error_deg(first.azimuth_rad, azimuth_deg + yaw_deg) < 5.0
    assert abs((first.header.stamp.sec * NS + first.header.stamp.nanosec - START_NS) / NS - 3.0) < 0.2
    assert math.isfinite(first.level_db)
    assert 0.0 < first.confidence < 1.0
    assert math.isnan(first.elevation_rad)


def test_noise_alone_publishes_no_detection_and_every_frame_a_heartbeat(call) -> None:
    node = call()
    robot = _bind(node)
    probe = _Probe(robot)
    try:
        _feed(node, robot, np.random.default_rng(2).standard_normal((2 * FS, 4)).astype(np.float32) * 1e-3)
        probe.wait(lambda: len(probe.beats) >= 2 * FS // BLOCK)
        beats = list(probe.beats)
    finally:
        probe.close()
    assert probe.detections == []
    stamps = [b.header.stamp.sec * NS + b.header.stamp.nanosec for b in beats]
    assert stamps == sorted(stamps)
    assert stamps[-1] == START_NS + 2 * NS


def test_a_stream_at_another_sample_rate_is_refused_but_still_heartbeats(call) -> None:
    node = call()
    robot = _bind(node)
    probe = _Probe(robot)
    try:
        _feed(node, robot, np.zeros((BLOCK * 3, 4), dtype=np.float32), fs=8000)
        probe.wait(lambda: len(probe.beats) >= 3)
    finally:
        probe.close()
    assert robot.pipeline is None
    assert "16000" in robot.refusal
    assert robot.refused == 3


def test_a_frame_without_microphone_sensitivity_is_refused(call) -> None:
    node = call()
    robot = _bind(node)
    frame = _frame(np.zeros((BLOCK, 4), dtype=np.float32), START_NS)
    frame.sensitivity_dbfs_at_94_dbspl = 0.0

    node._cb_audio(robot, frame)

    assert robot.pipeline is None
    assert "sensitivity" in robot.refusal


def test_robot_leaving_the_fleet_releases_its_topics(call) -> None:
    from arena_runtime_msgs.msg import RobotFleet

    node = call()
    prefix = f"p{uuid.uuid4().hex[:8]}"
    node._robots._cb_fleet(_fleet(prefix, "r0"))
    (robot,) = list(node._robots)
    stem = robot.motor_sub
    assert stem in list(node.subscriptions)

    node._robots._cb_fleet(RobotFleet())

    assert list(node._robots) == []
    assert robot.pub not in list(node.publishers)
    assert robot.sub not in list(node.subscriptions)
    assert stem not in list(node.subscriptions)


def test_channel_declares_a_hard_audio_heartbeat(call) -> None:
    node = call()
    robot = _bind(node)

    channel = node._channel(robot)

    assert (channel.name, channel.topic) == ("hearing/r0", robot.tick_pub.topic_name)
    assert channel.hard
