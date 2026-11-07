"""hearing_search_driver built in-process: map, detections and resets per robot, goals sent to a real navigate_to_pose server."""

from __future__ import annotations

import json
import math
import uuid

import numpy as np
import pytest

_spin_until = pytest.importorskip("arena_rclpy_mixins.testing").spin_until

RES = 0.1


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("nav2_msgs.action")
    pytest.importorskip("arena_robots_msgs.msg")
    pytest.importorskip("arena_runtime_msgs.msg")


@pytest.fixture
def driver(rclpy_context, loop):
    import rclpy
    from arena_rclpy_mixins import ArenaMixinNode

    from arena_hearing.search_driver_node import SearchDriverNode

    nodes = []

    def make(**params: object):
        overrides = [rclpy.Parameter(name, value=value) for name, value in params.items()]
        namespace = f"/t_{uuid.uuid4().hex[:8]}"

        class _Scope(ArenaMixinNode):
            def __init__(self, name: str, **kwargs: object) -> None:
                super().__init__(name, namespace=namespace, parameter_overrides=overrides, **kwargs)

        scoped = type("SearchDriverNode", (SearchDriverNode, _Scope), {})

        async def build():
            return scoped()

        node = loop.run_until_complete(build())
        nodes.append(node)
        return node

    yield make
    for node in nodes:
        node.destroy_node()


def _bind(node):
    from arena_runtime_msgs.msg import RobotFleet, RobotState

    prefix = f"p{uuid.uuid4().hex[:8]}"
    state = RobotState()
    state.descriptor.name = "r0"
    state.descriptor.model = "jackal"
    state.descriptor.ns = f"/{prefix}/r0"
    state.descriptor.frame = f"{prefix}/r0"
    node._robots._cb_fleet(RobotFleet(robots=[state]))
    (robot,) = list(node._robots)
    return robot


def _place(node, robot, x: float, y: float, yaw: float = 0.0) -> None:
    from geometry_msgs.msg import TransformStamped

    t = TransformStamped()
    t.header.frame_id = "map"
    t.child_frame_id = robot.binding.base_frame
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.rotation.z = math.sin(yaw / 2.0)
    t.transform.rotation.w = math.cos(yaw / 2.0)
    node._tf_buffer.set_transform_static(t, "test")


def _half_explored():
    """Free for x < 3 m, unknown beyond: one frontier along x = 2.95."""
    from nav_msgs.msg import OccupancyGrid

    data = np.full((40, 60), -1, dtype=np.int8)
    data[:, :30] = 0
    msg = OccupancyGrid()
    msg.header.frame_id = "map"
    msg.info.resolution = RES
    msg.info.height, msg.info.width = data.shape
    msg.info.origin.orientation.w = 1.0
    msg.data = data.reshape(-1).tolist()
    return msg


def _detection(robot, kind: str, azimuth: float):
    from arena_robots_msgs.msg import SoundDetection

    msg = SoundDetection()
    msg.header.frame_id = robot.binding.base_frame
    msg.kind = kind
    msg.azimuth_rad = azimuth
    msg.level_db = 60.0
    return msg


def _state(robot) -> dict:
    import rclpy
    from arena_rclpy_mixins import qos
    from std_msgs.msg import String

    states: list[String] = []
    probe = rclpy.create_node(f"search_probe_{uuid.uuid4().hex[:8]}")
    try:
        probe.create_subscription(String, robot.pub_state.topic_name, states.append, qos.latched())
        _spin_until(rclpy, [probe], lambda: bool(states))
    finally:
        probe.destroy_node()
    return json.loads(states[-1].data)


def test_the_driver_sends_the_nearest_frontier_to_navigate_to_pose(driver) -> None:
    import rclpy
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionServer

    node = driver()
    robot = _bind(node)
    node._cb_map(robot, _half_explored())
    _place(node, robot, 1.0, 2.0)
    goals: list[NavigateToPose.Goal] = []

    def execute(handle):
        goals.append(handle.request)
        handle.succeed()
        return NavigateToPose.Result()

    server_node = rclpy.create_node(f"nav2_probe_{uuid.uuid4().hex[:8]}")
    server = ActionServer(server_node, NavigateToPose, robot.action, execute)
    try:
        _spin_until(rclpy, [node, server_node], robot.client.server_is_ready)
        node._tick(robot)
        _spin_until(rclpy, [node, server_node], lambda: bool(goals))
    finally:
        server.destroy()
        server_node.destroy_node()
    pose = goals[0].pose
    assert pose.header.frame_id == "map"
    assert abs(pose.pose.position.x - 2.95) < 1e-6
    assert _state(robot)["goals_sent"] == 1


def test_a_call_becomes_a_map_frame_bearing_from_the_robot(driver) -> None:
    node = driver()
    robot = _bind(node)
    _place(node, robot, 1.0, 2.0, yaw=math.pi / 2.0)
    node._cb_detection(robot, _detection(robot, "speech", 0.25))
    assert robot.calls == 1
    assert robot.heard is not None
    assert robot.heard.robot_xy == pytest.approx((1.0, 2.0))
    assert robot.heard.bearing_rad == pytest.approx(math.pi / 2.0 + 0.25)


def test_a_sound_of_another_kind_is_no_call(driver) -> None:
    node = driver()
    robot = _bind(node)
    _place(node, robot, 1.0, 2.0)
    node._cb_detection(robot, _detection(robot, "footstep", 0.25))
    assert robot.calls == 0
    assert robot.heard is None


def test_a_tick_without_nav2_reports_its_phase_and_sends_nothing(driver) -> None:
    node = driver()
    robot = _bind(node)
    node._cb_map(robot, _half_explored())
    _place(node, robot, 1.0, 2.0)
    node._tick(robot)
    state = _state(robot)
    assert state["phase"] == "select"
    assert state["goals_sent"] == 0


def test_a_reset_forgets_the_calls_of_the_last_episode(driver) -> None:
    from std_msgs.msg import Bool

    node = driver()
    robot = _bind(node)
    _place(node, robot, 1.0, 2.0)
    node._cb_detection(robot, _detection(robot, "speech", 0.25))
    node._cb_reset(Bool(data=True))
    assert robot.calls == 0
    assert robot.heard is None


def test_a_reset_forgets_the_map_of_the_last_episode(driver) -> None:
    from std_msgs.msg import Bool

    node = driver()
    robot = _bind(node)
    node._cb_map(robot, _half_explored())
    node._cb_reset(Bool(data=True))
    assert robot.grid is None


def test_a_map_from_before_the_reset_ended_is_ignored(driver) -> None:
    from std_msgs.msg import Bool

    node = driver()
    robot = _bind(node)
    node._cb_reset(Bool(data=True))
    node._cb_map(robot, _half_explored())
    assert robot.grid is None
    node._cb_reset(Bool(data=False))
    node._cb_map(robot, _half_explored())
    assert robot.grid is not None


def _pending_server(robot):
    import rclpy
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionServer, CancelResponse

    accepted: list[object] = []
    cancelled: list[object] = []

    def cancel(handle):
        cancelled.append(handle.goal_id)
        return CancelResponse.ACCEPT

    server_node = rclpy.create_node(f"nav2_probe_{uuid.uuid4().hex[:8]}")
    server = ActionServer(server_node, NavigateToPose, robot.action, lambda handle: NavigateToPose.Result(), handle_accepted_callback=accepted.append, cancel_callback=cancel)
    return server_node, server, accepted, cancelled


def test_a_goal_accepted_after_a_reset_is_cancelled(driver) -> None:
    import rclpy
    from std_msgs.msg import Bool

    node = driver()
    robot = _bind(node)
    server_node, server, accepted, cancelled = _pending_server(robot)
    try:
        _spin_until(rclpy, [node, server_node], robot.client.server_is_ready)
        node._send(robot, (2.0, 2.0), 0.0)
        node._cb_reset(Bool(data=True))
        _spin_until(rclpy, [node, server_node], lambda: bool(cancelled))
    finally:
        server.destroy()
        server_node.destroy_node()
    assert len(accepted) == 1
    assert robot.goal_handle is None


def test_a_goal_replaced_by_a_newer_one_is_left_to_nav2(driver) -> None:
    import rclpy

    node = driver()
    robot = _bind(node)
    server_node, server, accepted, cancelled = _pending_server(robot)
    try:
        _spin_until(rclpy, [node, server_node], robot.client.server_is_ready)
        node._send(robot, (2.0, 2.0), 0.0)
        node._send(robot, (2.5, 2.0), 0.0)
        _spin_until(rclpy, [node, server_node], lambda: len(accepted) == 2 and robot.goal_handle is not None)
        settle = iter(range(25))
        _spin_until(rclpy, [node, server_node], lambda: next(settle, None) is None)
    finally:
        server.destroy()
        server_node.destroy_node()
    assert cancelled == []
