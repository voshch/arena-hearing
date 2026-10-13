"""hearing_policy built in-process on an L-shaped corridor: bend tracking, the yield machine per tick, and the mask, state and markers it publishes."""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid

import numpy as np
import pytest

SIZE = 200
RES = 0.1


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("scipy")
    pytest.importorskip("nav2_msgs.msg")
    pytest.importorskip("arena_robots_msgs.msg")
    pytest.importorskip("task_generator_msgs.msg")
    pytest.importorskip("visualization_msgs.msg")


@pytest.fixture
def loop():
    event_loop = asyncio.new_event_loop()
    try:
        yield event_loop
    finally:
        event_loop.close()


@pytest.fixture
def policy(rclpy_context, loop):
    import rclpy
    from arena_rclpy_mixins import ArenaMixinNode

    from arena_hearing.policy_node import PolicyNode

    nodes = []

    def make(**params: object):
        overrides = [rclpy.Parameter(name, value=value) for name, value in {"policy.max_linear_mps": 1.0, **params}.items()]
        namespace = f"/t_{uuid.uuid4().hex[:8]}"

        class _Scope(ArenaMixinNode):
            def __init__(self, name: str, **kwargs: object) -> None:
                super().__init__(name, namespace=namespace, parameter_overrides=overrides, **kwargs)

        scoped = type("PolicyNode", (PolicyNode, _Scope), {})

        async def build():
            return scoped()

        node = loop.run_until_complete(build())
        nodes.append(node)
        return node

    yield make
    for node in nodes:
        node.destroy_node()


def _fleet(prefix: str, *robots: tuple[str, str]):
    from task_generator_msgs.msg import RobotFleet, RobotState

    fleet = RobotFleet()
    for name, model in robots:
        state = RobotState()
        state.descriptor.name = name
        state.descriptor.model = model
        state.descriptor.ns = f"/{prefix}/{name}"
        state.descriptor.frame = f"{prefix}/{name}"
        fleet.robots.append(state)
    return fleet


def _bind(node, model: str = "jackal"):
    node._robots._cb_fleet(_fleet(f"p{uuid.uuid4().hex[:8]}", ("r0", model)))
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


def _grid(data: np.ndarray, resolution: float = RES, origin: float = 0.0):
    from nav_msgs.msg import OccupancyGrid

    msg = OccupancyGrid()
    msg.header.frame_id = "map"
    msg.info.resolution = resolution
    msg.info.height, msg.info.width = data.shape
    msg.info.origin.position.x = origin
    msg.info.origin.position.y = origin
    msg.info.origin.orientation.w = 1.0
    msg.data = np.asarray(data, dtype=np.int8).reshape(-1).tolist()
    return msg


def _l_occupancy() -> np.ndarray:
    """100 = occupied. Free legs: x in [0.5, 10] at y in [0.4, 1.6], and x in [9.4, 10.6] at y in [1, 10]."""
    centers = (np.arange(SIZE) + 0.5) * RES
    xx, yy = np.meshgrid(centers, centers)
    leg_a = (xx >= 0.5) & (xx <= 10.0) & (yy >= 0.4) & (yy <= 1.6)
    leg_b = (yy >= 1.0) & (yy <= 10.0) & (xx >= 9.4) & (xx <= 10.6)
    return np.where(leg_a | leg_b, 0, 100)


def _l_plan():
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path

    xs = np.round(np.arange(1.0, 10.0 + 1e-9, 0.1), 6)
    ys = np.round(np.arange(1.1, 9.5 + 1e-9, 0.1), 6)
    points = [(x, 1.0) for x in xs] + [(10.0, y) for y in ys]
    path = Path()
    for x, y in points:
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        path.poses.append(pose)
    return path


def _blob(center: tuple[float, float], radius: float = 0.3, value: int = 100) -> np.ndarray:
    centers = (np.arange(SIZE) + 0.5) * RES
    xx, yy = np.meshgrid(centers, centers)
    return np.where(np.hypot(xx - center[0], yy - center[1]) <= radius, value, 0)


def _corridor(node, robot, robot_xy: tuple[float, float] = (7.0, 1.0), belief: np.ndarray | None = None) -> None:
    node._cb_map(_grid(_l_occupancy()))
    node._cb_plan(robot, _l_plan())
    node._cb_belief(robot, _grid(np.zeros((SIZE, SIZE)) if belief is None else belief))
    _place(node, robot, *robot_xy)


def _cell(x: float, y: float) -> tuple[int, int]:
    return int(y / RES), int(x / RES)


def _spin_until(rclpy, nodes, predicate, timeout_sec: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)
        if predicate():
            return
    raise AssertionError("timed out waiting for ROS round-trip")


def _published(robot) -> tuple[np.ndarray, dict]:
    """The latched mask and state the robot's publishers last sent."""
    import rclpy
    from arena_rclpy_mixins import qos
    from nav_msgs.msg import OccupancyGrid
    from std_msgs.msg import String

    masks: list[OccupancyGrid] = []
    states: list[String] = []
    probe = rclpy.create_node(f"policy_probe_{uuid.uuid4().hex[:8]}")
    try:
        probe.create_subscription(OccupancyGrid, robot.pub_mask.topic_name, masks.append, qos.latched())
        probe.create_subscription(String, robot.pub_state.topic_name, states.append, qos.latched())
        _spin_until(rclpy, [probe], lambda: bool(masks) and bool(states))
    finally:
        probe.destroy_node()
    mask = masks[-1]
    return np.asarray(mask.data, dtype=np.int8).reshape(mask.info.height, mask.info.width), json.loads(states[-1].data)


def test_pct_maps_a_speed_to_a_clamped_percentage_of_the_robot_maximum(policy) -> None:
    node = policy()
    robot = _bind(node)
    assert robot.max_mps == 1.0
    assert [robot.pct(v) for v in (0.2, 0.03, 0.0001, 0.555, 3.0)] == [20, 3, 1, 56, 100]


def test_max_speed_falls_back_to_the_model_and_to_zero_percent_without_one(policy) -> None:
    node = policy(**{"policy.max_linear_mps": 0.0})
    robot = _bind(node)
    assert robot.max_mps == robot.binding.max_linear_mps
    assert robot.max_mps > 0.0
    unknown = _bind(policy(**{"policy.max_linear_mps": 0.0}), "no_such_model")
    assert unknown.binding.error
    assert unknown.max_mps == 0.0
    assert unknown.pct(0.2) == 0


def test_start_latches_a_percent_speed_filter_info_on_the_mask(policy) -> None:
    import rclpy
    from arena_rclpy_mixins import qos
    from nav2_msgs.msg import CostmapFilterInfo

    node = policy()
    robot = _bind(node)
    received: list[CostmapFilterInfo] = []
    probe = rclpy.create_node(f"info_probe_{uuid.uuid4().hex[:8]}")
    try:
        probe.create_subscription(CostmapFilterInfo, robot.pub_filter_info.topic_name, received.append, qos.latched())
        _spin_until(rclpy, [probe], lambda: bool(received))
    finally:
        probe.destroy_node()
    info = received[-1]
    assert (info.type, info.base, info.multiplier) == (1, 0.0, 1.0)
    assert info.filter_mask_topic == robot.pub_mask.topic_name
    assert info.filter_mask_topic.endswith("/r0/hearing/speed_filter_mask")


def test_channel_declares_a_hard_mask_heartbeat_at_the_publish_rate(policy) -> None:
    node = policy(**{"policy.publish_rate_hz": 20.0})
    robot = _bind(node)
    channel = node._channel(robot)
    assert (channel.name, channel.topic, channel.type) == ("policy/r0", robot.pub_mask.topic_name, "nav_msgs/msg/OccupancyGrid")
    assert channel.period_s == pytest.approx(1.1 / 20.0)
    assert channel.hard


def test_robot_leaving_the_fleet_releases_its_topics(policy) -> None:
    node = policy()
    prefix = f"p{uuid.uuid4().hex[:8]}"
    node._robots._cb_fleet(_fleet(prefix, ("r0", "jackal")))
    (robot,) = list(node._robots)
    node._robots._cb_fleet(_fleet(prefix))
    assert list(node._robots) == []
    publishers = list(node.publishers)
    subscriptions = list(node.subscriptions)
    for pub in (robot.pub_mask, robot.pub_state, robot.pub_markers.publisher, robot.pub_filter_info):
        assert pub not in publishers
    for sub in robot.subs:
        assert sub not in subscriptions


def test_map_marks_cells_at_or_above_the_threshold_occupied(policy) -> None:
    node = policy()
    node._cb_map(_grid(np.array([[0, 49, 50], [100, -1, 51]])))
    np.testing.assert_array_equal(node._occupied, [[False, False, True], [True, False, True]])


def test_plan_callback_stores_the_points_and_an_empty_plan_as_none(policy) -> None:
    from nav_msgs.msg import Path

    node = policy()
    robot = _bind(node)
    node._cb_plan(robot, _l_plan())
    assert robot.plan_dirty
    assert robot.plan_xy.shape == (176, 2)
    np.testing.assert_allclose(robot.plan_xy[[0, 90, -1]], [[1.0, 1.0], [10.0, 1.0], [10.0, 9.5]])
    robot.plan_dirty = False
    node._cb_plan(robot, Path())
    assert robot.plan_xy is None
    assert robot.plan_dirty


def test_detection_feeds_the_level_trend_and_skips_a_nan_level(policy) -> None:
    from arena_robots_msgs.msg import SoundDetection

    node = policy()
    robot = _bind(node)
    node._cb_detection(robot, SoundDetection(level_db=math.nan))
    assert robot.trend.ema is None
    assert robot.last_event_t == -1e9
    before = node._now()
    node._cb_detection(robot, SoundDetection(level_db=55.0))
    assert robot.trend.ema == 55.0
    assert before <= robot.last_event_t <= node._now()


def test_robot_pose_reads_position_and_yaw_from_tf(policy) -> None:
    node = policy()
    robot = _bind(node)
    assert node._robot_pose(robot) is None
    _place(node, robot, 3.0, -1.5, 2.0)
    assert node._robot_pose(robot) == pytest.approx((3.0, -1.5, 2.0))


def test_timer_waits_for_map_belief_and_pose(policy) -> None:
    node = policy()
    robot = _bind(node)
    node._on_timer()
    assert robot.last_tick is None
    node._cb_map(_grid(_l_occupancy()))
    node._on_timer()
    assert robot.last_tick is None
    node._cb_belief(robot, _grid(np.zeros((SIZE, SIZE))))
    node._on_timer()
    assert robot.last_tick is None
    _place(node, robot, 7.0, 1.0)
    node._on_timer()
    assert robot.last_tick is not None


def test_cruise_mask_slows_around_a_belief_blob_and_reports_the_belief_layer(policy) -> None:
    node = policy(**{"policy.listen.enabled": False, "policy.yield.enabled": False})
    robot = _bind(node)
    node._cb_map(_grid(np.zeros((SIZE, SIZE))))
    node._cb_belief(robot, _grid(_blob((5.0, 5.0))))
    _place(node, robot, 6.0, 5.0)
    node._on_timer()
    mask, state = _published(robot)
    assert mask[_cell(5.0, 5.0)] == 40
    assert mask[_cell(6.0, 5.0)] == 40
    assert mask[_cell(9.0, 5.0)] == 0
    assert state["state"] == "cruise"
    assert state["dist_to_bend_m"] is None
    assert (state["limit_pct"], state["binding_layer"]) == (40, "belief")
    assert (state["mass_total"], state["yield_count"]) == (0.0, 0)


def test_robot_outside_the_grid_reports_no_limit(policy) -> None:
    node = policy()
    robot = _bind(node)
    node._cb_map(_grid(np.zeros((SIZE, SIZE))))
    node._cb_belief(robot, _grid(_blob((5.0, 5.0))))
    _place(node, robot, -3.0, 5.0)
    node._on_timer()
    _mask, state = _published(robot)
    assert (state["limit_pct"], state["binding_layer"]) == (0, "none")


def test_listen_lane_on_the_approach_binds_at_the_robot(policy) -> None:
    from arena_hearing.corners import find_blind_bend

    node = policy()
    robot = _bind(node)
    _corridor(node, robot)
    node._on_timer()
    expected = find_blind_bend(robot.plan_xy, _l_occupancy() >= 50, (0.0, 0.0), RES, (7.0, 1.0), lookahead_m=4.0, approach_m=4.0, hold_len_m=1.5, hold_offset_m=1.0)
    assert robot.bend.bend_xy == expected.bend_xy
    assert not robot.plan_dirty
    mask, state = _published(robot)
    assert state["state"] == "listen"
    assert state["dist_to_bend_m"] == pytest.approx(expected.bend_s - 6.0, abs=0.01)
    assert (state["limit_pct"], state["binding_layer"]) == (20, "listen")
    for x, y in expected.approach_xy:
        assert mask[_cell(x, y)] == 20


def test_listen_disabled_keeps_the_state_but_paints_no_lane(policy) -> None:
    node = policy(**{"policy.listen.enabled": False})
    robot = _bind(node)
    _corridor(node, robot)
    node._on_timer()
    mask, state = _published(robot)
    assert state["state"] == "listen"
    assert (state["limit_pct"], state["binding_layer"]) == (0, "none")
    assert not mask.any()


def test_mass_at_the_bend_yields_with_a_hold_lane_and_counts_the_time(policy) -> None:
    node = policy()
    robot = _bind(node)
    _corridor(node, robot)
    node._on_timer()
    assert robot.machine.state.value == "listen"
    node._cb_belief(robot, _grid(_blob(robot.bend.bend_xy, radius=0.4)))
    node._on_timer()
    assert robot.machine.state.value == "yield"
    assert robot.time_yielding >= 0.0
    robot.last_tick -= 0.5
    node._on_timer()
    assert robot.time_yielding >= 0.5
    mask, state = _published(robot)
    assert state["state"] == "yield"
    assert state["yield_count"] == 1
    assert state["frac_ahead"] == 1.0
    assert state["mass_ahead"] == pytest.approx(state["mass_total"])
    assert state["time_yielding_s"] >= 0.5
    for x, y in robot.bend.hold_xy:
        assert mask[_cell(x, y)] == 3


def test_yield_disabled_holds_the_machine_in_listen(policy) -> None:
    node = policy(**{"policy.yield.enabled": False})
    robot = _bind(node)
    _corridor(node, robot)
    node._on_timer()
    node._cb_belief(robot, _grid(_blob(robot.bend.bend_xy, radius=0.4)))
    node._on_timer()
    assert robot.machine.state.value == "listen"
    assert robot.machine.yield_count == 1
    assert robot.time_yielding == 0.0


def test_falling_level_marks_the_robot_receding_until_the_trend_flattens(policy) -> None:
    node = policy()
    robot = _bind(node)
    _corridor(node, robot)
    node._on_timer()
    assert robot.receding_since is None
    now = node._now()
    for dt, level in ((-2.0, 80.0), (-1.0, 60.0), (0.0, 40.0)):
        robot.trend.observe(now + dt, level)
    node._on_timer()
    assert robot.receding_since is not None
    assert robot.receding_since >= now
    first = robot.receding_since
    node._on_timer()
    assert robot.receding_since == first
    robot.trend.reset()
    node._on_timer()
    assert robot.receding_since is None


def test_passing_the_bend_consumes_it_and_a_replan_nearby_does_not_rearm_it(policy) -> None:
    node = policy()
    robot = _bind(node)
    _corridor(node, robot)
    node._on_timer()
    bend_xy = robot.bend.bend_xy
    _place(node, robot, bend_xy[0] - 0.2, bend_xy[1])
    node._on_timer()
    assert robot.bend is None
    assert robot.consumed_xy == bend_xy
    assert robot.consumed_at == pytest.approx((bend_xy[0] - 0.2, bend_xy[1]))
    assert robot.machine.state.value == "cruise"

    _place(node, robot, 7.5, 1.0)
    node._cb_plan(robot, _l_plan())
    node._on_timer()
    assert robot.bend is None
    assert robot.consumed_xy == bend_xy

    _place(node, robot, 6.0, 1.0)
    node._cb_plan(robot, _l_plan())
    node._on_timer()
    assert robot.consumed_xy is None
    assert robot.bend.bend_xy == pytest.approx(bend_xy, abs=0.3)


def test_replanning_onto_the_same_bend_keeps_the_yield(policy) -> None:
    node = policy()
    robot = _bind(node)
    _corridor(node, robot)
    node._on_timer()
    node._cb_belief(robot, _grid(_blob(robot.bend.bend_xy, radius=0.4)))
    node._on_timer()
    assert robot.machine.state.value == "yield"
    node._cb_plan(robot, _l_plan())
    node._on_timer()
    assert robot.machine.state.value == "yield"
    assert robot.machine.contact


def test_update_bend_drops_the_bend_without_plan_or_map_and_on_a_straight_plan(policy) -> None:
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path

    node = policy()
    robot = _bind(node)
    _corridor(node, robot)
    node._update_bend(robot, (7.0, 1.0))
    assert robot.bend is not None
    kept = robot.bend
    node._update_bend(robot, (7.0, 1.0))
    assert robot.bend is kept

    node._cb_plan(robot, Path())
    node._update_bend(robot, (7.0, 1.0))
    assert robot.bend is None

    straight = Path()
    for x in np.arange(1.0, 9.0, 0.1):
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y = float(x), 1.0
        straight.poses.append(pose)
    node._cb_map(_grid(np.zeros((SIZE, SIZE))))
    node._cb_plan(robot, straight)
    node._update_bend(robot, (2.0, 1.0))
    assert robot.bend is None

    fresh = policy()
    other = _bind(fresh)
    fresh._cb_plan(other, _l_plan())
    fresh._update_bend(other, (7.0, 1.0))
    assert other.bend is None


def test_reset_returns_every_robot_to_cruise_and_forgets_the_bend(policy) -> None:
    from std_msgs.msg import Bool

    node = policy()
    robot = _bind(node)
    _corridor(node, robot)
    node._on_timer()
    node._cb_belief(robot, _grid(_blob(robot.bend.bend_xy, radius=0.4)))
    node._on_timer()
    robot.trend.observe(node._now(), 50.0)
    robot.consumed_xy = (1.0, 1.0)
    robot.receding_since = 1.0
    robot.time_yielding = 2.0
    node._cb_reset(Bool(data=False))
    assert robot.machine.state.value == "yield"
    node._cb_reset(Bool(data=True))
    assert robot.machine.state.value == "cruise"
    assert not robot.machine.contact
    assert (robot.bend, robot.consumed_xy, robot.receding_since, robot.trend.ema) == (None, None, None, None)
    assert robot.time_yielding == 0.0


def test_markers_draw_the_approach_and_the_hold_band_only_while_yielding(policy) -> None:
    import rclpy
    from arena_rclpy_mixins import qos
    from nav_msgs.msg import Path
    from visualization_msgs.msg import Marker, MarkerArray

    node = policy()
    robot = _bind(node)
    _corridor(node, robot)
    node._on_timer()
    approach_xy = robot.bend.approach_xy.copy()
    received: list[MarkerArray] = []
    probe = rclpy.create_node(f"marker_probe_{uuid.uuid4().hex[:8]}")
    try:
        probe.create_subscription(MarkerArray, robot.pub_markers.publisher.topic_name, received.append, qos.reliable(10))
        _spin_until(rclpy, [probe], lambda: robot.pub_markers.wanted)
        node._on_timer()
        _spin_until(rclpy, [probe], lambda: len(received) >= 1)
        node._cb_belief(robot, _grid(_blob(robot.bend.bend_xy, radius=0.4)))
        node._on_timer()
        _spin_until(rclpy, [probe], lambda: len(received) >= 2)
        node._cb_plan(robot, Path())
        node._on_timer()
        _spin_until(rclpy, [probe], lambda: len(received) >= 3)
    finally:
        probe.destroy_node()

    listen, held, cleared = received[:3]
    approach, hold = listen.markers
    assert (approach.id, approach.action, approach.type) == (0, Marker.ADD, Marker.LINE_STRIP)
    np.testing.assert_allclose([(p.x, p.y) for p in approach.points], approach_xy)
    assert all(p.z == pytest.approx(0.08) for p in approach.points)
    assert (hold.id, hold.action) == (1, Marker.DELETE)
    assert [m.action for m in held.markers] == [Marker.ADD, Marker.ADD]
    assert (held.markers[1].color.r, held.markers[1].color.g) == pytest.approx((1.0, 0.3))
    assert [m.action for m in cleared.markers] == [Marker.DELETE, Marker.DELETE]
