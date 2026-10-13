"""hearing_belief built in-process: map layouts, detection rotation into the map frame, decay, reset and wedge markers."""

from __future__ import annotations

import asyncio
import math
import time
import uuid

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("scipy")
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
def belief(rclpy_context, loop, default_sounds):
    import rclpy
    from arena_rclpy_mixins import ArenaMixinNode

    from arena_hearing.belief_node import BeliefNode

    nodes = []

    def make(**params: object):
        overrides = [rclpy.Parameter(name, value=value) for name, value in params.items()]
        namespace = f"/t_{uuid.uuid4().hex[:8]}"

        class _Scope(ArenaMixinNode):
            def __init__(self, name: str, **kwargs: object) -> None:
                super().__init__(name, namespace=namespace, parameter_overrides=overrides, **kwargs)

        scoped = type("BeliefNode", (BeliefNode, _Scope), {})

        async def build():
            return scoped()

        node = loop.run_until_complete(build())
        nodes.append(node)
        return node

    yield make
    for node in nodes:
        node.destroy_node()


def _standalone(**given: object) -> dict[str, object]:
    return {
        "belief.standalone.enabled": True,
        "belief.standalone.origin_m": [-5.0, -5.0],
        "belief.standalone.size_m": [10.0, 10.0],
        "belief.resolution_m": 0.1,
        "belief.markers.enabled": False,
        **given,
    }


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


def _bind(node, *robots: tuple[str, str]) -> list:
    node._robots._cb_fleet(_fleet(f"p{uuid.uuid4().hex[:8]}", *robots))
    return list(node._robots)


def _tf(node, child: str, x: float, y: float, yaw: float, *, stamp_s: int | None = None) -> None:
    from geometry_msgs.msg import TransformStamped

    t = TransformStamped()
    t.header.frame_id = "map"
    t.child_frame_id = child
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.rotation.z = math.sin(yaw / 2.0)
    t.transform.rotation.w = math.cos(yaw / 2.0)
    if stamp_s is None:
        node._tf_buffer.set_transform_static(t, "test")
    else:
        t.header.stamp.sec = stamp_s
        node._tf_buffer.set_transform(t, "test")


def _detection(frame_id: str, azimuth_rad: float, *, level_db: float = 50.0, kind: str = "footstep", stamp_s: int = 0):
    from arena_robots_msgs.msg import SoundDetection

    msg = SoundDetection()
    msg.header.frame_id = frame_id
    msg.header.stamp.sec = stamp_s
    msg.kind = kind
    msg.azimuth_rad = azimuth_rad
    msg.elevation_rad = math.nan
    msg.level_db = level_db
    return msg


def _mass_bearing(grid, x: float, y: float) -> float:
    rows, cols = np.nonzero(grid.mass)
    w = grid.mass[rows, cols].astype(np.float64)
    dx = grid.origin_x + (cols + 0.5) * grid.resolution - x
    dy = grid.origin_y + (rows + 0.5) * grid.resolution - y
    return math.atan2(float((w * dy).sum()), float((w * dx).sum()))


def _quat(roll: float, yaw: float):
    from geometry_msgs.msg import Quaternion

    cr, sr, cy, sy = math.cos(roll / 2), math.sin(roll / 2), math.cos(yaw / 2), math.sin(yaw / 2)
    return Quaternion(w=cr * cy, x=sr * cy, y=sr * sy, z=cr * sy)


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, -2.5, 0.99 * math.pi])
@pytest.mark.parametrize("roll", [0.0, 0.5])
def test_yaw_from_quat_recovers_the_heading_under_roll(yaw: float, roll: float) -> None:
    from arena_hearing.belief_node import yaw_from_quat

    assert yaw_from_quat(_quat(roll, yaw)) == pytest.approx(yaw, abs=1e-9)


@pytest.mark.parametrize(("resolution", "cells"), [(0.25, 40), (0.0, 100)])
def test_standalone_layout_spans_the_configured_rectangle(belief, resolution: float, cells: int) -> None:
    node = belief(**_standalone(**{"belief.resolution_m": resolution}))
    layout = node._layout
    assert (layout.origin_x, layout.origin_y, layout.width, layout.height) == (-5.0, -5.0, cells, cells)
    assert layout.free is None
    (robot,) = _bind(node, ("r0", "jackal"))
    assert (robot.grid.width, robot.grid.height, robot.grid.resolution) == (cells, cells, layout.resolution)
    assert robot.grid.free.all()


def _map_msg(width: int, height: int, resolution: float, origin: tuple[float, float], data: np.ndarray):
    from nav_msgs.msg import OccupancyGrid

    msg = OccupancyGrid()
    msg.header.frame_id = "map"
    msg.info.width, msg.info.height, msg.info.resolution = width, height, resolution
    msg.info.origin.position.x, msg.info.origin.position.y = origin
    msg.data = np.asarray(data, dtype=np.int8).reshape(-1).tolist()
    return msg


def test_standalone_node_ignores_the_map(belief) -> None:
    node = belief(**_standalone())
    layout = node._layout
    node._cb_map(_map_msg(4, 4, 0.5, (0.0, 0.0), np.zeros((4, 4))))
    assert node._layout is layout
    assert node._map_info is None


def _occupancy(height: int, width: int) -> np.ndarray:
    data = np.zeros((height, width), dtype=np.int8)
    data[:, ::2] = 100
    data[::3, :] = -1
    data[5, 7] = 49
    data[6, 8] = 50
    return data


def test_map_layout_at_native_resolution_frees_cells_below_the_occupied_threshold(belief) -> None:
    node = belief(**{"belief.markers.enabled": False})
    (robot,) = _bind(node, ("r0", "jackal"))
    assert node._layout is None
    assert robot.grid is None
    data = _occupancy(30, 40)
    node._cb_map(_map_msg(40, 30, 0.1, (1.0, 2.0), data))
    layout = node._layout
    assert (layout.origin_x, layout.origin_y, layout.resolution, layout.width, layout.height) == (1.0, 2.0, pytest.approx(0.1), 40, 30)
    np.testing.assert_array_equal(layout.free, data < 50)
    assert bool(layout.free[5, 7]) and not bool(layout.free[6, 8]) and bool(layout.free[0, 0])
    grid = robot.grid
    assert (grid.width, grid.height) == (40, 30)
    np.testing.assert_array_equal(grid.free, layout.free)

    node._cb_map(_map_msg(40, 30, 0.1, (1.0, 2.0), np.full((30, 40), 100)))
    assert node._layout is layout
    assert robot.grid is grid

    node._cb_map(_map_msg(40, 30, 0.1, (1.5, 2.0), data))
    assert node._layout is not layout
    assert node._layout.origin_x == 1.5
    assert robot.grid is not grid


@pytest.mark.parametrize(
    "map_resolution",
    [0.1, float(np.float32(0.1))],
)
def test_map_layout_at_a_coarser_resolution_samples_a_native_cell_inside_each_coarse_cell(belief, map_resolution: float) -> None:
    node = belief(**{"belief.markers.enabled": False, "belief.resolution_m": 0.2})
    rows, cols = np.mgrid[0:30, 0:40]
    data = np.where(((rows // 2) + (cols // 2)) % 2 == 1, 100, 0)
    node._cb_map(_map_msg(40, 30, map_resolution, (0.0, 0.0), data))
    layout = node._layout
    assert (layout.width, layout.height, layout.resolution) == (20, 15, 0.2)
    coarse_rows, coarse_cols = np.mgrid[0:15, 0:20]
    np.testing.assert_array_equal(layout.free, (coarse_rows + coarse_cols) % 2 == 0)


def test_map_layout_keeps_one_cell_when_the_map_is_finer_than_one_cell(belief) -> None:
    node = belief(**{"belief.markers.enabled": False, "belief.resolution_m": 2.0})
    node._cb_map(_map_msg(3, 2, 0.1, (0.0, 0.0), np.array([[0, 100, 0], [100, 100, 100]])))
    layout = node._layout
    assert (layout.width, layout.height) == (1, 1)
    np.testing.assert_array_equal(layout.free, [[True]])


def test_detection_without_a_grid_is_ignored(belief) -> None:
    node = belief(**{"belief.markers.enabled": False})
    (robot,) = _bind(node, ("r0", "jackal"))
    node._cb_detection(robot, _detection("map", 0.0))
    assert not robot.wedges
    assert (robot.dropped_no_pose, robot.dropped_no_bearing) == (0, 0)
    node._on_timer()
    assert robot.grid is None


@pytest.mark.parametrize(
    ("frame", "azimuth", "expected"),
    [("map", 0.0, 0.0), ("/map", 1.0, 1.0), ("", -2.0, -2.0), ("base", 0.0, 0.5), ("base", -0.3, 0.2), ("mic", 0.4, -0.6)],
)
def test_detection_bearing_is_rotated_by_the_frame_yaw_into_the_map(belief, frame: str, azimuth: float, expected: float) -> None:
    node = belief(**_standalone())
    (robot,) = _bind(node, ("r0", "jackal"))
    base = robot.binding.base_frame
    _tf(node, base, 1.0, -2.0, 0.5)
    mic = f"{base}_mic"
    _tf(node, mic, 1.0, -2.0, -1.0)
    node._cb_detection(robot, _detection({"base": base, "mic": mic}.get(frame, frame), azimuth))
    (wedge,) = robot.wedges
    assert wedge.theta == pytest.approx(expected, abs=1e-9)
    assert (wedge.x, wedge.y) == pytest.approx((1.0, -2.0))
    assert wedge.range_m == 15.0
    assert wedge.kind == "footstep"
    assert robot.grid.n_events == 1
    assert _mass_bearing(robot.grid, 1.0, -2.0) == pytest.approx(expected, abs=0.1)


def test_detection_kind_filter_normalizes_case_and_skips_other_kinds(belief) -> None:
    node = belief(**_standalone())
    (robot,) = _bind(node, ("r0", "jackal"))
    _tf(node, robot.binding.base_frame, 0.0, 0.0, 0.0)
    node._cb_detection(robot, _detection("map", 0.0, kind="alarm"))
    assert robot.grid.n_events == 0
    node._cb_detection(robot, _detection("map", 0.0, kind=" Speech "))
    assert robot.grid.n_events == 1
    assert robot.wedges[0].kind == "speech"


def test_detection_of_any_kind_paints_with_all_kinds(belief) -> None:
    node = belief(**_standalone(**{"belief.kinds": ["all"]}))
    (robot,) = _bind(node, ("r0", "jackal"))
    _tf(node, robot.binding.base_frame, 0.0, 0.0, 0.0)
    node._cb_detection(robot, _detection("map", 0.0, kind="alarm"))
    assert robot.grid.n_events == 1


def test_detection_below_the_minimum_level_is_skipped_and_a_nan_level_paints(belief) -> None:
    node = belief(**_standalone(**{"belief.min_level_db": 30.0}))
    (robot,) = _bind(node, ("r0", "jackal"))
    _tf(node, robot.binding.base_frame, 0.0, 0.0, 0.0)
    node._cb_detection(robot, _detection("map", 0.0, level_db=29.0))
    assert robot.grid.n_events == 0
    node._cb_detection(robot, _detection("map", 0.0, level_db=math.nan))
    node._cb_detection(robot, _detection("map", 0.0, level_db=30.0))
    assert robot.grid.n_events == 2
    assert [w.id for w in robot.wedges] == [0, 1]
    assert robot.next_wedge_id == 2


def test_detection_with_nan_azimuth_counts_as_no_bearing(belief) -> None:
    node = belief(**_standalone())
    (robot,) = _bind(node, ("r0", "jackal"))
    _tf(node, robot.binding.base_frame, 0.0, 0.0, 0.0)
    node._cb_detection(robot, _detection("map", math.nan))
    assert robot.dropped_no_bearing == 1
    assert robot.grid.n_events == 0


def test_detection_without_robot_or_frame_transform_counts_as_no_pose(belief) -> None:
    node = belief(**_standalone())
    (robot,) = _bind(node, ("r0", "jackal"))
    node._cb_detection(robot, _detection("map", 0.0))
    node._cb_detection(robot, _detection("map", 0.0))
    assert robot.dropped_no_pose == 2
    _tf(node, robot.binding.base_frame, 0.0, 0.0, 0.0)
    node._cb_detection(robot, _detection("unknown_array_frame", 0.0))
    assert robot.dropped_no_pose == 3
    assert robot.grid.n_events == 0


def test_lookup_falls_back_to_the_latest_transform_when_the_stamp_is_not_covered(belief) -> None:
    node = belief(**_standalone())
    (robot,) = _bind(node, ("r0", "jackal"))
    _tf(node, robot.binding.base_frame, 2.0, 1.0, 0.25, stamp_s=100)
    from builtin_interfaces.msg import Time

    x, y, yaw = node._lookup(robot.binding.base_frame, Time(sec=1))
    assert (x, y) == pytest.approx((2.0, 1.0))
    assert yaw == pytest.approx(0.25)
    assert node._lookup("nowhere", Time(sec=1)) is None
    node._cb_detection(robot, _detection(robot.binding.base_frame, 0.0, stamp_s=1))
    assert robot.wedges[0].theta == pytest.approx(0.25)


def test_timer_decays_the_mass_by_the_elapsed_time_and_applies_new_parameters(belief) -> None:
    from rclpy.duration import Duration

    node = belief(**_standalone())
    (robot,) = _bind(node, ("r0", "jackal"))
    _tf(node, robot.binding.base_frame, 0.0, 0.0, 0.0)
    node._cb_detection(robot, _detection("map", 0.0))
    before = robot.grid.mass.copy()
    node._last_update = node.get_clock().now() - Duration(seconds=2.0)
    node._on_timer()
    ratio = float(robot.grid.mass.sum() / before.sum())
    assert ratio == pytest.approx(math.exp(-1.0), rel=0.02)

    node.conf.Belief.TAU_S.value = 4.0
    node._on_timer()
    assert robot.grid.config.tau_s == 4.0


def test_as_grid_scales_mass_to_percent_of_full_scale(belief) -> None:
    from builtin_interfaces.msg import Time

    from arena_hearing.belief_grid import BeliefGrid
    from arena_hearing.belief_node import BeliefNode

    node = belief(**_standalone())
    grid = BeliefGrid(-1.0, 3.0, 0.5, 4, 3, node._config)
    full = node._config.full_scale()
    assert full == pytest.approx(1.0 * 2.0 * 2.0)
    grid.mass[1, 2] = 0.5 * full
    grid.mass[2, 3] = 7.0 * full
    msg = BeliefNode._as_grid(grid, Time(sec=9), "map")
    assert (msg.header.frame_id, msg.header.stamp.sec) == ("map", 9)
    assert (msg.info.width, msg.info.height, msg.info.resolution) == (4, 3, 0.5)
    assert (msg.info.origin.position.x, msg.info.origin.position.y, msg.info.origin.orientation.w) == (-1.0, 3.0, 1.0)
    data = np.asarray(msg.data, dtype=np.int8).reshape(3, 4)
    expected = np.zeros((3, 4), dtype=np.int8)
    expected[1, 2], expected[2, 3] = 50, 100
    np.testing.assert_array_equal(data, expected)


def test_reset_clears_every_grid_and_retires_its_wedges(belief) -> None:
    from std_msgs.msg import Bool

    node = belief(**_standalone())
    (robot,) = _bind(node, ("r0", "jackal"))
    _tf(node, robot.binding.base_frame, 0.0, 0.0, 0.0)
    node._cb_detection(robot, _detection("map", 0.0))
    node._cb_detection(robot, _detection("map", 1.0))
    node._cb_reset(Bool(data=False))
    assert len(robot.wedges) == 2
    assert robot.grid.mass.any()
    node._cb_reset(Bool(data=True))
    assert not robot.grid.mass.any()
    assert robot.grid.n_events == 0
    assert not robot.wedges
    assert robot.retired == [0, 1]


def test_reset_skips_a_robot_without_a_grid(belief) -> None:
    from std_msgs.msg import Bool

    node = belief(**{"belief.markers.enabled": False})
    (robot,) = _bind(node, ("r0", "jackal"))
    node._cb_reset(Bool(data=True))
    assert robot.grid is None
    assert robot.retired == []


def _wedge(wid: int, t_ns: int, *, theta: float = 0.0, range_m: float = 15.0, kind: str = "footstep", x: float = 0.0, y: float = 0.0):
    from arena_hearing.belief_node import _Wedge

    return _Wedge(id=wid, t_ns=t_ns, x=x, y=y, theta=theta, range_m=range_m, kind=kind)


def test_retire_wedges_drops_expired_and_surplus_wedges_oldest_first(belief) -> None:
    node = belief(**_standalone(**{"belief.markers.enabled": True, "belief.markers.max_count": 2}))
    (robot,) = _bind(node, ("r0", "jackal"))
    s = 1_000_000_000
    robot.wedges.extend([_wedge(0, 0), _wedge(1, 7 * s), _wedge(2, 8 * s), _wedge(3, 9 * s)])
    node._retire_wedges(robot, 10 * s)
    assert [w.id for w in robot.wedges] == [2, 3]
    assert robot.retired == [0, 1]
    node._retire_wedges(robot, 11 * s + 1)
    assert [w.id for w in robot.wedges] == [3]
    assert robot.retired == [0, 1, 2]


def test_wedge_markers_delete_retired_ids_and_draw_live_wedges_faded_by_age(belief) -> None:
    from builtin_interfaces.msg import Time
    from visualization_msgs.msg import Marker

    node = belief(**_standalone(**{"belief.markers.enabled": True}))
    (robot,) = _bind(node, ("r0", "jackal"))
    s = 1_000_000_000
    robot.retired.extend([7, 8])
    robot.wedges.extend([_wedge(9, 0, theta=0.3, x=1.0, y=2.0), _wedge(10, 2 * s, theta=-1.0, range_m=1.5, kind="ghost")])
    arr = node._wedge_markers(robot, 2 * s, Time(sec=2), "map")
    assert robot.retired == []
    deletes = [m for m in arr.markers if m.action == Marker.DELETE]
    adds = [m for m in arr.markers if m.action == Marker.ADD]
    assert [m.id for m in deletes] == [7, 8]
    assert [m.id for m in adds] == [9, 10]
    assert all(m.ns == "hearing_wedge" and m.header.frame_id == "map" for m in arr.markers)

    old, new = adds
    assert old.type == Marker.TRIANGLE_LIST
    assert len(old.points) == 24
    assert old.color.a == pytest.approx(0.35 * math.exp(-1.0))
    assert (old.color.r, old.color.g, old.color.b) == pytest.approx((1.0, 0.8, 0.25))
    assert new.color.a == pytest.approx(0.35)
    assert (new.color.r, new.color.g, new.color.b) == pytest.approx((0.9, 0.9, 0.9))

    half = math.radians(10.0) / 2
    apex, first, _ = old.points[:3]
    assert (apex.x, apex.y, apex.z) == pytest.approx((1.0, 2.0, 0.05))
    assert (first.x, first.y) == pytest.approx((1.0 + 4.0 * math.cos(0.3 - half), 2.0 + 4.0 * math.sin(0.3 - half)))
    last = old.points[-1]
    assert (last.x, last.y) == pytest.approx((1.0 + 4.0 * math.cos(0.3 + half), 2.0 + 4.0 * math.sin(0.3 + half)))
    assert math.hypot(new.points[1].x, new.points[1].y) == pytest.approx(1.5)


def _spin_until(rclpy, nodes, predicate, timeout_sec: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.02)
        if predicate():
            return
    raise AssertionError("timed out waiting for ROS round-trip")


def test_timer_publishes_wedges_only_while_someone_listens(belief) -> None:
    import rclpy
    from arena_rclpy_mixins import qos
    from visualization_msgs.msg import Marker, MarkerArray

    node = belief(**_standalone(**{"belief.markers.enabled": True}))
    (robot,) = _bind(node, ("r0", "jackal"))
    _tf(node, robot.binding.base_frame, 0.0, 0.0, 0.0)
    robot.retired.append(42)
    node._on_timer()
    assert robot.retired == []

    node._cb_detection(robot, _detection("map", 0.0))
    probe = rclpy.create_node(f"wedge_probe_{uuid.uuid4().hex[:8]}")
    received: list[MarkerArray] = []
    try:
        probe.create_subscription(MarkerArray, robot.pub_markers.publisher.topic_name, received.append, qos.reliable(1))
        _spin_until(rclpy, [probe], lambda: robot.pub_markers.publisher.get_subscription_count() > 0)
        node._on_timer()
        _spin_until(rclpy, [probe], lambda: bool(received))
    finally:
        probe.destroy_node()
    assert [(m.id, m.action) for m in received[-1].markers] == [(0, Marker.ADD)]


def test_robot_leaving_the_fleet_releases_its_topics(belief) -> None:
    node = belief(**_standalone(**{"belief.markers.enabled": True}))
    prefix = f"p{uuid.uuid4().hex[:8]}"
    node._robots._cb_fleet(_fleet(prefix, ("r0", "jackal"), ("r1", "no_such_model")))
    r0, r1 = list(node._robots)
    assert r1.binding.error
    assert r1.binding.base_frame == f"{prefix}/r1/base_link"
    node._robots._cb_fleet(_fleet(prefix, ("r1", "no_such_model")))
    assert list(node._robots) == [r1]
    publishers = list(node.publishers)
    assert r0.pub_belief not in publishers
    assert r0.pub_markers.publisher not in publishers
    assert r0.sub not in list(node.subscriptions)
    assert r1.pub_belief in publishers


def test_robot_without_markers_leaves_only_the_grid_publisher(belief) -> None:
    node = belief(**_standalone())
    prefix = f"p{uuid.uuid4().hex[:8]}"
    node._robots._cb_fleet(_fleet(prefix, ("r0", "jackal")))
    (robot,) = list(node._robots)
    assert robot.pub_markers is None
    node._robots._cb_fleet(_fleet(prefix))
    assert robot.pub_belief not in list(node.publishers)


def test_channel_declares_a_hard_grid_heartbeat_at_the_publish_rate(belief) -> None:
    node = belief(**_standalone(**{"belief.publish_rate_hz": 4.0}))
    (robot,) = _bind(node, ("r0", "jackal"))
    channel = node._channel(robot)
    assert channel.name == "belief/r0"
    assert channel.topic == robot.pub_belief.topic_name
    assert channel.topic.endswith("/r0/hearing/belief_grid")
    assert channel.type == "nav_msgs/msg/OccupancyGrid"
    assert channel.period_s == pytest.approx(1.1 / 4.0)
    assert channel.hard


def test_belief_config_takes_an_emission_parameter_set_away_from_its_default(belief) -> None:
    node = belief(**_standalone())
    declared = node._declared_emission["footstep"]
    assert node._belief_config().emission_db["footstep"] == declared
    node._emission_db["footstep"].value = declared + 12.0
    assert node._belief_config().emission_db["footstep"] == declared + 12.0
    assert node._belief_config().emission_db["speech"] == node._declared_emission["speech"]


def test_world_message_with_a_blank_name_loads_nothing(belief) -> None:
    from std_msgs.msg import String

    node = belief(**_standalone())
    levels = node._world_emission
    node._cb_world(String(data="  "))
    node._loader.submit(lambda: None).result()
    assert node._world_emission is levels


def test_unknown_world_keeps_the_previous_levels(belief) -> None:
    node = belief(**_standalone())
    levels, kinds = node._world_emission, node._kinds
    node._use_world(f"no_such_world_{uuid.uuid4().hex[:6]}")
    assert node._world_emission is levels
    assert node._kinds is kinds


def test_world_message_reloads_kinds_and_levels_of_a_known_world(belief) -> None:
    from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary
    from std_msgs.msg import String

    node = belief(**_standalone())
    levels = node._world_emission
    try:
        node._cb_world(String(data=" map_empty "))
        node._loader.submit(lambda: None).result()
        assert node._world_emission is not levels
        assert node._world_emission["footstep"] == pytest.approx(levels["footstep"])
        assert "footstep" in node._kinds
    finally:
        SoundLibrary.default().use_world(None)
