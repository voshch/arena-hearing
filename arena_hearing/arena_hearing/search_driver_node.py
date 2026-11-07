"""Acoustic search driver: per fleet robot, frontier goals on its SLAM map sent to its navigate_to_pose.

Each tick builds a ``search.driver.Tick`` from the robot's own ``<tg>/<robot>/map`` (slam_toolbox under
``robot.mobile.mapping:=slam``), its belief grid, the latest call among ``driver.kinds`` and its global plan,
and turns the driver's ``Send`` / ``Cancel`` into NavigateToPose goals. It runs beside a judge-only task
(``task.robots:=search``), so it is the only writer of the robot's Nav2 goal. The counters land as JSON on
``<tg>/<robot>/hearing/search_state``.
"""

from __future__ import annotations

import functools
import json
import math

import attrs
import numpy as np
import tf2_ros
from action_msgs.msg import GoalStatus
from arena_rclpy_mixins import ArenaMixinNode, qos
from arena_rclpy_mixins.param_groups import configure
from arena_rclpy_mixins.transforms import ThreadedTransformListener
from arena_robots.fleet import RobotBinding
from arena_robots_msgs.msg import SoundDetection
from arena_runtime_msgs.msg import LockstepChannel
from nav2_msgs.action import NavigateToPose
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.task import Future
from rclpy.time import Time
from std_msgs.msg import Bool, String

from arena_hearing.belief_node import yaw_from_quat
from arena_hearing.constants import BELIEF_GRID, MAP, NAVIGATE_TO_POSE, SEARCH_STATE, SPEED_LIMIT, STATE_RESETTING, detections, plan
from arena_hearing.fleet import FleetRobots
from arena_hearing.params import Configuration, GoalRuleName, StopRuleName
from arena_hearing.search.driver import Cancel, DriverConfig, NavStatus, SearchDriver, Send, Tick
from arena_hearing.search.frontiers import MapGrid
from arena_hearing.search.goals import BeliefGoal, BeliefMap, GoalRule, Heard, NearestGoal
from arena_hearing.search.stops import EveryGoalStop, FixedBendStop, NeverStop, OnCallStop, StopRule


@attrs.define(eq=False)
class _Robot:
    binding: RobotBinding
    driver: SearchDriver
    action: str
    client: ActionClient
    pub_state: Publisher
    subs: list[Subscription] = attrs.Factory(list)
    grid: MapGrid | None = None
    belief: BeliefMap | None = None
    plan_xy: np.ndarray | None = None
    heard: Heard | None = None
    heard_t: float = -math.inf
    calls: int = 0
    limited: bool = False
    status: NavStatus = NavStatus.IDLE
    goal_handle: ClientGoalHandle | None = None
    generation: int = 0
    sent_generation: int = 0


class SearchDriverNode(ArenaMixinNode):
    def __init__(self) -> None:
        super().__init__("hearing_search_driver")
        self.conf = Configuration(self)
        self._map_conf = self.conf.Map
        d = self.conf.Driver
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = ThreadedTransformListener(self._tf_buffer)
        self._tg = self.conf.Hearing.TG_NODE.value
        self._frontend = self.conf.Hearing.FRONTEND.value
        self._rate = d.RATE_HZ.value
        self._resetting = False
        self.create_subscription(Bool, f"{self._tg}/{STATE_RESETTING}", self._cb_reset, qos.latched())
        self._robots = FleetRobots(self, self._tg, start=self._start, stop=self._stop, channel=self._channel)
        self.create_timer(1.0 / self._rate, self._on_timer)
        self.get_logger().info(f"hearing_search_driver up: goal {d.GOAL_RULE.value} stop {d.STOP_RULE.value} listen {d.LISTEN_S.value} s snap {d.SNAP.value}")

    def destroy_node(self) -> None:
        self._tf_listener.close()
        super().destroy_node()

    def _goal_rule(self) -> GoalRule:
        d = self.conf.Driver
        if d.GOAL_RULE.value is GoalRuleName.NEAREST:
            return NearestGoal()
        return BeliefGoal(
            radius_m=d.BELIEF_RADIUS_M.value,
            min_mass=d.BELIEF_MIN_MASS.value,
            locate_threshold=d.LOCATE_THRESHOLD.value,
            distance_scale_m=d.DISTANCE_SCALE_M.value,
            snap=d.SNAP.value,
            snap_max_error_rad=math.radians(d.SNAP_MAX_ERROR_DEG.value),
            standoff_m=d.STANDOFF_M.value,
            approach_reach=d.APPROACH_REACH.value,
        )

    def _stop_rule(self) -> StopRule:
        d = self.conf.Driver
        match d.STOP_RULE.value:
            case StopRuleName.NEVER:
                return NeverStop()
            case StopRuleName.EVERY_GOAL:
                return EveryGoalStop()
            case StopRuleName.ON_CALL:
                return OnCallStop()
            case StopRuleName.FIXED_BEND:
                return FixedBendStop(lookahead_m=d.BEND_LOOKAHEAD_M.value, stop_before_m=d.BEND_STOP_BEFORE_M.value, hysteresis_m=d.BEND_HYSTERESIS_M.value)

    def _start(self, binding: RobotBinding) -> _Robot:
        scope = f"{self._tg}/{binding.name}"
        action = self.resolve_topic_name(f"{scope}/{NAVIGATE_TO_POSE}")
        robot = _Robot(
            binding=binding,
            driver=SearchDriver(self._goal_rule(), self._stop_rule(), configure(DriverConfig, self.conf.Driver)),
            action=action,
            client=ActionClient(self, NavigateToPose, action),
            pub_state=self.create_publisher(String, f"{scope}/{SEARCH_STATE}", qos.latched()),
        )
        robot.subs = [
            self.create_subscription(OccupancyGrid, f"{scope}/{MAP}", functools.partial(self._cb_map, robot), qos.latched()),
            self.create_subscription(OccupancyGrid, f"{scope}/{BELIEF_GRID}", functools.partial(self._cb_belief, robot), qos.reliable(1)),
            self.create_subscription(Path, f"{self._tg}/{plan(binding.name)}", functools.partial(self._cb_plan, robot), qos.reliable(1)),
            self.create_subscription(SoundDetection, f"{self._tg}/{detections(binding.name, self._frontend)}", functools.partial(self._cb_detection, robot), qos.reliable(50)),
            self.create_subscription(SpeedLimit, f"{scope}/{SPEED_LIMIT}", functools.partial(self._cb_speed_limit, robot), qos.reliable(1)),
        ]
        self.get_logger().info(f"hearing_search_driver bound to {binding.name}: goals on {robot.action!r}")
        return robot

    def _stop(self, robot: _Robot) -> None:
        self._cancel(robot)
        for sub in robot.subs:
            self.destroy_subscription(sub)
        self.destroy_publisher(robot.pub_state)
        robot.client.destroy()

    def _channel(self, robot: _Robot) -> LockstepChannel:
        return LockstepChannel(name=f"search/{robot.binding.name}", topic=robot.pub_state.topic_name, type="std_msgs/msg/String", period_s=1.1 / self._rate, hard=False)

    def _cb_map(self, robot: _Robot, msg: OccupancyGrid) -> None:
        if self._resetting:
            return
        info = msg.info
        data = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        robot.grid = MapGrid(data=data, origin=(info.origin.position.x, info.origin.position.y), resolution=float(info.resolution), occupied_threshold=self._map_conf.OCCUPIED_THRESHOLD.value)

    @staticmethod
    def _cb_belief(robot: _Robot, msg: OccupancyGrid) -> None:
        info = msg.info
        data = np.asarray(msg.data, dtype=np.float32).reshape(info.height, info.width) / 100.0
        robot.belief = BeliefMap(data=data, origin=(info.origin.position.x, info.origin.position.y), resolution=float(info.resolution))

    @staticmethod
    def _cb_plan(robot: _Robot, msg: Path) -> None:
        robot.plan_xy = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses], dtype=np.float64) if msg.poses else None

    @staticmethod
    def _cb_speed_limit(robot: _Robot, msg: SpeedLimit) -> None:
        robot.limited = 0.0 < msg.speed_limit < (100.0 if msg.percentage else math.inf)

    def _cb_detection(self, robot: _Robot, msg: SoundDetection) -> None:
        azimuth = float(msg.azimuth_rad)
        if msg.kind.strip().lower() not in self.conf.Driver.KINDS.value or not math.isfinite(azimuth):
            return
        pose = self._pose(robot.binding.base_frame)
        frame = msg.header.frame_id.strip("/")
        if frame in ("", self._map_conf.FRAME.value):
            frame_yaw: float | None = 0.0
        elif frame == robot.binding.base_frame:
            frame_yaw = None if pose is None else pose[2]
        else:
            frame_pose = self._pose(frame)
            frame_yaw = None if frame_pose is None else frame_pose[2]
        if pose is None or frame_yaw is None:
            return
        robot.heard = Heard(robot_xy=(pose[0], pose[1]), bearing_rad=azimuth + frame_yaw)
        robot.heard_t = self._now()
        robot.calls += 1

    def _cb_reset(self, msg: Bool) -> None:
        self._resetting = msg.data
        if not msg.data:
            return
        for robot in self._robots:
            self._cancel(robot)
            robot.driver.reset()
            robot.grid = None
            robot.heard, robot.heard_t, robot.calls = None, -math.inf, 0
            robot.plan_xy = None

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _pose(self, frame: str) -> tuple[float, float, float] | None:
        try:
            tf = self._tf_buffer.lookup_transform(self._map_conf.FRAME.value, frame, Time())
        except tf2_ros.TransformException:
            return None
        t = tf.transform.translation
        return float(t.x), float(t.y), yaw_from_quat(tf.transform.rotation)

    def _on_timer(self) -> None:
        for robot in self._robots:
            self._tick(robot)

    def _tick(self, robot: _Robot) -> None:
        pose = self._pose(robot.binding.base_frame)
        now = self._now()
        if pose is not None and robot.client.server_is_ready():
            fresh = now - robot.heard_t <= self.conf.Driver.HEARD_MAX_AGE_S.value
            tick = Tick(now=now, robot_xy=(pose[0], pose[1]), grid=robot.grid, belief=robot.belief, heard=robot.heard if fresh else None, calls=robot.calls, plan_xy=robot.plan_xy, status=robot.status, limited=robot.limited)
            action = robot.driver.step(tick)
            if isinstance(action, Send):
                self._send(robot, action.xy, math.atan2(action.xy[1] - pose[1], action.xy[0] - pose[0]))
            elif isinstance(action, Cancel):
                self._cancel(robot)
        robot.pub_state.publish(String(data=json.dumps({**robot.driver.state(now), "calls": robot.calls, "nav": robot.status.value})))

    def _send(self, robot: _Robot, xy: tuple[float, float], yaw: float) -> None:
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self._map_conf.FRAME.value
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(xy[0]), float(xy[1])
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        robot.generation += 1
        robot.sent_generation = robot.generation
        robot.status = NavStatus.ACTIVE
        robot.client.send_goal_async(goal).add_done_callback(functools.partial(self._on_goal_response, robot, robot.generation))

    def _cancel(self, robot: _Robot) -> None:
        robot.generation += 1
        robot.status = NavStatus.IDLE
        if robot.goal_handle is not None:
            robot.goal_handle.cancel_goal_async()
            robot.goal_handle = None

    def _on_goal_response(self, robot: _Robot, generation: int, future: Future) -> None:
        handle = future.result()
        if not isinstance(handle, ClientGoalHandle):
            return
        if generation != robot.generation:
            if handle.accepted and generation == robot.sent_generation:
                handle.cancel_goal_async()
            return
        if not handle.accepted:
            robot.status = NavStatus.FAILED
            return
        robot.goal_handle = handle
        handle.get_result_async().add_done_callback(functools.partial(self._on_result, robot, generation))

    @staticmethod
    def _on_result(robot: _Robot, generation: int, future: Future) -> None:
        if generation != robot.generation:
            return
        robot.goal_handle = None
        result = future.result()
        if result is None:
            robot.status = NavStatus.FAILED
            return
        status = result.status
        robot.status = NavStatus.SUCCEEDED if status == GoalStatus.STATUS_SUCCEEDED else NavStatus.IDLE if status == GoalStatus.STATUS_CANCELED else NavStatus.FAILED


def main() -> None:
    SearchDriverNode.run_main()
