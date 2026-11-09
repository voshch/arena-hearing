"""Call front-end of every fleet robot: band-flux call detection with a drivetrain ego gate and a whitened SRP heading in the map frame, published as ``kind="onset"`` detections."""

from __future__ import annotations

import functools
import math
from collections import deque
from pathlib import Path

import attrs
import numpy as np
import tf2_ros
from arena_rclpy_mixins import ArenaMixinNode, qos
from arena_rclpy_mixins.transforms import ThreadedTransformListener
from arena_robots.audio import NS_PER_S, ArrayStream, Vec3, array_stream, dbfs_from_rms, dbfs_to_spl
from arena_robots.fleet import RobotBinding
from arena_robots_msgs.msg import AudioFrame, SoundDetection
from arena_runtime_msgs.msg import LockstepChannel, LockstepHeartbeat
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.time import Time

from arena_hearing.belief_node import yaw_from_quat
from arena_hearing.call import SAMPLE_RATE_HZ, BlockStream, Calibration, Detector, Onset, default_calibration, load_calibration
from arena_hearing.constants import detections, hearing_heartbeat
from arena_hearing.doa import frame_positions
from arena_hearing.fleet import HEARTBEAT_TYPE, REFUSAL_LOG_EVERY, FleetRobots, heartbeat
from arena_hearing.params import Configuration, Frontend
from arena_hearing.srp_frontend_node import ONSET_KIND, onset_confidence

HEARTBEAT_PERIOD_S = 0.1


@attrs.define(eq=False)
class _Pipeline:
    detector: Detector
    raw: BlockStream
    sensitivity_dbfs_at_94_dbspl: float
    motor: BlockStream | None = None


@attrs.define(eq=False)
class _Robot:
    binding: RobotBinding
    pub: Publisher
    tick_pub: Publisher
    sub: Subscription = attrs.field(init=False)
    motor_sub: Subscription = attrs.field(init=False)
    layout: tuple[int, int, tuple[Vec3, ...], float] | None = None
    pipeline: _Pipeline | None = None
    refusal: str = ""
    frame_id: str = ""
    yaw: float = 0.0
    poses: deque[tuple[float, float, float]] = attrs.Factory(deque)
    moving: bool = False
    n_events: int = 0
    n_no_bearing: int = 0
    bad_frames: int = 0
    refused: int = 0


class CallFrontendNode(ArenaMixinNode):
    def __init__(self) -> None:
        super().__init__("call_frontend")
        self.conf = Configuration(self)
        self._audio_qos = qos.reliable(200) if self.conf.Audio.RELIABLE_ENABLED.value else qos.best_effort(200)
        c = self.conf.Call
        self._moving_mps = c.MOVING_MPS.value
        self._speed_window_s = c.SPEED_WINDOW_S.value
        self._calibration: Calibration = load_calibration(Path(c.CALIB_FILE.value)) if c.CALIB_FILE.value else default_calibration()
        self._map_frame = self.conf.Map.FRAME.value
        self._tg = self.conf.Hearing.TG_NODE.value
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = ThreadedTransformListener(self._tf_buffer)
        self._robots = FleetRobots(self, self._tg, start=self._start, stop=self._stop, channel=self._channel)
        self.create_timer(self.conf.Diagnostics.PERIOD_S.value, self._diag)
        self.get_logger().info(f"call_frontend up: thresholds {self._calibration.thresholds}, whitened {self._calibration.whiteners is not None}")

    def destroy_node(self) -> None:
        self._tf_listener.close()
        super().destroy_node()

    def _start(self, binding: RobotBinding) -> _Robot:
        robot = _Robot(
            binding=binding,
            pub=self.create_publisher(SoundDetection, f"{self._tg}/{detections(binding.name, Frontend.CALL)}", qos.reliable(50)),
            tick_pub=self.create_publisher(LockstepHeartbeat, f"{self._tg}/{hearing_heartbeat(binding.name)}", qos.reliable(10)),
        )
        robot.sub = self.create_subscription(AudioFrame, f"{self._tg}/{array_stream(binding.name, ArrayStream.RAW)}", functools.partial(self._cb_audio, robot), self._audio_qos)
        robot.motor_sub = self.create_subscription(AudioFrame, f"{self._tg}/{array_stream(binding.name, ArrayStream.STEM_MOTOR)}", functools.partial(self._cb_motor, robot), self._audio_qos)
        self.get_logger().info(f"call_frontend: audio {robot.sub.topic_name!r} and {robot.motor_sub.topic_name!r} -> {robot.pub.topic_name!r}")
        return robot

    def _stop(self, robot: _Robot) -> None:
        self.destroy_subscription(robot.sub)
        self.destroy_subscription(robot.motor_sub)
        self.destroy_publisher(robot.pub)
        self.destroy_publisher(robot.tick_pub)

    def _channel(self, robot: _Robot) -> LockstepChannel:
        return LockstepChannel(name=f"hearing/{robot.binding.name}", topic=robot.tick_pub.topic_name, type=HEARTBEAT_TYPE, period_s=HEARTBEAT_PERIOD_S, hard=True)

    def _pipeline(self, fs: int, channels: int, positions: tuple[Vec3, ...], sensitivity_dbfs_at_94_dbspl: float) -> _Pipeline:
        """Raises ValueError when the stream cannot feed the detector."""
        if fs != SAMPLE_RATE_HZ:
            raise ValueError(f"stream is {fs} Hz, the detector needs {SAMPLE_RATE_HZ} Hz")
        if len(positions) != channels:
            raise ValueError(f"frame declares {len(positions)} microphone positions for {channels} channels")
        if sensitivity_dbfs_at_94_dbspl == 0.0:
            raise ValueError("frame declares no microphone sensitivity")
        return _Pipeline(Detector(np.asarray(positions, dtype=np.float64), self._calibration), BlockStream(channels), sensitivity_dbfs_at_94_dbspl)

    def _lookup(self, frame: str, stamp: TimeMsg) -> tuple[float, float, float] | None:
        """(x, y, yaw) of frame in the map frame at stamp, else at the latest transform."""
        for when in (Time.from_msg(stamp), Time()):
            try:
                tf = self._tf_buffer.lookup_transform(self._map_frame, frame, when)
            except tf2_ros.TransformException:
                continue
            t = tf.transform.translation
            return float(t.x), float(t.y), yaw_from_quat(tf.transform.rotation)
        return None

    def _track(self, robot: _Robot, msg: AudioFrame, stamp_s: float) -> None:
        array_pose = self._lookup(msg.header.frame_id, msg.header.stamp)
        if array_pose is not None:
            robot.yaw = array_pose[2]
        base_pose = self._lookup(robot.binding.base_frame, msg.header.stamp)
        if base_pose is None:
            return
        robot.poses.append((stamp_s, base_pose[0], base_pose[1]))
        while robot.poses and stamp_s - robot.poses[0][0] > self._speed_window_s:
            robot.poses.popleft()
        first, last = robot.poses[0], robot.poses[-1]
        robot.moving = last[0] > first[0] and math.hypot(last[1] - first[1], last[2] - first[2]) / (last[0] - first[0]) > self._moving_mps

    def _cb_motor(self, robot: _Robot, msg: AudioFrame) -> None:
        pipe = robot.pipeline
        ch, n = int(msg.channel_count), int(msg.frame_count)
        data = np.asarray(msg.data, dtype=np.float32)
        if pipe is None or int(msg.sample_rate) != SAMPLE_RATE_HZ or data.size != ch * n:
            return
        if pipe.motor is None:
            pipe.motor = BlockStream(ch)
        block = data.reshape(n, ch) if msg.interleaved else data.reshape(ch, n).T
        for stamp_ns, chunk in pipe.motor.push(msg.header.stamp.sec * NS_PER_S + msg.header.stamp.nanosec, block):
            pipe.detector.push_motor(stamp_ns, chunk)

    def _cb_audio(self, robot: _Robot, msg: AudioFrame) -> None:
        fs, ch, n = int(msg.sample_rate), int(msg.channel_count), int(msg.frame_count)
        data = np.asarray(msg.data, dtype=np.float32)
        if fs <= 0 or data.size != ch * n:
            robot.bad_frames += 1
            return
        robot.frame_id = msg.header.frame_id
        stamp_ns = msg.header.stamp.sec * NS_PER_S + msg.header.stamp.nanosec
        layout = (fs, ch, frame_positions(msg), float(msg.sensitivity_dbfs_at_94_dbspl))
        if layout != robot.layout:
            robot.layout, robot.refused = layout, 0
            try:
                robot.pipeline, robot.refusal = self._pipeline(*layout), ""
                self.get_logger().info(f"call_frontend {robot.binding.name}: {ch} microphones at {fs} Hz")
            except ValueError as exc:
                robot.pipeline, robot.refusal = None, str(exc)
        pipe = robot.pipeline
        if pipe is None:
            robot.refused += 1
            if robot.refused % REFUSAL_LOG_EVERY == 1:
                self.get_logger().error(f"call_frontend {robot.binding.name}: {robot.refusal}, {robot.refused} frames refused")
            robot.tick_pub.publish(heartbeat(stamp_ns + n * NS_PER_S // fs, robot.frame_id))
            return
        self._track(robot, msg, stamp_ns / NS_PER_S)
        block = data.reshape(n, ch) if msg.interleaved else data.reshape(ch, n).T
        for block_ns, chunk in pipe.raw.push(stamp_ns, block):
            for onset in pipe.detector.push(block_ns, chunk, robot.moving, robot.yaw):
                self._publish(robot, pipe, onset)
        robot.tick_pub.publish(heartbeat(pipe.raw.end_ns, robot.frame_id))

    def _publish(self, robot: _Robot, pipe: _Pipeline, onset: Onset) -> None:
        if not math.isfinite(onset.heading_rad):
            robot.n_no_bearing += 1
            return
        msg = SoundDetection()
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(onset.stamp_ns, NS_PER_S)
        msg.header.frame_id = self._map_frame
        msg.robot = robot.binding.name
        msg.frontend = Frontend.CALL.value
        msg.kind = ONSET_KIND
        msg.event_id = f"call:{robot.n_events}"
        msg.azimuth_rad = onset.heading_rad
        msg.elevation_rad = math.nan
        msg.level_db = dbfs_to_spl(dbfs_from_rms(onset.peak_rms), pipe.sensitivity_dbfs_at_94_dbspl)
        msg.confidence = onset_confidence(onset.stat_db - pipe.detector.thresholds[onset.state])
        robot.pub.publish(msg)
        robot.n_events += 1

    def _diag(self) -> None:
        for robot in self._robots:
            frames = robot.pipeline.detector.frames if robot.pipeline is not None else 0
            self.get_logger().debug(f"{robot.binding.name}: frames {frames} calls {robot.n_events} no-bearing {robot.n_no_bearing} bad frames {robot.bad_frames} refused {robot.refused}")


def main() -> None:
    CallFrontendNode.run_main()
