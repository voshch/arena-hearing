from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import math
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("arena_mcp")
rclpy = pytest.importorskip("rclpy")
pytest.importorskip("scipy")
pytest.importorskip("nav2_msgs.msg")
pytest.importorskip("tf2_msgs.msg")
pytest.importorskip("arena_robots_msgs.msg")
pytest.importorskip("arena_runtime_msgs.msg")
pytest.importorskip("task_generator_msgs.msg")

from arena_mcp.core import ROS, ToolError

from arena_hearing import mcp as hearing

ROBOT = "r0"
STARTUP_S = 60.0
SIZE, RESOLUTION, ORIGIN = 100, 0.1, -5.0


@dataclasses.dataclass
class Rig:
    loop: asyncio.AbstractEventLoop
    tg: str
    peer: object
    publishers: dict
    logs: list[Path]

    def run(self, coro):
        return self.loop.run_until_complete(coro)

    def log_tail(self) -> str:
        return "\n".join(log.read_text()[-3000:] for log in self.logs)


def _param_text(value: object) -> str:
    return ("true" if value else "false") if isinstance(value, bool) else str(value)


@contextlib.contextmanager
def _process(module: str, ns: str, params: dict[str, object], log: Path):
    args = [sys.executable, "-c", f"from arena_hearing.{module} import main; main()", "--ros-args", "-r", f"__ns:={ns}", "-r", f"/tf:={ns}/tf", "-r", f"/tf_static:={ns}/tf_static"]
    for name, value in params.items():
        args += ["-p", f"{name}:={_param_text(value)}"]
    with log.open("w") as out:
        proc = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            yield proc
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10.0)


def _grid():
    from nav_msgs.msg import OccupancyGrid

    msg = OccupancyGrid()
    msg.header.frame_id = "map"
    msg.info.resolution = RESOLUTION
    msg.info.width = msg.info.height = SIZE
    msg.info.origin.position.x = msg.info.origin.position.y = ORIGIN
    msg.info.origin.orientation.w = 1.0
    msg.data = [0] * (SIZE * SIZE)
    return msg


def _detection(rig: Rig, frontend: str):
    from arena_robots_msgs.msg import SoundDetection

    msg = SoundDetection(robot=ROBOT, frontend=frontend, kind="footstep", azimuth_rad=0.0, elevation_rad=math.nan, level_db=50.0, confidence=1.0)
    msg.header.frame_id = "map"
    msg.header.stamp = rig.peer.get_clock().now().to_msg()
    return msg


@pytest.fixture(scope="module")
def rig(tmp_path_factory, default_sounds):
    from arena_rclpy_mixins import qos
    from arena_robots.fleet import RobotBinding
    from arena_robots_msgs.msg import SoundDetection
    from arena_runtime_msgs.msg import EnvRecord, EnvRegistry
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid
    from rclpy.executors import SingleThreadedExecutor
    from task_generator_msgs.msg import RobotDescriptor, RobotFleet, RobotState
    from tf2_msgs.msg import TFMessage

    ns = f"/t_{uuid.uuid4().hex[:8]}"
    tg = f"{ns}/task_generator_node"
    prefix = f"{ns.strip('/')}/{ROBOT}"
    tmp = tmp_path_factory.mktemp("hearing")
    logs = [tmp / "belief_node.log", tmp / "policy_node.log"]
    peer = rclpy.create_node(f"hearing_peer_{uuid.uuid4().hex[:8]}")
    publishers = {
        "envs": peer.create_publisher(EnvRegistry, "/arena/state/envs", qos.latched()),
        "robots": peer.create_publisher(RobotFleet, f"{tg}/state/robots", qos.latched()),
        "tf": peer.create_publisher(TFMessage, f"{ns}/tf_static", qos.latched()),
        "map": peer.create_publisher(OccupancyGrid, f"{tg}/map", qos.latched()),
        "bus": peer.create_publisher(SoundDetection, f"{tg}/{ROBOT}/hearing/bus/detections", qos.reliable(50)),
        "srp": peer.create_publisher(SoundDetection, f"{tg}/{ROBOT}/hearing/srp/detections", qos.reliable(50)),
    }
    publishers["envs"].publish(EnvRegistry(envs=[EnvRecord(env_id=0, fqn=tg, ready=True)]))
    base_frame = RobotBinding.resolve(name=ROBOT, model="jackal", namespace=f"{tg}/{ROBOT}", frame_prefix=prefix).base_frame
    publishers["robots"].publish(RobotFleet(robots=[RobotState(descriptor=RobotDescriptor(name=ROBOT, model="jackal", ns=f"{tg}/{ROBOT}", frame=prefix, base_frame=base_frame))]))
    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.child_frame_id = base_frame
    transform.transform.rotation.w = 1.0
    publishers["tf"].publish(TFMessage(transforms=[transform]))
    publishers["map"].publish(_grid())
    executor = SingleThreadedExecutor()
    executor.add_node(peer)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    loop = asyncio.new_event_loop()
    belief_params = {"belief.markers.enabled": False}
    policy_params = {"policy.max_linear_mps": 1.0, "policy.listen.enabled": False, "policy.yield.enabled": False}
    with _process("belief_node", ns, belief_params, logs[0]), _process("policy_node", ns, policy_params, logs[1]):
        try:
            yield Rig(loop=loop, tg=tg, peer=peer, publishers=publishers, logs=logs)
        finally:
            executor.shutdown()
            peer.destroy_node()
            loop.run_until_complete(ROS.shutdown())
            loop.close()
            if not rclpy.ok():
                rclpy.init()


async def _until(rig: Rig, predicate, what: str, timeout_s: float = STARTUP_S) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}:\n{rig.log_tail()}")
        await asyncio.sleep(0.1)


async def _later(delay_s: float, action) -> None:
    await asyncio.sleep(delay_s)
    action()


async def _paint(rig: Rig, until: float) -> None:
    while time.monotonic() < until:
        rig.publishers["bus"].publish(_detection(rig, "bus"))
        await asyncio.sleep(0.2)


def test_hearing_domain_is_served_from_its_entry_point():
    from arena_mcp.server import discover

    (loaded,) = discover(["hearing"])
    assert loaded.served, loaded.reason
    assert {spec.name for spec in loaded.domain.tools} == {"hearing_detections", "hearing_belief", "hearing_policy"}


def test_detections_belief_and_policy_of_live_hearing_nodes(rig: Rig):
    async def scenario():
        await _until(rig, lambda: rig.publishers["bus"].get_subscription_count() >= 2, "hearing_belief and hearing_policy to bind r0")
        srp = _detection(rig, "srp")
        detections, _, _ = await asyncio.gather(
            hearing.hearing_detections(window_s=3.0),
            _paint(rig, time.monotonic() + 3.0),
            _later(1.0, lambda: rig.publishers["srp"].publish(srp)),
        )
        belief = None
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            await _paint(rig, time.monotonic() + 0.5)
            belief = await hearing.hearing_belief(robot=ROBOT)
            if belief[0]["max"] > 0:
                break
        policy = await hearing.hearing_policy()
        with pytest.raises(ToolError, match="no detections topic"):
            await hearing.hearing_detections(frontend="seld")
        with pytest.raises(ToolError, match="unknown frontend"):
            await hearing.hearing_detections(frontend="ears")
        with pytest.raises(ToolError, match="no robot"):
            await hearing.hearing_belief(robot="ghost")
        return detections, srp, belief, policy

    detections, srp, belief, policy = rig.run(scenario())
    assert set(detections["detections"]) == {"bus", "srp"}
    assert detections["detections"]["bus"]
    assert all(entry["kind"] == "footstep" for entry in detections["detections"]["bus"])
    assert [entry["frontend"] for entry in detections["detections"]["srp"]] == ["srp"]
    summary, image = belief
    assert summary["robot_ns"] == f"{rig.tg}/{ROBOT}"
    assert (summary["width"], summary["height"]) == (SIZE, SIZE)
    assert summary["max"] > 0
    assert summary["max_at"]["x"] > 0.0
    assert image.to_image_content().mime_type == "image/png"
    state, mask = policy
    assert state["state"]["state"] == "cruise"
    assert state["filter_info"]["filter_mask_topic"] == f"{rig.tg}/{ROBOT}/hearing/speed_filter_mask"
    assert sum(state["mask"]["limit_pct_cells"].values()) == SIZE * SIZE
    assert mask.to_image_content().mime_type == "image/png"
