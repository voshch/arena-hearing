from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from arena_hearing.constants import BELIEF_GRID, COSTMAP_FILTER_INFO, POLICY_STATE, SPEED_FILTER_MASK, detections
from arena_mcp.core import LATCHED, RELIABLE, ROS, Domain, ToolError, media, msgs, robot_ns

if TYPE_CHECKING:
    from nav_msgs.msg import OccupancyGrid

    from arena_mcp.core.ros import TopicCache

domain = Domain(
    "hearing",
    """Robot hearing (robot.hearing:=bus|srp|seld): sound detections per front-end, the belief grid and the speed-filter policy of each robot.
    Every tool takes env_id and robot. Detections are watched from the first call on, so a first call waits up to window_s while it collects.""",
    requires=("arena_hearing", "arena_robots_msgs", "nav_msgs"),
)

NO_HEARING = "the env runs with robot.hearing:=none, or its hearing nodes are not up"
DEPTH = 4096
MAX_WAIT_S = 10.0

_watched: dict[str, float] = {}


async def _watch(msg_type: type, topic: str) -> tuple[TopicCache, float]:
    cache = await ROS.cache(msg_type, topic, qos=RELIABLE, depth=DEPTH)
    return cache, _watched.setdefault(topic, time.monotonic())


def _grid(grid: OccupancyGrid) -> dict[str, Any]:
    import numpy as np

    info = grid.info
    cells = np.asarray(grid.data, dtype=np.int16).reshape(info.height, info.width)
    out: dict[str, Any] = {
        "frame": grid.header.frame_id,
        "stamp": msgs.seconds(grid.header.stamp),
        "resolution_m": info.resolution,
        "width": int(info.width),
        "height": int(info.height),
        "origin": msgs.pose_dict(info.origin),
        "nonzero_cells": int((cells > 0).sum()),
        "max": int(cells.max()) if cells.size else 0,
    }
    if out["max"] > 0:
        row, col = np.unravel_index(int(np.argmax(cells)), cells.shape)
        out["max_at"] = {"x": info.origin.position.x + (col + 0.5) * info.resolution, "y": info.origin.position.y + (row + 0.5) * info.resolution}
    return out


@domain.tool(read_only=True)
async def hearing_detections(
    env_id: int | None = None,
    robot: str | None = None,
    frontend: Annotated[str | None, Field(description="bus, srp or seld, default every front-end publishing for the robot")] = None,
    window_s: Annotated[float, Field(description="sim seconds back from the newest detection or the sim clock")] = 5.0,
) -> dict[str, Any]:
    """SoundDetections of a robot over the last window_s, per front-end."""
    from arena_robots_msgs.msg import SoundDetection

    from arena_hearing.params import Frontend

    if window_s <= 0.0:
        raise ToolError("window_s must be positive")
    if frontend is not None and frontend not in {item.value for item in Frontend}:
        raise ToolError(f"unknown frontend {frontend!r}", hint=f"frontends: {', '.join(item.value for item in Frontend)}")
    ns = await robot_ns(env_id, robot)
    topics = {item.value: detections(ns, item.value) for item in Frontend if frontend in (None, item.value)}
    published = set((await ROS.graph())["topics"])
    live = {name: topic for name, topic in topics.items() if topic in published}
    if not live:
        raise ToolError(f"no detections topic under {ns}/hearing for {frontend or 'any front-end'}", hint=NO_HEARING)
    watches = {name: await _watch(SoundDetection, topic) for name, topic in live.items()}
    began = max(start for _, start in watches.values())
    if time.monotonic() - began < window_s:
        await asyncio.sleep(min(window_s - (time.monotonic() - began), MAX_WAIT_S))
    now = (await ROS.node()).sim_time.to_seconds()
    out: dict[str, Any] = {"robot_ns": ns, "window_s": window_s, "watched_s": round(time.monotonic() - min(start for _, start in watches.values()), 1), "detections": {}}
    for name, (cache, _) in watches.items():
        received = list(cache.history)
        stamps = [msgs.seconds(msg.header.stamp) for msg in received]
        reference = now if now > 0.0 else max(stamps, default=0.0)
        out["detections"][name] = [{"stamp": stamp, **msgs.to_dict(msg)} for msg, stamp in zip(received, stamps, strict=True) if reference - window_s <= stamp <= reference + 1.0]
    return out


@domain.tool(read_only=True)
async def hearing_belief(env_id: int | None = None, robot: str | None = None) -> list[Any]:
    """The robot's pedestrian belief grid (0..100 per cell, darker is likelier) as a summary and an image."""
    from nav_msgs.msg import OccupancyGrid

    ns = await robot_ns(env_id, robot)
    grid = await ROS.latest(OccupancyGrid, f"{ns}/{BELIEF_GRID}", qos=RELIABLE, timeout=5.0, hint=NO_HEARING)
    return [{"robot_ns": ns, **_grid(grid)}, media.occupancy_png(grid)]


@domain.tool(read_only=True)
async def hearing_policy(env_id: int | None = None, robot: str | None = None) -> list[Any]:
    """The speed-filter policy of a robot: its state, the costmap filter info and the speed mask (cell = speed limit percent, 0 = no limit) as an image."""
    import numpy as np
    from nav2_msgs.msg import CostmapFilterInfo
    from nav_msgs.msg import OccupancyGrid
    from std_msgs.msg import String

    ns = await robot_ns(env_id, robot)
    state = await ROS.latest(String, f"{ns}/{POLICY_STATE}", qos=LATCHED, timeout=3.0, hint=NO_HEARING)
    mask = await ROS.latest(OccupancyGrid, f"{ns}/{SPEED_FILTER_MASK}", qos=LATCHED, timeout=3.0, hint=NO_HEARING)
    info = await ROS.latest(CostmapFilterInfo, f"{ns}/{COSTMAP_FILTER_INFO}", qos=LATCHED, timeout=3.0, hint=NO_HEARING)
    values, counts = np.unique(np.asarray(mask.data, dtype=np.int16), return_counts=True)
    summary = {"robot_ns": ns, "state": json.loads(state.data), "filter_info": msgs.to_dict(info), "mask": {**_grid(mask), "limit_pct_cells": {str(int(value)): int(count) for value, count in zip(values, counts, strict=True)}}}
    return [summary, media.occupancy_png(mask)]
