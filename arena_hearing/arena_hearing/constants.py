"""Topic names of the hearing stack. Names resolve below the task generator node, BELIEF_GRID to COSTMAP_FILTER_INFO below <tg>/<robot>."""

from __future__ import annotations

STATE_ROBOTS = "state/robots"
STATE_RESETTING = "state/resetting"
MAP = "map"

BELIEF_GRID = "hearing/belief_grid"
BELIEF_WEDGES = "hearing/belief_wedges"
SPEED_FILTER_MASK = "hearing/speed_filter_mask"
POLICY_STATE = "hearing/policy_state"
POLICY_MARKERS = "hearing/policy_markers"
COSTMAP_FILTER_INFO = "hearing/costmap_filter_info"


def plan(robot: str) -> str:
    return f"{robot}/plan"


def detections(robot: str, frontend: str) -> str:
    return f"{robot}/hearing/{frontend}/detections"


def hearing_heartbeat(robot: str) -> str:
    return f"{robot}/lockstep/hearing"
