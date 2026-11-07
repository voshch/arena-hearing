"""Topic names of the hearing stack. Names resolve below the task generator node, BELIEF_GRID to NAVIGATE_TO_POSE below <tg>/<robot>, MAP there too for a robot that maps itself."""

from __future__ import annotations

STATE_ROBOTS = "state/robots"
STATE_RESETTING = "state/resetting"
STATE_WORLD = "state/world"
MAP = "map"

BELIEF_GRID = "hearing/belief_grid"
BELIEF_WEDGES = "hearing/belief_wedges"
SPEED_FILTER_MASK = "hearing/speed_filter_mask"
POLICY_STATE = "hearing/policy_state"
POLICY_MARKERS = "hearing/policy_markers"
COSTMAP_FILTER_INFO = "hearing/costmap_filter_info"
SPEED_LIMIT = "hearing/speed_limit"
SEARCH_STATE = "hearing/search_state"
NAVIGATE_TO_POSE = "navigate_to_pose"

type RecordedTopicRow = tuple[str, str, str, bool, bool, bool, bool, int, bool]


def plan(robot: str) -> str:
    return f"{robot}/plan"


def detections(robot: str, frontend: str) -> str:
    return f"{robot}/hearing/{frontend}/detections"


def hearing_heartbeat(robot: str) -> str:
    return f"{robot}/lockstep/hearing"


def recorded_topics(frontends: tuple[str, ...]) -> tuple[RecordedTopicRow, ...]:
    """Recordable topics as (key, topic template, msg type, robot scoped, throttled, transient local, reliable, depth, recorded)."""
    return (
        *((f"hearing_{frontend}_detections", detections("{ns}", frontend), "arena_robots_msgs/msg/SoundDetection", True, False, False, True, 0, True) for frontend in frontends),
        ("hearing_policy_state", f"{{ns}}/{POLICY_STATE}", "std_msgs/msg/String", True, False, False, True, 0, True),
        ("hearing_speed_limit", f"{{ns}}/{SPEED_LIMIT}", "nav2_msgs/msg/SpeedLimit", True, False, False, True, 0, True),
        ("hearing_search_state", f"{{ns}}/{SEARCH_STATE}", "std_msgs/msg/String", True, False, True, True, 0, True),
        ("hearing_slam_map", f"{{ns}}/{MAP}", "nav_msgs/msg/OccupancyGrid", True, False, True, True, 0, True),
    )
