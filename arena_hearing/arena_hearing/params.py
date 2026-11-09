"""Every hearing node parameter, grouped. Defaults live only here."""

from __future__ import annotations

import enum
import functools
import typing
from collections.abc import Mapping

from arena_rclpy_mixins.params import Param, ParamGroup, ParamHandle, Range, declare, floats, names

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer


class Frontend(enum.StrEnum):
    BUS = "bus"
    SRP = "srp"
    SELD = "seld"
    CALL = "call"

    @property
    def array_spec(self) -> str:
        """Array spec the front-end runs on unless auditory.array.spec names one, empty for the bus."""
        return "" if self is Frontend.BUS else "four_mic"


class BearingSource(enum.StrEnum):
    GCC = "gcc"
    SELD = "seld"


class MapGroup(ParamGroup):
    FRAME = Param[str]("map.frame", "map", "Frame of the occupancy map.")
    OCCUPIED_THRESHOLD = Param[int]("map.occupied_threshold", 50, "Occupancy grid value from which a cell counts as occupied.", constraint=Range(lo=0))


class DiagnosticsGroup(ParamGroup):
    PERIOD_S = Param[float]("diagnostics.period_s", 5.0, "Seconds between diagnostics log lines.", constraint=Range(lo=0.0, lo_open=True))
    REPORT_PERIOD_S = Param[float]("diagnostics.report_period_s", 0.5, "Seconds between diagnostics reports on the topic.", constraint=Range(lo=0.0, lo_open=True))
    ACTIVITY_THRESHOLD_DBFS = Param[float]("diagnostics.activity_threshold_dbfs", -70.0, "Level in dBFS above which a channel counts as active.")


class HearingGroup(ParamGroup):
    FRONTEND = Param[Frontend]("hearing.frontend", Frontend.BUS, "Detection front-end, bus, srp or seld, set by the launch file.")
    TG_NODE = Param[str]("hearing.tg_node", "arena_env_node", "Task generator node name the auditory topics live below, set by the launch file.")


class BeliefGroup(ParamGroup):
    MARKERS_ENABLED = Param[bool]("belief.markers.enabled", True, "Publish belief markers.")
    MARKERS_RANGE_M = Param[float]("belief.markers.range_m", 4.0, "Range in m within which belief markers are drawn.", constraint=Range(lo=0.0, lo_open=True))
    MARKERS_MAX_COUNT = Param[int]("belief.markers.max_count", 12, "Most belief markers published at once.", constraint=Range(lo=0))
    PUBLISH_RATE_HZ = Param[float]("belief.publish_rate_hz", 5.0, "Belief grid publish rate in Hz.", constraint=Range(lo=0.0, lo_open=True))
    RESOLUTION_M = Param[float]("belief.resolution_m", 0.0, "Belief grid cell size in m (0 = the env map resolution).", constraint=Range(lo=0.0))
    STANDALONE_ENABLED = Param[bool]("belief.standalone.enabled", False, "Run the belief grid on its own grid instead of the env map.")
    STANDALONE_ORIGIN_M = Param[tuple[float, ...]]("belief.standalone.origin_m", (-10.0, -10.0), "Standalone grid origin x y in m.", parse=floats)
    STANDALONE_SIZE_M = Param[tuple[float, ...]]("belief.standalone.size_m", (0.0, 0.0), "Standalone grid size in m, 0 = the extent of the env map, whose cells it never reads.", parse=floats)
    WEDGE_DEG = Param[float]("belief.wedge_deg", 10.0, "Opening angle in degrees of the wedge a bearing paints.", constraint=Range(lo=0.0, lo_open=True))
    MIN_HALF_WIDTH_M = Param[float]("belief.min_half_width_m", 0.3, "Narrowest half width in m of a bearing wedge.", constraint=Range(lo=0.0))
    TAU_S = Param[float]("belief.tau_s", 2.0, "Decay time constant in s of belief mass.", constraint=Range(lo=0.0, lo_open=True))
    MAX_RANGE_M = Param[float]("belief.max_range_m", 15.0, "Longest range in m a bearing paints.", constraint=Range(lo=0.0, lo_open=True))
    MIN_RANGE_M = Param[float]("belief.min_range_m", 0.5, "Shortest range in m a bearing paints.", constraint=Range(lo=0.0))
    REFERENCE_DISTANCE_M = Param[float]("belief.reference_distance_m", 1.0, "Distance in m at which emission levels are referenced.", constraint=Range(lo=0.0, lo_open=True))
    LEVEL_RANGE_ENABLED = Param[bool]("belief.level_range.enabled", False, "Estimate the range of a bearing from the received level.")
    RANGE_SIGMA_FRAC = Param[float]("belief.range_sigma_frac", 0.35, "Relative width of the level range estimate.", constraint=Range(lo=0.0, lo_open=True))
    RANGE_FLOOR_WEIGHT = Param[float]("belief.range_floor_weight", 0.15, "Weight a bearing keeps outside its level range estimate.", constraint=Range(lo=0.0))
    EVENT_MASS = Param[float]("belief.event_mass", 1.0, "Mass one detection adds to the belief grid.", constraint=Range(lo=0.0, lo_open=True))
    EVENT_RATE_HZ = Param[float]("belief.event_rate_hz", 2.0, "Detection rate in Hz of the front-end, defaults per front-end in the launch file.", constraint=Range(lo=0.0, lo_open=True))
    MASS_FULL_SCALE = Param[float]("belief.mass_full_scale", 0.0, "Mass that maps to full belief (0 = the largest cell mass).", constraint=Range(lo=0.0))
    KINDS = Param[tuple[str, ...]]("belief.kinds", ("footstep", "speech"), "Detection kinds the belief grid accepts.", parse=names)
    MIN_LEVEL_DB = Param[float]("belief.min_level_db", -1e9, "Level in dB below which a detection is ignored.")

    EMISSION_DB_PREFIX: typing.ClassVar[str] = "belief.emission_db."

    def emission_db(self, levels: Mapping[str, float]) -> dict[str, ParamHandle[float]]:
        """Declare belief.emission_db.<kind> per kind, defaulting to the level of the kind's default asset, NaN for unknown."""
        return {kind: declare(self._server, Param[float](f"{self.EMISSION_DB_PREFIX}{kind}", float(level), f"Emission level of {kind} in dB.")) for kind, level in levels.items()}


class PolicyGroup(ParamGroup):
    PUBLISH_RATE_HZ = Param[float]("policy.publish_rate_hz", 10.0, "Policy mask publish rate in Hz.", constraint=Range(lo=0.0, lo_open=True))
    MAX_LINEAR_MPS = Param[float]("policy.max_linear_mps", 0.0, "Speed cap in m/s of the policy (0 = the robot's own limit).", constraint=Range(lo=0.0))
    LISTEN_ENABLED = Param[bool]("policy.listen.enabled", True, "Cap the speed near blind corners so the robot can listen.")
    YIELD_ENABLED = Param[bool]("policy.yield.enabled", True, "Hold the robot back when a source approaches around a bend.")
    LISTEN_MPS = Param[float]("policy.listen_mps", 0.2, "Speed cap in m/s while listening at a blind corner.", constraint=Range(lo=0.0))
    HOLD_MPS = Param[float]("policy.hold_mps", 0.03, "Speed cap in m/s while holding for a source.", constraint=Range(lo=0.0))
    LOOKAHEAD_M = Param[float]("policy.lookahead_m", 4.0, "Path length in m ahead of the robot searched for bends.", constraint=Range(lo=0.0))
    APPROACH_M = Param[float]("policy.approach_m", 4.0, "Distance in m from a bend inside which the listen cap applies.", constraint=Range(lo=0.0))
    HOLD_LEN_M = Param[float]("policy.hold_len_m", 1.5, "Length in m of the hold zone before a bend.", constraint=Range(lo=0.0))
    HOLD_OFFSET_M = Param[float]("policy.hold_offset_m", 1.0, "Distance in m between the bend and the hold zone.", constraint=Range(lo=0.0))
    LANE_RADIUS_M = Param[float]("policy.lane_radius_m", 0.6, "Radius in m of the lane around the plan that counts belief mass.", constraint=Range(lo=0.0))
    CORNER_RADIUS_M = Param[float]("policy.corner_radius_m", 2.0, "Radius in m around a bend that counts belief mass for yielding.", constraint=Range(lo=0.0))
    BEND_HYSTERESIS_M = Param[float]("policy.bend_hysteresis_m", 1.0, "Distance in m a bend must move before it counts as a new bend.", constraint=Range(lo=0.0))
    REARM_AFTER_M = Param[float]("policy.rearm_after_m", 3.0, "Distance in m the robot travels past a bend before it rearms.", constraint=Range(lo=0.0))
    YIELD_FRACTION = Param[float]("policy.yield_fraction", 0.5, "Share of belief mass near a bend above which the robot yields.", constraint=Range(lo=0.0, hi=1.0))
    RELEASE_FRACTION = Param[float]("policy.release_fraction", 0.25, "Share of belief mass near a bend below which the robot releases a yield.", constraint=Range(lo=0.0, hi=1.0))
    MIN_YIELD_S = Param[float]("policy.min_yield_s", 3.0, "Shortest yield in s.", constraint=Range(lo=0.0))
    YIELD_TIMEOUT_S = Param[float]("policy.yield_timeout_s", 15.0, "Longest yield in s before the robot proceeds.", constraint=Range(lo=0.0))
    RECEDE_S = Param[float]("policy.recede_s", 2.0, "Seconds a receding source must be tracked before the yield ends.", constraint=Range(lo=0.0))
    LEVEL_TREND_DB_PER_S = Param[float]("policy.level_trend_db_per_s", -1.0, "Level trend in dB/s below which a source counts as receding.")
    LEVEL_TREND_TAU_S = Param[float]("policy.level_trend_tau_s", 5.0, "Time constant in s of the level trend filter.", constraint=Range(lo=0.0, lo_open=True))
    BELIEF_THRESHOLD = Param[float]("policy.belief_threshold", 0.6, "Belief value from which a cell blocks the plan.", constraint=Range(lo=0.0, hi=1.0))
    SPEED_MIN_PCT = Param[int]("policy.speed_min_pct", 40, "Lowest speed limit in percent the policy publishes.", constraint=Range(lo=0))
    SPEED_FREE_PCT = Param[int]("policy.speed_free_pct", 100, "Speed limit in percent the policy publishes when nothing blocks.", constraint=Range(lo=0))
    REACTION_RADIUS_M = Param[float]("policy.reaction_radius_m", 2.0, "Radius in m around the robot inside which belief mass is ignored.", constraint=Range(lo=0.0))


class GoalRuleName(enum.StrEnum):
    NEAREST = "nearest"
    BELIEF = "belief"


class StopRuleName(enum.StrEnum):
    NEVER = "never"
    EVERY_GOAL = "every_goal"
    ON_CALL = "on_call"
    FIXED_BEND = "fixed_bend"
    FIXED = "fixed"
    ROOM_ENTRY = "room_entry"
    THREE_POSITIONS = "three_positions"


class DriverGroup(ParamGroup):
    ENABLED = Param[bool]("driver.enabled", False, "Start the search driver: frontier goals on the robot's SLAM map, sent to its navigate_to_pose.")
    GOAL_RULE = Param[GoalRuleName]("driver.goal_rule", GoalRuleName.NEAREST, "Search goal rule, nearest frontier (no audio) or belief.")
    STOP_RULE = Param[StopRuleName]("driver.stop_rule", StopRuleName.NEVER, "Search stop rule: never, every_goal, on_call, fixed_bend, fixed, room_entry or three_positions.")
    LISTEN_S = Param[float]("driver.listen_s", 3.0, "Seconds the search driver halts to listen per stop.", constraint=Range(lo=0.0))
    SNAP = Param[bool]("driver.snap", False, "Snap a heard call onto the SLAM-map opening or frontier it points at (belief rule).")
    REPLAN_ON_CALL = Param[bool]("driver.replan_on_call", True, "Ask the goal rule again on every new call while driving and switch to the goal it picks.")
    STUCK_S = Param[float]("driver.stuck_s", 15.0, "Seconds without leaving reached_radius_m after which a goal counts as failed, 0 = never.", constraint=Range(lo=0.0))
    RATE_HZ = Param[float]("driver.rate_hz", 2.0, "Search driver tick rate in Hz.", constraint=Range(lo=0.0, lo_open=True))
    KINDS = Param[tuple[str, ...]]("driver.kinds", ("speech",), "Detection kinds the search driver reacts to.", parse=names)
    HEARD_MAX_AGE_S = Param[float]("driver.heard_max_age_s", 10.0, "Age in s after which a heard call no longer steers the search.", constraint=Range(lo=0.0))
    MIN_FRONTIER_CELLS = Param[int]("driver.min_frontier_cells", 8, "Smallest frontier in cells the search considers.", constraint=Range(lo=0))
    REACHED_RADIUS_M = Param[float]("driver.reached_radius_m", 0.5, "Distance in m from a goal at which it counts as reached.", constraint=Range(lo=0.0, lo_open=True))
    OPENING_MIN_WIDTH_M = Param[float]("driver.opening_min_width_m", 0.6, "Narrowest opening in m the search detects.", constraint=Range(lo=0.0, lo_open=True))
    OPENING_MAX_WIDTH_M = Param[float]("driver.opening_max_width_m", 1.6, "Widest opening in m the search detects.", constraint=Range(lo=0.0, lo_open=True))
    OPENING_DEPTH_M = Param[float]("driver.opening_depth_m", 0.6, "Depth in m of free space required on both sides of an opening.", constraint=Range(lo=0.0, lo_open=True))
    BELIEF_RADIUS_M = Param[float]("driver.belief_radius_m", 1.5, "Radius in m around a frontier that sums belief mass.", constraint=Range(lo=0.0, lo_open=True))
    BELIEF_MIN_MASS = Param[float]("driver.belief_min_mass", 1.0, "Least belief mass a frontier needs for the belief rule to pick it.", constraint=Range(lo=0.0))
    LOCATE_THRESHOLD = Param[float]("driver.locate_threshold", 0.8, "Belief value from which a person counts as located.", constraint=Range(lo=0.0, hi=1.0))
    DISTANCE_SCALE_M = Param[float]("driver.distance_scale_m", 5.0, "Distance in m that scales the cost of a goal.", constraint=Range(lo=0.0, lo_open=True))
    STANDOFF_M = Param[float]("driver.standoff_m", 1.0, "Distance the belief rule stops short of a located person, below the task's found radius.", constraint=Range(lo=0.0, lo_open=True))
    APPROACH_REACH = Param[float]("driver.approach_reach", 2.0, "Standoffs from a located person within which the belief rule takes a free cell, beyond it the frontier nearest the person.", constraint=Range(lo=0.0, lo_open=True))
    SNAP_MAX_ERROR_DEG = Param[float]("driver.snap_max_error_deg", 20.0, "Largest angle in degrees between a call and an opening for it to snap.", constraint=Range(lo=0.0, hi=180.0))
    BEND_LOOKAHEAD_M = Param[float]("driver.bend_lookahead_m", 4.0, "Path length in m ahead searched for bends by the fixed_bend stop rule.", constraint=Range(lo=0.0))
    BEND_STOP_BEFORE_M = Param[float]("driver.bend_stop_before_m", 2.0, "Distance in m before a bend at which the fixed_bend rule stops.", constraint=Range(lo=0.0, lo_open=True))
    BEND_HYSTERESIS_M = Param[float]("driver.bend_hysteresis_m", 1.0, "Distance in m a bend must move before it counts as a new bend.", constraint=Range(lo=0.0))
    FIXED_TRIGGER_M = Param[float]("driver.fixed.trigger_m", 2.0, "fixed stop rule along-path distance before a corner or door at which the search stops.", constraint=Range(lo=0.0))
    FIXED_LATERAL_M = Param[float]("driver.fixed.lateral_m", 1.5, "fixed stop rule distance from the plan within which a corner or door is on the path.", constraint=Range(lo=0.0))
    FIXED_PATH_HORIZON_M = Param[float]("driver.fixed.path_horizon_m", 8.0, "fixed stop rule length of the plan ahead it looks at.", constraint=Range(lo=0.0, lo_open=True))
    FIXED_REARM_M = Param[float]("driver.fixed.rearm_m", 3.0, "fixed stop rule distance from a served corner or door beyond which it stops again.", constraint=Range(lo=0.0))
    FIXED_DOOR_MAX_WIDTH_M = Param[float]("driver.fixed.door_max_width_m", 1.5, "fixed stop rule widest aperture between two corners that counts as a door.", constraint=Range(lo=0.0, lo_open=True))
    ROOM_ENTRY_RADIUS_M = Param[float]("driver.room_entry.radius_m", 0.6, "room_entry stop rule distance from a door's midpoint that counts as crossing it.", constraint=Range(lo=0.0, lo_open=True))
    ROOM_ENTRY_REARM_M = Param[float]("driver.room_entry.rearm_m", 3.0, "room_entry stop rule distance from a served door beyond which it stops again.", constraint=Range(lo=0.0))
    ROOM_ENTRY_DOOR_MAX_WIDTH_M = Param[float]("driver.room_entry.door_max_width_m", 1.5, "room_entry stop rule widest aperture between two corners that counts as a door.", constraint=Range(lo=0.0, lo_open=True))
    THREE_POSITIONS_COUNT = Param[int]("driver.three_positions.count", 3, "three_positions stop rule number of distinct heard positions after which it never stops again.", constraint=Range(lo=0))
    THREE_POSITIONS_SEPARATION_M = Param[float]("driver.three_positions.separation_m", 1.0, "three_positions stop rule distance below which two listening points are one position.", constraint=Range(lo=0.0, lo_open=True))


class AudioGroup(ParamGroup):
    RELIABLE_ENABLED = Param[bool]("audio.reliable.enabled", False, "Subscribe to audio frames with reliable QoS.")


class SeldGroup(ParamGroup):
    CHECKPOINT = Param[str]("seld.checkpoint", "", "Path of the SELDnet checkpoint (empty = the packaged one).")
    SCALER = Param[str]("seld.scaler", "", "Path of the SELDnet feature scaler (empty = the packaged one).")
    DEVICE = Param[str]("seld.device", "cuda", "Torch device of the SELDnet front-end.")
    TORCH_THREADS = Param[int]("seld.torch_threads", 2, "Torch CPU threads of the SELDnet front-end.", constraint=Range(lo=0))
    DET_THRESHOLD = Param[float]("seld.det_threshold", 0.5, "Detection score above which SELDnet reports a source.", constraint=Range(lo=0.0, hi=1.0))
    LOOKAHEAD_FRAMES = Param[int]("seld.lookahead_frames", 5, "SELDnet front-end label frames of future context, 100 ms each.", constraint=Range(lo=0))
    BEARING_SOURCE = Param[BearingSource]("seld.bearing_source", BearingSource.GCC, "SELDnet front-end bearing, gcc fits GCC-PHAT over the array, seld takes the model azimuth.")


class SrpGroup(ParamGroup):
    HOP_S = Param[float]("srp.hop_s", 0.1, "srp front-end hop length in seconds.", constraint=Range(lo=0.0, lo_open=True))
    FLOOR_WINDOW_S = Param[float]("srp.floor_window_s", 5.0, "srp front-end noise-floor median window in seconds.", constraint=Range(lo=0.0, lo_open=True))
    ONSET_DB = Param[float]("srp.onset_db", 6.0, "srp front-end onset threshold above the floor in dB.")


class CallGroup(ParamGroup):
    CALIB_FILE = Param[str]("call.calib_file", "", description="call front-end calibration npz with threshold__<state> and cov__<state>__4096, empty = default thresholds and no whitening.")
    MOVING_MPS = Param[float]("call.moving_mps", 0.02, "Speed in m/s below which a call source counts as still.", constraint=Range(lo=0.0))
    SPEED_WINDOW_S = Param[float]("call.speed_window_s", 0.3, "Window in s over which the call front-end measures source speed.", constraint=Range(lo=0.0, lo_open=True))


GROUPS: tuple[type[ParamGroup], ...] = (MapGroup, DiagnosticsGroup, HearingGroup, BeliefGroup, PolicyGroup, DriverGroup, AudioGroup, SeldGroup, SrpGroup, CallGroup)


def all_params() -> dict[str, Param[object]]:
    """Every declared parameter by full name, belief.emission_db.<kind> excluded."""
    return {param.name: param for group in GROUPS for param in group.params()}


class Configuration:
    """Node parameters by group. A group declares its parameters on first access, so a node touches each group it uses in __init__."""

    def __init__(self, server: ROSParamServer) -> None:
        self._server = server

    def group[G: ParamGroup](self, cls: type[G]) -> G:
        """The instance of cls behind its named accessor, BeliefGroup behind Belief."""
        group = getattr(self, cls.__name__.removesuffix("Group"))
        if not isinstance(group, cls):
            raise TypeError(f"{cls.__name__} has no accessor on {type(self).__name__}")
        return group

    @functools.cached_property
    def Map(self) -> MapGroup:
        return MapGroup(self._server)

    @functools.cached_property
    def Diagnostics(self) -> DiagnosticsGroup:
        return DiagnosticsGroup(self._server)

    @functools.cached_property
    def Hearing(self) -> HearingGroup:
        return HearingGroup(self._server)

    @functools.cached_property
    def Belief(self) -> BeliefGroup:
        return BeliefGroup(self._server)

    @functools.cached_property
    def Policy(self) -> PolicyGroup:
        return PolicyGroup(self._server)

    @functools.cached_property
    def Driver(self) -> DriverGroup:
        return DriverGroup(self._server)

    @functools.cached_property
    def Audio(self) -> AudioGroup:
        return AudioGroup(self._server)

    @functools.cached_property
    def Seld(self) -> SeldGroup:
        return SeldGroup(self._server)

    @functools.cached_property
    def Srp(self) -> SrpGroup:
        return SrpGroup(self._server)

    @functools.cached_property
    def Call(self) -> CallGroup:
        return CallGroup(self._server)
