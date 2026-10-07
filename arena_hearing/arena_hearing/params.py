"""Every hearing node parameter, grouped. Defaults live only here."""

from __future__ import annotations

import enum
import functools
import math
import typing
from collections.abc import Mapping

from arena_rclpy_mixins.param_groups import Param, ParamGroup, count, finite, floats, names, non_negative, positive, within

if typing.TYPE_CHECKING:
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer, ROSParamT


class Frontend(enum.StrEnum):
    BUS = "bus"
    SRP = "srp"
    SELD = "seld"

    @property
    def array_spec(self) -> str:
        """Array spec the front-end runs on unless auditory.array.spec names one, empty for the bus."""
        return "" if self is Frontend.BUS else "four_mic"


class BearingSource(enum.StrEnum):
    GCC = "gcc"
    SELD = "seld"


class MapGroup(ParamGroup):
    FRAME = Param[str]("map.frame", "map")
    OCCUPIED_THRESHOLD = Param[int]("map.occupied_threshold", 50, parse=count)


class DiagnosticsGroup(ParamGroup):
    PERIOD_S = Param[float]("diagnostics.period_s", 5.0, parse=positive)
    REPORT_PERIOD_S = Param[float]("diagnostics.report_period_s", 0.5, parse=positive)
    ACTIVITY_THRESHOLD_DBFS = Param[float]("diagnostics.activity_threshold_dbfs", -70.0, parse=finite)


class HearingGroup(ParamGroup):
    FRONTEND = Param[Frontend]("hearing.frontend", Frontend.BUS.value, parse=Frontend)
    TG_NODE = Param[str]("hearing.tg_node", "task_generator_node")


def _level(value: object) -> float:
    result = float(typing.cast(float, value))
    if math.isinf(result):
        raise ValueError(f"{value!r} is not a level")
    return result


class BeliefGroup(ParamGroup):
    MARKERS_ENABLED = Param[bool]("belief.markers.enabled", True)
    MARKERS_RANGE_M = Param[float]("belief.markers.range_m", 4.0, parse=positive)
    MARKERS_MAX_COUNT = Param[int]("belief.markers.max_count", 12, parse=count)
    PUBLISH_RATE_HZ = Param[float]("belief.publish_rate_hz", 5.0, parse=positive)
    RESOLUTION_M = Param[float]("belief.resolution_m", 0.0, parse=non_negative)
    STANDALONE_ENABLED = Param[bool]("belief.standalone.enabled", False)
    STANDALONE_ORIGIN_M = Param[tuple[float, ...]]("belief.standalone.origin_m", [-10.0, -10.0], parse=floats)
    STANDALONE_SIZE_M = Param[tuple[float, ...]]("belief.standalone.size_m", [20.0, 20.0], parse=floats)
    WEDGE_DEG = Param[float]("belief.wedge_deg", 10.0, parse=positive)
    MIN_HALF_WIDTH_M = Param[float]("belief.min_half_width_m", 0.3, parse=non_negative)
    TAU_S = Param[float]("belief.tau_s", 2.0, parse=positive)
    MAX_RANGE_M = Param[float]("belief.max_range_m", 15.0, parse=positive)
    MIN_RANGE_M = Param[float]("belief.min_range_m", 0.5, parse=non_negative)
    REFERENCE_DISTANCE_M = Param[float]("belief.reference_distance_m", 1.0, parse=positive)
    LEVEL_RANGE_ENABLED = Param[bool]("belief.level_range.enabled", False)
    RANGE_SIGMA_FRAC = Param[float]("belief.range_sigma_frac", 0.35, parse=positive)
    RANGE_FLOOR_WEIGHT = Param[float]("belief.range_floor_weight", 0.15, parse=non_negative)
    EVENT_MASS = Param[float]("belief.event_mass", 1.0, parse=positive)
    EVENT_RATE_HZ = Param[float]("belief.event_rate_hz", 2.0, parse=positive)
    MASS_FULL_SCALE = Param[float]("belief.mass_full_scale", 0.0, parse=non_negative)
    KINDS = Param[tuple[str, ...]]("belief.kinds", ["footstep", "speech"], parse=names)
    MIN_LEVEL_DB = Param[float]("belief.min_level_db", -1e9, parse=finite)

    EMISSION_DB_PREFIX: typing.ClassVar[str] = "belief.emission_db."

    def emission_db(self, levels: Mapping[str, float]) -> dict[str, ROSParamT[float]]:
        """Declare belief.emission_db.<kind> per kind, defaulting to the level of the kind's default asset, NaN for unknown."""
        return {kind: self._server.ROSParam(f"{self.EMISSION_DB_PREFIX}{kind}", float(level), parse=_level) for kind, level in levels.items()}


class PolicyGroup(ParamGroup):
    PUBLISH_RATE_HZ = Param[float]("policy.publish_rate_hz", 10.0, parse=positive)
    MAX_LINEAR_MPS = Param[float]("policy.max_linear_mps", 0.0, parse=non_negative)
    LISTEN_ENABLED = Param[bool]("policy.listen.enabled", True)
    YIELD_ENABLED = Param[bool]("policy.yield.enabled", True)
    LISTEN_MPS = Param[float]("policy.listen_mps", 0.2, parse=non_negative)
    HOLD_MPS = Param[float]("policy.hold_mps", 0.03, parse=non_negative)
    LOOKAHEAD_M = Param[float]("policy.lookahead_m", 4.0, parse=non_negative)
    APPROACH_M = Param[float]("policy.approach_m", 4.0, parse=non_negative)
    HOLD_LEN_M = Param[float]("policy.hold_len_m", 1.5, parse=non_negative)
    HOLD_OFFSET_M = Param[float]("policy.hold_offset_m", 1.0, parse=non_negative)
    LANE_RADIUS_M = Param[float]("policy.lane_radius_m", 0.6, parse=non_negative)
    CORNER_RADIUS_M = Param[float]("policy.corner_radius_m", 2.0, parse=non_negative)
    BEND_HYSTERESIS_M = Param[float]("policy.bend_hysteresis_m", 1.0, parse=non_negative)
    REARM_AFTER_M = Param[float]("policy.rearm_after_m", 3.0, parse=non_negative)
    YIELD_FRACTION = Param[float]("policy.yield_fraction", 0.5, parse=within(0.0, 1.0))
    RELEASE_FRACTION = Param[float]("policy.release_fraction", 0.25, parse=within(0.0, 1.0))
    MIN_YIELD_S = Param[float]("policy.min_yield_s", 3.0, parse=non_negative)
    YIELD_TIMEOUT_S = Param[float]("policy.yield_timeout_s", 15.0, parse=non_negative)
    RECEDE_S = Param[float]("policy.recede_s", 2.0, parse=non_negative)
    LEVEL_TREND_DB_PER_S = Param[float]("policy.level_trend_db_per_s", -1.0, parse=finite)
    LEVEL_TREND_TAU_S = Param[float]("policy.level_trend_tau_s", 5.0, parse=positive)
    BELIEF_THRESHOLD = Param[float]("policy.belief_threshold", 0.6, parse=within(0.0, 1.0))
    SPEED_MIN_PCT = Param[int]("policy.speed_min_pct", 40, parse=count)
    SPEED_FREE_PCT = Param[int]("policy.speed_free_pct", 100, parse=count)
    REACTION_RADIUS_M = Param[float]("policy.reaction_radius_m", 2.0, parse=non_negative)


class AudioGroup(ParamGroup):
    RELIABLE_ENABLED = Param[bool]("audio.reliable.enabled", False)


class SeldGroup(ParamGroup):
    CHECKPOINT = Param[str]("seld.checkpoint", "")
    SCALER = Param[str]("seld.scaler", "")
    DEVICE = Param[str]("seld.device", "cuda", description="Torch device of the SELDnet front-end.")
    TORCH_THREADS = Param[int]("seld.torch_threads", 2, parse=count)
    DET_THRESHOLD = Param[float]("seld.det_threshold", 0.5, parse=within(0.0, 1.0))
    LOOKAHEAD_FRAMES = Param[int]("seld.lookahead_frames", 5, parse=count, description="SELDnet front-end label frames of future context, 100 ms each.")
    BEARING_SOURCE = Param[BearingSource]("seld.bearing_source", BearingSource.GCC.value, parse=BearingSource, description="SELDnet front-end bearing, gcc fits GCC-PHAT over the array, seld takes the model azimuth.")


class SrpGroup(ParamGroup):
    HOP_S = Param[float]("srp.hop_s", 0.1, parse=positive, description="srp front-end hop length in seconds.")
    FLOOR_WINDOW_S = Param[float]("srp.floor_window_s", 5.0, parse=positive, description="srp front-end noise-floor median window in seconds.")
    ONSET_DB = Param[float]("srp.onset_db", 6.0, parse=finite, description="srp front-end onset threshold above the floor in dB.")


GROUPS: tuple[type[ParamGroup], ...] = (MapGroup, DiagnosticsGroup, HearingGroup, BeliefGroup, PolicyGroup, AudioGroup, SeldGroup, SrpGroup)


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
    def Audio(self) -> AudioGroup:
        return AudioGroup(self._server)

    @functools.cached_property
    def Seld(self) -> SeldGroup:
        return SeldGroup(self._server)

    @functools.cached_property
    def Srp(self) -> SrpGroup:
        return SrpGroup(self._server)
