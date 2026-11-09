"""Robot hearing layer of one env: belief grid, speed-filter policy, the srp, seld or call front-end and, with driver.enabled, the search driver, ``robot.hearing.<key>`` forwarded as ``<key>``."""

import launch
import launch.actions
from arena_rclpy_mixins.launch_argument import LaunchArgument
from arena_rclpy_mixins.params import Param
from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary
from launch_ros.actions import Node

from arena_hearing.belief_grid import NOMINAL_EVENT_RATE_HZ
from arena_hearing.params import BeliefGroup, DriverGroup, Frontend, HearingGroup, PolicyGroup, all_params

_PREFIX = "robot.hearing."
_POLICY_KEY = f"{_PREFIX}policy"
_POLICIES = ("belief", "listen", "full")
_PLUMBING = (HearingGroup.FRONTEND.name, HearingGroup.TG_NODE.name, BeliefGroup.EVENT_RATE_HZ.name, PolicyGroup.LISTEN_ENABLED.name, PolicyGroup.YIELD_ENABLED.name)
_FRONTEND_NODES = {Frontend.SRP: ("srp_frontend", "hearing_srp_frontend"), Frontend.SELD: ("seld_frontend", "hearing_seld_frontend"), Frontend.CALL: ("call_frontend", "hearing_call_frontend")}


def _hearing(context: launch.LaunchContext) -> list[launch.LaunchDescriptionEntity]:
    configs = context.launch_configurations
    params = all_params()
    frontend = Frontend(configs["frontend"])
    policy = configs["policy"]
    derived: dict[str, object] = {
        "hearing.frontend": frontend.value,
        "hearing.tg_node": configs["tg_node"],
        "belief.event_rate_hz": NOMINAL_EVENT_RATE_HZ[frontend.value],
        "policy.listen.enabled": policy != "belief",
        "policy.yield.enabled": policy == "full",
    }
    if frontend in (Frontend.SRP, Frontend.CALL):
        derived["belief.kinds"] = ["onset"]
        derived["driver.kinds"] = ["onset"]
    forwarded: dict[str, object] = {}
    detect_kinds = sorted(name for name, kind in SoundLibrary.default().kinds().items() if kind.detect)
    for key, raw in configs.items():
        if not key.startswith(_PREFIX) or key == _POLICY_KEY or not raw:
            continue
        name = key.removeprefix(_PREFIX)
        param = Param[float](name, 0.0, f"Emission level of {name.removeprefix(BeliefGroup.EMISSION_DB_PREFIX)} in dB.") if name.startswith(BeliefGroup.EMISSION_DB_PREFIX) else params.get(name)
        if param is None:
            raise ValueError(f"{key} is not a hearing node parameter")
        if name.startswith(BeliefGroup.EMISSION_DB_PREFIX) and name.removeprefix(BeliefGroup.EMISSION_DB_PREFIX) not in detect_kinds:
            raise ValueError(f"{key} names no detect kind, expected one of {detect_kinds}")
        forwarded[name] = param.coerce(raw)
    hearing_params = {"use_sim_time": True, **derived, **forwarded}
    env_ns = "/" + configs["env.ns"].strip("/")

    def node(name: str, executable: str) -> Node:
        return Node(package="arena_hearing", executable=executable, name=name, namespace=env_ns, output="screen", parameters=[hearing_params])

    actions: list[launch.LaunchDescriptionEntity] = [node("hearing_belief", "hearing_belief_node"), node("hearing_policy", "hearing_policy")]
    if frontend in _FRONTEND_NODES:
        actions.append(node(*_FRONTEND_NODES[frontend]))
    if forwarded.get(DriverGroup.ENABLED.name):
        actions.append(node("hearing_search_driver", "hearing_search_driver"))
    return actions


def generate_launch_description() -> launch.LaunchDescription:
    return launch.LaunchDescription(
        [
            LaunchArgument("env.ns", default_value="/arena/env_0", description="Env namespace."),
            LaunchArgument("tg_node", default_value="arena_env_node", description="Task generator node name."),
            LaunchArgument("frontend", default_value=Frontend.BUS.value, choices=[frontend.value for frontend in Frontend], description="Detection source."),
            LaunchArgument("policy", default_value="full", choices=list(_POLICIES), description="Mask layers."),
            *(LaunchArgument.from_param(param, prefix=_PREFIX) for name, param in all_params().items() if name not in _PLUMBING),
            launch.actions.OpaqueFunction(function=_hearing),
        ]
    )
