"""hearing.launch.py: which hearing nodes an env gets for its robot.hearing.* configuration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

LAUNCH = Path(__file__).parents[2] / "launch" / "hearing.launch.py"


@pytest.fixture(autouse=True)
def _launch_gate():
    pytest.importorskip("launch_ros")
    pytest.importorskip("arena_simulation_setup")


def _executables(**configs: str) -> list[str]:
    import launch
    from launch_ros.actions import Node

    spec = importlib.util.spec_from_file_location("hearing_launch", LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = launch.LaunchContext()
    context.launch_configurations.update({"env.ns": "/arena/env_0", "tg_node": "arena_env_node", "frontend": "bus", "policy": "belief", **configs})
    return [action.node_executable for action in module._hearing(context) if isinstance(action, Node)]


def test_the_search_driver_starts_only_when_enabled() -> None:
    assert "hearing_search_driver" not in _executables()
    assert "hearing_search_driver" in _executables(**{"robot.hearing.driver.enabled": "true"})


def test_a_disabled_search_driver_leaves_the_hearing_nodes_as_they_were() -> None:
    assert _executables(**{"robot.hearing.driver.enabled": "false"}) == ["hearing_belief_node", "hearing_policy"]


def _driver_params(**configs: str) -> dict:
    import launch
    import yaml
    from launch.substitutions import TextSubstitution
    from launch_ros.actions import Node

    spec = importlib.util.spec_from_file_location("hearing_launch", LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = launch.LaunchContext()
    context.launch_configurations.update({"env.ns": "/arena/env_0", "tg_node": "arena_env_node", "frontend": "bus", "policy": "belief", "robot.hearing.driver.enabled": "true", **configs})
    (driver,) = [action for action in module._hearing(context) if isinstance(action, Node) and action.node_executable == "hearing_search_driver"]
    driver._perform_substitutions(context)
    cmd = ["".join(s.text for s in part) if all(isinstance(s, TextSubstitution) for s in part) else "" for part in driver.cmd]
    params: dict = {}
    for path in (cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--params-file"):
        for root in yaml.load(Path(path).read_text(), Loader=yaml.FullLoader).values():
            params.update(root["ros__parameters"])
    return params


def test_the_srp_front_end_feeds_its_onsets_to_the_search_driver() -> None:
    assert list(_driver_params(frontend="srp")["driver.kinds"]) == ["onset"]


def test_the_bus_front_end_leaves_the_search_driver_its_default_kinds() -> None:
    assert "driver.kinds" not in _driver_params()
