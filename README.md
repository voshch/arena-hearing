# arena_hearing

Robot hearing for Arena: the `srp` and `seld` front-ends on any robot audio
source, the directional belief grid, the speed-filter policy and the Nav2
overlay that applies its mask. It is the consumer side of the audio a robot
receives. The audio itself comes from an acoustics simulator, `acoustics:=arena`
(the arena_auditory backend) today. The `bus` front-end, which republishes the
simulator's propagated receptions, lives in arena_auditory.

This repository is the optional hearing feature, a submodule of Arena. Install
it with `arena feature hearing install`, which checks it out, installs its
system dependencies through rosdep, syncs its Python dependencies (torch,
librosa, scipy, huggingface-hub) and rebuilds. `robot.hearing` other than
`none` also needs `acoustics:=arena`, installed by `arena feature auditory install`.

## Layout

```
arena_hearing/                  this repository
`-- arena_hearing/              ament_python package
    |-- arena_hearing/          python package
    |   |-- constants.py        topic names of the hearing stack
    |   |-- params.py           every hearing node parameter and its default
    |   |-- fleet.py            per-robot lifecycle and lockstep channels
    |   |-- belief_grid.py belief_node.py      belief grid and its node
    |   |-- policy.py corners.py policy_node.py speed-filter policy and its node
    |   |-- doa.py onset.py timeline.py srp_frontend_node.py   srp front-end
    |   |-- seld.py dcase.py weights.py seld_frontend_node.py  seld front-end
    |   `-- audio_replay.py     wav replay as AudioFrame
    |-- config/                 nav2_overlay.yaml, weights.yaml
    |-- launch/                 hearing.launch.py
    `-- tools/                  source-only scripts, not installed: belief replay
```

## Front-ends

Every detection source publishes `arena_robots_msgs/SoundDetection` on
`<r>/hearing/<frontend>/detections`, so the belief node is source-agnostic:

| Frontend | Source |
|---|---|
| `bus` | the arena_auditory bus node, the propagated receptions of the robot, map-frame bearings |
| `srp` | `hearing_srp_frontend`, energy onsets plus GCC-PHAT over `raw_array`, array-frame bearings, kind `onset` |
| `seld` | `hearing_seld_frontend`, the SELDnet model over `raw_array`, array-frame bearings |

Every hearing node covers every fleet robot of its env: it follows
`state/robots`, keeps its state per robot and registers one hard lockstep
channel per robot. `hearing_belief` turns each robot's detections into a
decaying pedestrian-likelihood grid (`hearing/belief_grid`,
`hearing/belief_wedges`). `hearing_policy` reads it and is the single writer
of the robot's Nav2 `SpeedFilter` mask (`hearing/speed_filter_mask`) and its
`CostmapFilterInfo` (`hearing/costmap_filter_info`).

The front-ends read the array geometry off each `AudioFrame`: `srp` fits
GCC-PHAT over every pair of the microphone positions the frame declares, any
array of two or more microphones apart in the plane, and rebuilds when the
layout changes. `seld` reads only the array its weights were trained on
(`array` in `config/weights.yaml`): a stream whose sample rate, channel names
or microphone positions differ is refused with an error. Both convert levels
to dB SPL with the MEMS sensitivity the frame carries.

## Running in an Arena env

`robot.hearing:=bus|srp|seld` makes the task generator's hearing axis include
`launch/hearing.launch.py` per env and merge `config/nav2_overlay.yaml` (a
`SpeedFilter` on the local costmap and the controller's `speed_limit_topic`,
both below the robot namespace) into every robot's Nav2 parameters:

```bash
export ARENA_WORLD_PATH=$ARENA_WS_DIR/src/Arena/_assets/arena-benchmarks-prod-public/suites/acoustics/worlds
arena launch sim:=gazebo robot:=jackal world:=acoustics_bend_narrow_O \
    task.robots:=scenario task.obstacles:=scenario \
    task.scenario.file:=hearing__world-acoustics_bend_narrow_O__robot-moving__pedestrians-1__ends-a-to-b \
    acoustics:=arena robot.hearing:=bus
```

`srp` and `seld` default `auditory.array.spec` to `four_mic`.

## Launch arguments

`hearing.launch.py` declares:

| Argument | Default | Effect |
|---|---|---|
| `env.ns` | `/arena/env_0` | Env namespace, the hearing nodes live in it |
| `tg_node` | `task_generator_node` | Task generator node, the robot topics live below it |
| `frontend` | `bus` | `bus`, `srp` or `seld`, `srp` and `seld` also start their front-end node |
| `policy` | `full` | Mask layers: `belief` only, `listen` adds the corner listen cap, `full` adds yield |
| `robot.hearing.seld.device` | empty (`cuda`) | Torch device of the SELDnet front-end |
| `robot.hearing.seld.lookahead_frames` | empty (`5`) | SELDnet label frames of future context, 100 ms each |
| `robot.hearing.seld.bearing_source` | empty (`gcc`) | `gcc` fits GCC-PHAT over the array, `seld` takes the model azimuth |
| `robot.hearing.srp.hop_s` | empty (`0.1`) | srp hop length in seconds |
| `robot.hearing.srp.floor_window_s` | empty (`5.0`) | srp noise-floor median window in seconds |
| `robot.hearing.srp.onset_db` | empty (`6.0`) | srp onset threshold above the floor in dB |

The `robot.hearing.*` arguments are declared from the parameters in
[`params.py`](arena_hearing/arena_hearing/params.py) that carry a description.
Every `robot.hearing.<param>:=<value>` reaches the hearing nodes as parameter
`<param>`, coerced to the type of its default, for example
`robot.hearing.belief.tau_s:=3.0` or `robot.hearing.belief.emission_db.footstep:=55.0`.
An unknown key is an error, an empty value keeps the node default. These keys
never reach the robot adapters. The launch also derives `hearing.frontend`,
`hearing.tg_node`, `belief.event_rate_hz`, `policy.listen.enabled`,
`policy.yield.enabled` and, for `srp`, `belief.kinds`.

## Topics

Below the task generator node `/arena/env_0/task_generator_node`, `<r>` is the
robot namespace:

| Direction | Topic | Note |
|---|---|---|
| in | `<r>/audio/raw_array` | `arena_robots_msgs/AudioFrame` (`ArrayStream.RAW` of `arena_robots.audio`), srp and seld |
| in | `<r>/hearing/<frontend>/detections` | `arena_robots_msgs/SoundDetection` |
| in | `map`, `state/resetting`, `state/robots` | grid geometry, reset, fleet |
| in | `<r>/plan` | Nav2 global plan, for blind-bend detection |
| out | `<r>/hearing/srp/detections`, `<r>/hearing/seld/detections` | the front-ends |
| out | `<r>/lockstep/hearing` | front-end heartbeat per robot |
| out | `<r>/hearing/belief_grid`, `<r>/hearing/belief_wedges` | RViz, the wedges only while subscribed |
| out | `<r>/hearing/speed_filter_mask` | `OccupancyGrid`, latched, read by the robot's SpeedFilter |
| out | `<r>/hearing/costmap_filter_info` | `CostmapFilterInfo`, latched, points the SpeedFilter at the mask |
| out | `<r>/hearing/policy_state` | JSON: state, distances, masses, level slope, limit, binding layer, yield count and time |
| out | `<r>/hearing/policy_markers` | approach lane and hold band, only while subscribed |
| Nav2 | `<r>/hearing/speed_limit` | SpeedFilter to controller |

## Belief and policy

The belief layer of the mask is the belief max-filtered over a disc of
`policy.reaction_radius_m` (2.0 m) before thresholding. The robot slows from
`policy.speed_free_pct` (100) to `policy.speed_min_pct` (40) as the belief
goes from `policy.belief_threshold` (0.6) to 1. A wedge is never narrower
than `belief.min_half_width_m` (0.3 m). `belief.level_range.enabled` (default
false) keeps the wedge flat out to `belief.max_range_m`, since walls and doors
attenuate an occluded pedestrian and the level would place it too far away.
The emission level per kind is the level of the kind's default asset in the
sound catalog, overridable by `belief.emission_db.<kind>`.

`hearing_policy` composes the mask from three layers by lowest nonzero
percentage: belief, listen (`policy.listen_mps` on the approach to a blind
bend) and hold (`policy.hold_mps` before the bend while yielding). The robot's
own drivetrain masks the pedestrian it listens for, and its level grows with
speed.

A plan point is blind when at least 25 % of the free cells within 2 m of the
point `policy.lookahead_m` further along the plan, on its far side, are hidden
from it by occupied cells. The bend is the first point from which that clears,
or the plan end when the plan ends blind. Only blind stretches that start
within `policy.approach_m` plus `policy.lookahead_m` are searched. States: cruise, listen inside `policy.approach_m`,
yield once the belief mass within `policy.corner_radius_m` of the bend exceeds
`policy.yield_fraction`, and pass once the mass moved behind the robot, faded
below `policy.release_fraction`, or the level fell for `policy.recede_s`
(each after `policy.min_yield_s`), or at `policy.yield_timeout_s`. Pass takes
the bend at listen speed. Nav2 reads 0 % as no limit, so the hold is a
0.03 m/s creep. `robot.hearing.policy:=belief|listen|full` picks the active
layers.

## Evaluation

The arms are contestants of `contests/hearing.yaml` in arena_evaluation, run
against the `acoustics` suite:

```bash
arena evaluation benchmark --suite acoustics --contest hearing
```

`yield_count` and `time_yielding_s` on `<r>/hearing/policy_state` are
cumulative per robot. `python3 arena_hearing/tools/replay_belief.py EPISODE_DIR` (from a source checkout) replays an exported episode
through the belief grid offline, from SELDnet detections or the ground-truth
labels.

## Front-end checks

```bash
ros2 run arena_hearing hearing_audio_replay <4ch.wav> <robot> --array four_mic --ros-args -r __ns:=/arena/env_0
```

`hearing_audio_replay` publishes every channel of the wav on
`<tg>/<robot>/audio/raw_array`, `--array` declares the geometry the
front-ends need. A front-end picks the robot up from `state/robots`.

The streaming SELDnet path re-runs the model on a 5 s sliding window once per
100 ms label frame and emits the frame `seld.lookahead_frames` (default 5)
before the window end. The bearing comes from `seld.bearing_source`: `gcc`
(default) fits the array geometry to GCC-PHAT delays over every microphone
pair, `seld` takes the model azimuth, which is front-biased on the shipped
checkpoint.

## Weights

The SELDnet checkpoint and feature scaler are pinned with sha256 in
`config/weights.yaml`, hosted on Hugging Face and stored in
`$ARENA_DATA_DIR/auditory/seld/`:

```bash
ros2 run arena_hearing hearing_setup
```

The front-end fetches them on first use when they are missing. The model,
SALSA-Lite features and multi-ACCDOA decode are in `dcase.py`, adapted from
the DCASE 2023 SELD baseline (MIT).

## Tests

```bash
python3 -m pytest arena_hearing/tests/unit -q
python3 -m pytest arena_hearing/tests/ros -q
```

The ROS tests run the belief and policy nodes as subprocesses against a
synthetic fleet, map and TF, and need a sourced ROS 2 environment.

## Configuration

| File | Content |
|---|---|
| `config/weights.yaml` | SELDnet weights and pins |
| `config/nav2_overlay.yaml` | Nav2 SpeedFilter overlay |
| `arena_hearing/params.py` | every hearing node parameter and default |
| `task_generator/launch/hearing/` (Arena) | hearing axis dispatch and the include of `launch/hearing.launch.py` |
| `task_generator/task_generator/simulators/hearing/` (Arena) | node-side hearing axis, the single gateway into this package |
