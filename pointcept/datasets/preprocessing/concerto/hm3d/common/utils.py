#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from typing import List, Optional

import habitat_sim
import numpy as np
import quaternion as qt
from habitat_sim.utils.common import quat_from_angle_axis


def get_random_quaternion() -> qt.quaternion:
    return quat_from_angle_axis(
        math.radians(random.uniform(-180.0, 180.0)), np.array([0, 1.0, 0])
    )


def convert_heading_to_quaternion(heading: float) -> qt.quaternion:
    # heading angle in degrees
    return quat_from_angle_axis(math.radians(heading), np.array([0, 1.0, 0]))


def make_habitat_configuration(scene_path: str, use_sensor: bool = False):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path

    # agent configuration
    sensor_cfg = habitat_sim.CameraSensorSpec()
    sensor_cfg.resolution = [1080, 960]
    sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_cfg] if use_sensor else []

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def robust_load_sim(scene_path: str) -> habitat_sim.Simulator:
    sim_cfg = make_habitat_configuration(scene_path, use_sensor=False)
    hsim = habitat_sim.Simulator(sim_cfg)
    if not hsim.pathfinder.is_loaded:
        hsim.close()
        sim_cfg = make_habitat_configuration(scene_path, use_sensor=True)
        hsim = habitat_sim.Simulator(sim_cfg)
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        hsim.recompute_navmesh(hsim.pathfinder, navmesh_settings)
    return hsim


def get_filtered_scenes(scenes: List[str], filter_scenes_path: str) -> List[str]:
    """
    Filter scenes to only include valid scenes.
    """
    with open(filter_scenes_path, "r") as fp:
        filter_scenes = fp.readlines()
    filter_scenes = [f.strip("\n") for f in filter_scenes]
    filtered_scenes = []
    for scene in scenes:
        scene_name = scene.split("/")[-1][: -len(".glb")]
        if scene_name in filter_scenes:
            filtered_scenes.append(scene)
    return filtered_scenes


def quaternion_to_list(q: qt.quaternion):
    return q.imag.tolist() + [q.real]


# -------------------------------------------------------------------------------
# Functionality from habitat-lab
# -------------------------------------------------------------------------------
def calculate_meters_per_pixel(map_resolution: int, pathfinder=None):
    r"""Calculate the meters_per_pixel for a given map resolution"""
    lower_bound, upper_bound = pathfinder.get_bounds()
    return min(
        abs(upper_bound[coord] - lower_bound[coord]) / map_resolution
        for coord in [0, 2]
    )


def get_topdown_map(
    pathfinder,
    height: float,
    map_resolution: int = 1024,
    meters_per_pixel: Optional[float] = None,
) -> np.ndarray:
    r"""Return a top-down occupancy map for a sim. Note, this only returns valid
    values for whatever floor the agent is currently on.
    :param pathfinder: A habitat-sim pathfinder instances to get the map from
    :param height: The height in the environment to make the topdown map
    :param map_resolution: Length of the longest side of the map.  Used to calculate :p:`meters_per_pixel`
    :param draw_border: Whether or not to draw a border
    :param meters_per_pixel: Overrides map_resolution an
    :return: Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
        the flag is set).
    """

    if meters_per_pixel is None:
        meters_per_pixel = calculate_meters_per_pixel(
            map_resolution, pathfinder=pathfinder
        )

    top_down_map = pathfinder.get_topdown_view(
        meters_per_pixel=meters_per_pixel, height=height
    ).astype(np.uint8)

    return np.ascontiguousarray(top_down_map)
