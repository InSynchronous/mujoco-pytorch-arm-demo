"""Shared helpers for the pick-and-place behavior-cloning pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ARM_JOINTS = ("yaw", "shoulder", "elbow", "wrist")
FINGER_JOINTS = ("fl", "fr")
MOTOR_COUNT = 6


def shard_dirs(data_dir: str | Path) -> list[Path]:
    shards = sorted(Path(data_dir).glob("shard_*"))
    if not shards:
        raise FileNotFoundError(f"no shards found under {data_dir}")
    return shards


def load_info(data_dir: str | Path) -> dict:
    """Load the dataset metadata written by collect_data."""
    return json.loads((shard_dirs(data_dir)[0] / "info.json").read_text())


def sample_cube(info: dict, rng: np.random.Generator) -> tuple[float, float]:
    """Same rejection sampler as the C++ episode generator.

    Only used for resetting the environment / choosing evaluation cubes. This
    is never part of the policy observation.
    """
    place = info["place_position"]
    xmin, xmax, ymin, ymax = info["cube_bounds"]
    rmin, rmax = info["cube_radius"]
    clearance = info["place_clearance"]
    for _ in range(100_000):
        x = float(rng.uniform(xmin, xmax))
        y = float(rng.uniform(ymin, ymax))
        radius = math.hypot(x, y)
        if radius < rmin or radius > rmax:
            continue
        if math.hypot(x - place[0], y - place[1]) < clearance:
            continue
        return x, y
    raise RuntimeError("could not sample a cube position")


class MotorModel:
    """Index/ranges/lookups for the six motors and the scene's cube."""

    def __init__(self, model):
        import mujoco

        self.model = model
        self.qadr = []
        self.dadr = []
        for name in ARM_JOINTS + FINGER_JOINTS:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise RuntimeError(f"missing joint {name}")
            self.qadr.append(model.jnt_qposadr[joint])
            self.dadr.append(model.jnt_dofadr[joint])
        self.qadr = np.asarray(self.qadr, dtype=np.int32)
        self.dadr = np.asarray(self.dadr, dtype=np.int32)

        self.actuators = []
        for name in ("a_yaw", "a_shoulder", "a_elbow", "a_wrist", "a_fl", "a_fr"):
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act < 0:
                raise RuntimeError(f"missing actuator {name}")
            self.actuators.append(act)
        self.actuators = np.asarray(self.actuators, dtype=np.int32)
        self.ctrl_low = model.actuator_ctrlrange[self.actuators, 0].copy()
        self.ctrl_high = model.actuator_ctrlrange[self.actuators, 1].copy()

        self.joint_low = np.empty(MOTOR_COUNT)
        self.joint_high = np.empty(MOTOR_COUNT)
        for i, qadr in enumerate(self.qadr):
            joint = np.where(model.jnt_qposadr == qadr)[0][0]
            self.joint_low[i] = model.jnt_range[joint, 0]
            self.joint_high[i] = model.jnt_range[joint, 1]

        cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.cube_qadr = int(model.jnt_qposadr[cube_joint])
        cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        self.cube_half_height = float(model.geom_size[cube_geom, 2])

    def qpos(self, data) -> np.ndarray:
        return np.asarray(data.qpos[self.qadr], dtype=np.float32)

    def qvel(self, data) -> np.ndarray:
        return np.asarray(data.qvel[self.dadr], dtype=np.float32)

    def cube_xy(self, data) -> np.ndarray:
        return np.asarray(data.qpos[self.cube_qadr : self.cube_qadr + 2], dtype=np.float64)


def set_cube(data, motors: MotorModel, x: float, y: float) -> None:
    yaw = math.atan2(y, x)
    idx = motors.cube_qadr
    data.qpos[idx + 0] = x
    data.qpos[idx + 1] = y
    data.qpos[idx + 2] = motors.cube_half_height + 5e-4
    data.qpos[idx + 3] = math.cos(0.5 * yaw)
    data.qpos[idx + 4] = 0.0
    data.qpos[idx + 5] = 0.0
    data.qpos[idx + 6] = math.sin(0.5 * yaw)


def make_camera(info: dict):
    import mujoco

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = float(info["camera"]["azimuth"])
    cam.elevation = float(info["camera"]["elevation"])
    cam.distance = float(info["camera"]["distance"])
    cam.lookat[:] = np.asarray(info["camera"]["lookat"], dtype=np.float64)
    return cam
