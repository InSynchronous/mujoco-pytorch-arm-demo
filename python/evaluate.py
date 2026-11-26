"""Closed-loop evaluation of the vision policy in MuJoCo.

The policy sees exactly two things: the rendered frame and the six motor
positions. The cube state is read only for resetting the scene and for scoring,
never to build the observation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from common import (MOTOR_COUNT, MotorModel, load_info, make_camera,
                    sample_cube, set_cube)
from model import make_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/bc/policy.pt")
    parser.add_argument("--data", default="data")
    parser.add_argument("--scene", default="scene.xml")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--video", type=int, default=3,
                        help="number of rollouts to record")
    parser.add_argument("--out", default="media")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def wide_camera():
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = -45.0
    cam.elevation = -21.0
    cam.distance = 1.28
    cam.lookat[:] = (0.255, -0.04, 0.19)
    return cam


class Env:
    def __init__(self, info, scene):
        self.info = info
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self.motors = MotorModel(self.model)
        self.camera = make_camera(info)
        self.renderer = mujoco.Renderer(self.model, height=info["height"],
                                        width=info["width"])
        self.options = mujoco.MjvOption()
        self.options.sitegroup[3] = 0

        self.home = np.asarray(info["home_qpos"], dtype=np.float64)
        self.place = np.asarray(info["place_position"], dtype=np.float64)
        self.kp, self.kv = info["arm_gains"]
        self.gkp, self.gkv = info["gripper_gains"]
        self.steps_per_tick = int(info["steps_per_tick"])
        self.ticks = int(info["ticks"])
        self.gains = np.array([self.kp] * 4 + [self.gkp] * 2)
        self.damping = np.array([self.kv] * 4 + [self.gkv] * 2)

    def reset(self, x, y):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.motors.qadr] = self.home
        set_cube(self.data, self.motors, x, y)
        mujoco.mj_forward(self.model, self.data)

    def observe(self):
        self.renderer.update_scene(self.data, camera=self.camera,
                                   scene_option=self.options)
        image = self.renderer.render()
        q = self.motors.qpos(self.data)
        return image, q

    def cube_xy(self):
        return self.motors.cube_xy(self.data)

    def cube_z(self):
        return float(self.data.qpos[self.motors.cube_qadr + 2])

    def apply(self, qcmd):
        qcmd = np.clip(qcmd, self.motors.joint_low, self.motors.joint_high)
        for _ in range(self.steps_per_tick):
            q = self.data.qpos[self.motors.qadr]
            qd = self.data.qvel[self.motors.dadr]
            tau = (self.data.qfrc_bias[self.motors.dadr]
                   + self.gains * (qcmd - q) - self.damping * qd)
            self.data.ctrl[self.motors.actuators] = np.clip(
                tau, self.motors.ctrl_low, self.motors.ctrl_high)
            mujoco.mj_step(self.model, self.data)


def preprocess(image, q, norm):
    x = image.astype(np.float32) / 255.0
    x = (x - norm.image_mean) / norm.image_std
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
    qn = torch.from_numpy(
        ((q - norm.q_mean) / norm.q_std).astype(np.float32)).unsqueeze(0)
    return x, qn


def success(info, cube, max_z):
    place = info["place_position"]
    distance = math.hypot(cube[0] - place[0], cube[1] - place[1])
    return distance < 0.05 and max_z > 0.08


def draw_label(draw, xy, text, color=(255, 255, 255)):
    draw.text(xy, text, fill=color, font=FONT)


def run_episode(env, policy, norm, cube_xy, record_video=None, use_image=True,
                wide=None):
    info = env.info
    env.reset(*cube_xy)

    if record_video is not None:
        record_video.mkdir(parents=True, exist_ok=True)

    max_z = 0.0
    actions = []
    for t in range(env.ticks):
        image, q = env.observe()
        if not use_image:
            image = np.full_like(image, 128)
        max_z = max(max_z, env.cube_z())

        with torch.no_grad():
            x, qn = preprocess(image, q, norm)
            prediction = policy(x, qn).numpy()[0]
        action = prediction * norm.a_std + norm.a_mean
        action = np.clip(action, -0.4, 0.4)
        actions.append(action)
        qcmd = q + action
        env.apply(qcmd)

        if wide is not None:
            wide.update_scene(env.data, camera=wide_camera(),
                               scene_option=env.options)
            panel_scene = wide.render()
            panel_policy = np.asarray(
                Image.fromarray(image).resize((512, 512), Image.NEAREST))
            canvas = Image.new("RGB", (1024, 512))
            canvas.paste(Image.fromarray(panel_policy), (0, 0))
            canvas.paste(Image.fromarray(panel_scene), (512, 0))
            draw = ImageDraw.Draw(canvas)
            draw_label(draw, (12, 12), "policy input 128x128")
            draw_label(draw, (524, 12), "scene")
            draw_label(draw, (12, 482),
                       f"episode {record_video.name[2:]}  t={t * info['control_dt']:.1f}s")
            canvas.save(record_video / f"frame_{t:04d}.png")

    if record_video is not None:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-framerate", f"{1.0 / info['control_dt']:.6g}",
             "-i", str(record_video / "frame_%04d.png"),
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p",
             str(record_video.with_suffix(".mp4"))],
            check=True)
        shutil.rmtree(record_video)

    cube = env.cube_xy()
    ok = success(info, cube, max_z)
    return {
        "cube_start": list(map(float, cube_xy)),
        "cube_end": list(map(float, cube)),
        "cube_max_z": max_z,
        "success": bool(ok),
        "mean_action": np.mean(actions, axis=0).tolist(),
    }


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    checkpoint = torch.load(args.checkpoint, map_location="cpu",
                            weights_only=False)
    model = make_policy()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    from data import Norm
    norm = Norm.from_dict(checkpoint["norm"])

    info = load_info(args.data)
    env = Env(info, args.scene)
    print(f"scene {args.scene}: {env.ticks} ticks of "
          f"{info['control_dt']:.3f}s, policy {count_params(model)} params")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cubes = [sample_cube(info, rng) for _ in range(args.episodes)]

    wide = None
    if args.video > 0:
        wide = mujoco.Renderer(env.model, height=512, width=512)

    results = []
    for i, cube in enumerate(cubes):
        video_dir = None
        if i < args.video:
            video_dir = out / "video" / f"ep{i:02d}"
        result = run_episode(env, model, norm, cube, video_dir, wide=wide)
        results.append(result)
        status = "OK " if result["success"] else "MISS"
        print(f"  episode {i:02d}: {status} "
              f"start=({cube[0]:.3f},{cube[1]:.3f}) "
              f"end=({result['cube_end'][0]:.3f},{result['cube_end'][1]:.3f}) "
              f"lift={result['cube_max_z']:.3f}")

    success_rate = sum(r["success"] for r in results) / len(results)
    print(f"policy success: {success_rate * 100:.1f}% "
          f"({sum(r['success'] for r in results)}/{len(results)})")

    # Vision ablation: same policy and cube positions, constant image.
    blind = []
    for cube in cubes[: min(args.episodes, 20)]:
        blind.append(run_episode(env, model, norm, cube, None,
                                 use_image=False))
    blind_rate = sum(r["success"] for r in blind) / len(blind)
    print(f"constant-image ablation: {blind_rate * 100:.1f}% "
          f"({sum(r['success'] for r in blind)}/{len(blind)})")

    with open(out / "metrics.json", "w") as handle:
        json.dump({
            "checkpoint": args.checkpoint,
            "episodes": results,
            "success_rate": success_rate,
            "constant_image_success_rate": blind_rate,
        }, handle, indent=2)

    print("wrote", out / "metrics.json")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


FONT = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 22)


if __name__ == "__main__":
    main()
