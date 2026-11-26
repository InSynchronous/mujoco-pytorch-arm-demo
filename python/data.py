"""Dataset index, normalization stats, and a memmapped frame dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import MOTOR_COUNT, load_info, shard_dirs


@dataclass
class Norm:
    image_mean: np.ndarray  # (3,)
    image_std: np.ndarray   # (3,)
    q_mean: np.ndarray      # (6,)
    q_std: np.ndarray       # (6,)
    a_mean: np.ndarray      # (6,)
    a_std: np.ndarray       # (6,)

    def to_dict(self) -> dict:
        return {
            "image_mean": self.image_mean.tolist(),
            "image_std": self.image_std.tolist(),
            "q_mean": self.q_mean.tolist(),
            "q_std": self.q_std.tolist(),
            "a_mean": self.a_mean.tolist(),
            "a_std": self.a_std.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Norm":
        return cls(
            image_mean=np.asarray(data["image_mean"], dtype=np.float32),
            image_std=np.asarray(data["image_std"], dtype=np.float32),
            q_mean=np.asarray(data["q_mean"], dtype=np.float32),
            q_std=np.asarray(data["q_std"], dtype=np.float32),
            a_mean=np.asarray(data["a_mean"], dtype=np.float32),
            a_std=np.asarray(data["a_std"], dtype=np.float32),
        )


def load_index(data_dir: str | Path, with_qvel: bool = False):
    """Concatenate shard CSVs and frame memmaps.

    The returned arrays are in the same row order as the raw frame stream.
    cube_x/cube_y are kept for analysis only and are *not* model inputs.
    """
    info = load_info(data_dir)
    height, width = info["height"], info["width"]

    q_list, qd_list, a_list, cube_list = [], [], [], []
    episode_list, frame_list, shard_list = [], [], []
    frames = []
    offset = 0

    for shard_id, shard in enumerate(shard_dirs(data_dir)):
        steps = pd.read_csv(shard / "steps.csv")
        episodes = pd.read_csv(shard / "episodes.csv")
        successful = set(episodes.loc[episodes["success"] == 1, "episode"].tolist())
        mask = steps["episode"].isin(successful)
        steps = steps.loc[mask].reset_index(drop=True)
        if steps.empty:
            continue

        raw = shard / "frames.bin"
        nbytes = raw.stat().st_size
        count = nbytes // (height * width * 3)
        expected = len(steps) * height * width * 3
        if nbytes != expected:
            raise RuntimeError(
                f"{raw}: {nbytes} bytes but {len(steps)} frames of "
                f"{height}x{width} need {expected}"
            )
        frames.append(np.memmap(raw, dtype=np.uint8, mode="r",
                                shape=(count, height, width, 3)))
        if count != len(steps):
            raise RuntimeError(
                f"{raw}: {count} frames but {len(steps)} CSV rows")

        q_cols = [f"q{i}" for i in range(MOTOR_COUNT)]
        qd_cols = [f"qd{i}" for i in range(MOTOR_COUNT)]
        a_cols = [f"a{i}" for i in range(MOTOR_COUNT)]

        q_list.append(steps[q_cols].to_numpy(np.float32))
        qd_list.append(steps[qd_cols].to_numpy(np.float32))
        a_list.append(steps[a_cols].to_numpy(np.float32))
        cube_list.append(steps[["cube_x", "cube_y"]].to_numpy(np.float32))
        episode_list.append(steps["episode"].to_numpy(np.int32))
        frame_list.append(steps["frame"].to_numpy(np.int16))
        shard_list.append(np.full(len(steps), shard_id, dtype=np.int16))
        offset += len(steps)

    if not frames:
        raise RuntimeError("no successful episodes found")

    data = {
        "info": info,
        "frames": frames,
        "q": np.concatenate(q_list),
        "qdot": np.concatenate(qd_list),
        "action": np.concatenate(a_list),
        "cube": np.concatenate(cube_list),
        "episode": np.concatenate(episode_list),
        "frame": np.concatenate(frame_list),
        "shard": np.concatenate(shard_list),
    }
    data["row_in_shard"] = np.concatenate(
        [np.arange(len(s), dtype=np.int32) for s in q_list]
    )
    return data


def split_rows(episode: np.ndarray, val_every: int = 20):
    """Split by episode, never by frame."""
    episodes = np.unique(episode)
    val_episodes = set(episodes[::val_every].tolist())
    is_val = np.isin(episode, list(val_episodes))
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def compute_norm(data, train_rows: np.ndarray, rng: np.random.Generator) -> Norm:
    q = data["q"][train_rows]
    a = data["action"][train_rows]
    q_std = np.maximum(q.std(axis=0), 1e-6)
    a_std = np.maximum(a.std(axis=0), 1e-6)

    sample = rng.choice(train_rows, size=min(2000, len(train_rows)), replace=False)
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for row in sample:
        image = data["frames"][data["shard"][row]][data["row_in_shard"][row]]
        x = image.astype(np.float64) / 255.0
        total += x.reshape(-1, 3).sum(axis=0)
        total_sq += (x.reshape(-1, 3) ** 2).sum(axis=0)
        count += x.shape[0] * x.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 1e-8))

    return Norm(
        image_mean=mean.astype(np.float32),
        image_std=np.maximum(std, 1e-3).astype(np.float32),
        q_mean=q.mean(axis=0).astype(np.float32),
        q_std=q_std.astype(np.float32),
        a_mean=a.mean(axis=0).astype(np.float32),
        a_std=a_std.astype(np.float32),
    )


class FrameDataset(Dataset):
    """Reads one frame + q/action row from the memmapped shard files."""

    def __init__(self, data, rows: np.ndarray, norm: Norm, with_qvel=False):
        self.data = data
        self.rows = rows
        self.norm = norm
        self.with_qvel = with_qvel

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = int(self.rows[index])
        shard = int(self.data["shard"][row])
        local = int(self.data["row_in_shard"][row])
        image = np.asarray(self.data["frames"][shard][local], dtype=np.float32)
        image = image / 255.0
        image = (image - self.norm.image_mean) / self.norm.image_std
        image = torch.from_numpy(image.transpose(2, 0, 1))

        q = (self.data["q"][row] - self.norm.q_mean) / self.norm.q_std
        action = (self.data["action"][row] - self.norm.a_mean) / self.norm.a_std
        extra = ()
        if self.with_qvel:
            extra = (
                torch.from_numpy((self.data["qdot"][row] - 0.0) / 1.0),
            )
        return (image, torch.from_numpy(q.astype(np.float32)),
                torch.from_numpy(action.astype(np.float32)), *extra)
