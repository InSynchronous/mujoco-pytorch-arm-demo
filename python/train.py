"""Train the behavior-cloning policy on collected expert episodes."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import MOTOR_COUNT
from data import FrameDataset, compute_norm, load_index, split_rows
from model import make_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="runs/bc")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--val-every", type=int, default=20,
                        help="hold out every Nth episode")
    parser.add_argument("--max-train", type=int, default=0,
                        help="subsample training rows (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def evaluate(model, loader, norm):
    model.eval()
    total = 0.0
    count = 0
    sq = np.zeros(MOTOR_COUNT, dtype=np.float64)
    with torch.no_grad():
        for image, q, action in loader:
            prediction = model(image, q)
            loss = torch.mean((prediction - action) ** 2)
            total += loss.item() * image.shape[0]
            count += image.shape[0]
            raw_pred = prediction.numpy() * norm.a_std + norm.a_mean
            raw_true = action.numpy() * norm.a_std + norm.a_mean
            sq += ((raw_pred - raw_true) ** 2).sum(axis=0)
    rmse = np.sqrt(sq / max(count, 1))
    return total / max(count, 1), rmse


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_index(args.data)
    info = data["info"]
    if info["width"] != 128 or info["height"] != 128:
        raise SystemExit("this model expects 128x128 frames")

    train_rows, val_rows = split_rows(data["episode"], args.val_every)
    if args.max_train and len(train_rows) > args.max_train:
        rng = np.random.default_rng(args.seed)
        train_rows = rng.choice(train_rows, args.max_train, replace=False)
    if len(val_rows) == 0:
        raise SystemExit("empty validation split")

    norm = compute_norm(data, train_rows, np.random.default_rng(args.seed))
    print(f"frames: {len(data['q'])}, train: {len(train_rows)}, "
          f"val: {len(val_rows)}")
    print("action std:", np.round(norm.a_std, 5))

    train_ds = FrameDataset(data, train_rows, norm)
    val_ds = FrameDataset(data, val_rows, norm)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None,
        drop_last=False)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None,
        drop_last=False)

    model = make_policy()
    print("parameters:", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)
    loss_fn = nn.MSELoss()

    history = []
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        seen = 0
        for step, (image, q, action) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image, q)
            loss = loss_fn(prediction, action)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * image.shape[0]
            seen += image.shape[0]
            if (step + 1) % 100 == 0:
                rate = seen / (time.time() - started)
                print(f"  epoch {epoch} step {step + 1}/{len(train_loader)} "
                      f"loss {running / seen:.5f}  {rate:.0f} frames/s")
        train_loss = running / seen
        val_loss, rmse = evaluate(model, val_loader, norm)
        scheduler.step()
        elapsed = time.time() - started
        print(f"epoch {epoch}: train {train_loss:.5f}  val {val_loss:.5f}  "
              f"rmse {np.round(rmse, 4)}  ({elapsed:.0f}s)")
        history.append((epoch, train_loss, val_loss, *rmse))

        checkpoint = {
            "model": model.state_dict(),
            "norm": norm.to_dict(),
            "info": info,
            "epoch": epoch,
            "val_loss": val_loss,
            "args": vars(args),
        }
        torch.save(checkpoint, out / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(checkpoint, out / "policy.pt")
            print("  saved best policy.pt")

    with open(out / "metrics.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train", "val", "rmse0", "rmse1", "rmse2",
                         "rmse3", "rmse4", "rmse5"])
        writer.writerows(history)

    epochs = [row[0] for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [row[1] for row in history], label="train")
    plt.plot(epochs, [row[2] for row in history], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE (normalized action)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "loss.png", dpi=120)
    print("wrote", out / "policy.pt")


if __name__ == "__main__":
    main()
