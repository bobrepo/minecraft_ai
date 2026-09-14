"""Supervised Behavioral Cloning & GPU Pretrainer for Minecraft PvP.

Pretrains the BranchingQNetwork (BDQ) on human demonstration datasets from train_videos/
using synchronized video frames and input telemetry (WASD, Sprint, Jump, Attack, Aim dx/dy).
Saves pretrained weights directly to models/rl_pvp_model.pth.
"""

import argparse
import glob
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rl_agent import BranchingQNetwork
from vision_detector import VisionDetector

# BDQ Discrete Aim Vectors (matches rl_agent.py)
AIM_DELTAS = [
    (0.0, 0.0),       # 0: Center / None
    (-35.0, -15.0),   # 1: Diagonal Up-Left
    (-35.0, 15.0),    # 2: Diagonal Down-Left
    (35.0, -15.0),    # 3: Diagonal Up-Right
    (35.0, 15.0),     # 4: Diagonal Down-Right
    (-55.0, 0.0),     # 5: Fast Horizontal Left
    (55.0, 0.0),      # 6: Fast Horizontal Right
]


def map_aim_to_branch(dx: float, dy: float) -> int:
    """Map continuous camera delta (dx, dy) to nearest discrete BDQ aim branch action."""
    mag = math.hypot(dx, dy)
    if mag < 5.0:
        return 0  # Dead-center hold
    best_act = 0
    best_dist = float("inf")
    for idx, (adx, ady) in enumerate(AIM_DELTAS):
        d = (dx - adx) ** 2 + (dy - ady) ** 2
        if d < best_dist:
            best_dist = d
            best_act = idx
    return best_act


def map_move_to_branch(w: bool, s: bool, a: bool, d: bool, sprint: bool) -> int:
    """Map movement inputs to discrete BDQ movement branch action [0..5]."""
    if s and not w:
        return 5  # Back S
    elif a and not d:
        return 3  # Strafe Left A
    elif d and not a:
        return 4  # Strafe Right D
    elif w and sprint:
        return 2  # Sprint W
    elif w:
        return 1  # Walk W
    return 0      # Idle


class PvPTelemetryDataset(Dataset):
    """Dataset extracting 240x320 frames and discrete BDQ action targets from gameplay sessions."""

    def __init__(self, data_dirs: List[str], max_samples: Optional[int] = None):
        self.samples: List[Tuple[np.ndarray, int, int, int, int]] = []
        detector = VisionDetector((640, 480))

        # Find all MP4 video files
        all_videos = []
        for d in data_dirs:
            if os.path.exists(d):
                all_videos.extend(glob.glob(os.path.join(d, "*.mp4")))

        print(f"[+] Found {len(all_videos)} session video(s) across: {data_dirs}", flush=True)

        for vpath in all_videos:
            json_path = vpath.replace(".mp4", "_actions.json")
            has_telemetry = os.path.exists(json_path)

            telemetry_data = []
            if has_telemetry:
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        telemetry_data = json.load(f)
                except Exception as e:
                    print(f"[!] Warning: Could not read {json_path} ({e}), falling back to vision labels.", flush=True)
                    has_telemetry = False

            cap = cv2.VideoCapture(vpath)
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Input resolution for BranchingQNetwork: (240, 320)
                resized = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                if has_telemetry and frame_idx < len(telemetry_data):
                    t = telemetry_data[frame_idx]
                    dx = float(t.get("dx", 0.0))
                    dy = float(t.get("dy", 0.0))
                    w = bool(t.get("w", False))
                    s = bool(t.get("s", False))
                    a = bool(t.get("a", False))
                    d = bool(t.get("d", False))
                    sprint = bool(t.get("sprint", False))
                    jump = bool(t.get("jump", False))
                    attack = bool(t.get("attack", False))

                    aim_act = map_aim_to_branch(dx, dy)
                    move_act = map_move_to_branch(w, s, a, d, sprint)
                    jump_act = 1 if jump else 0
                    atk_act = 1 if attack else 0
                else:
                    # Fallback auto-labeling via computer vision detector
                    det = detector.detect(frame)
                    dx = det["dx"]
                    dy = det["dy"]
                    aim_act = map_aim_to_branch(dx, dy)
                    move_act = 2 if det["has_target"] else 0
                    jump_act = 1 if (det["in_attack_range"] and random.random() < 0.25) else 0
                    atk_act = 1 if det["in_attack_range"] else 0

                self.samples.append((rgb, aim_act, move_act, jump_act, atk_act))
                frame_idx += 1

                if max_samples and len(self.samples) >= max_samples:
                    break

            cap.release()

        print(f"[+] Loaded {len(self.samples)} combat frames ready for Behavioral Cloning training.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb, aim_act, move_act, jump_act, atk_act = self.samples[idx]

        # Augmentation 1: Random horizontal flip (50% chance)
        if random.random() > 0.5:
            rgb = np.ascontiguousarray(np.fliplr(rgb))
            # Swap aim Left/Right actions
            flip_aim_map = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 6, 6: 5}
            aim_act = flip_aim_map.get(aim_act, aim_act)
            # Swap move A/D actions
            flip_move_map = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 5}
            move_act = flip_move_map.get(move_act, move_act)

        # Augmentation 2: Random brightness variation (±12%)
        if random.random() > 0.5:
            factor = random.uniform(0.88, 1.12)
            rgb = np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Convert to Tensor (3, 240, 320) normalized to [0.0, 1.0]
        img_tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float() / 255.0

        return (
            img_tensor,
            torch.tensor(aim_act, dtype=torch.long),
            torch.tensor(move_act, dtype=torch.long),
            torch.tensor(jump_act, dtype=torch.long),
            torch.tensor(atk_act, dtype=torch.long),
        )


def train(
    data_dirs: Optional[List[str]] = None,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 2e-4,
    save_path: str = "models/rl_pvp_model.pth",
):
    """Pre-train BranchingQNetwork using multi-task Cross-Entropy on human demonstration data."""
    if data_dirs is None:
        data_dirs = ["train_videos", "out_vid"]

    dataset = PvPTelemetryDataset(data_dirs)
    if len(dataset) < 8:
        print(f"[!] No training sessions found in {data_dirs}. Record gameplay using run_cap_game.bat first!", flush=True)
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n========================================================", flush=True)
    print(f" BEHAVIORAL CLONING PRETRAINER (BranchingQNetwork)", flush=True)
    print(f" Device:  {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f" Samples: {len(dataset)} frames | Epochs: {epochs} | Batch: {batch_size}", flush=True)
    print(f" Target:  {os.path.abspath(save_path)}", flush=True)
    print(f"========================================================\n", flush=True)

    model = BranchingQNetwork().to(device)

    # Load existing weights if available to continuously improve
    if os.path.exists(save_path):
        try:
            model.load_state_dict(torch.load(save_path, map_location=device))
            print(f"[+] Loaded existing network weights from: {save_path} (continual pre-training)", flush=True)
        except Exception as e:
            print(f"[!] Warning: Could not load previous weights ({e}), starting fresh.", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_ce = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    model.train()
    start_total = time.perf_counter()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        aim_correct, move_correct, jump_correct, atk_correct = 0, 0, 0, 0
        total_samples = 0
        t0 = time.perf_counter()

        for imgs, aim_tgt, move_tgt, jump_tgt, atk_tgt in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            aim_tgt = aim_tgt.to(device, non_blocking=True)
            move_tgt = move_tgt.to(device, non_blocking=True)
            jump_tgt = jump_tgt.to(device, non_blocking=True)
            atk_tgt = atk_tgt.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                aim_q, move_q, jump_q, atk_q = model(imgs)

                loss_aim = loss_ce(aim_q, aim_tgt)
                loss_move = loss_ce(move_q, move_tgt)
                loss_jump = loss_ce(jump_q, jump_tgt)
                loss_atk = loss_ce(atk_q, atk_tgt)
                loss = loss_aim + loss_move + loss_jump + loss_atk

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)

            # Accuracy metrics
            aim_correct += (aim_q.argmax(dim=1) == aim_tgt).sum().item()
            move_correct += (move_q.argmax(dim=1) == move_tgt).sum().item()
            jump_correct += (jump_q.argmax(dim=1) == jump_tgt).sum().item()
            atk_correct += (atk_q.argmax(dim=1) == atk_tgt).sum().item()
            total_samples += imgs.size(0)

        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / max(1, total_samples)
        acc_aim = (aim_correct / total_samples) * 100.0
        acc_move = (move_correct / total_samples) * 100.0
        acc_atk = (atk_correct / total_samples) * 100.0

        print(
            f" Epoch [{epoch:02d}/{epochs:02d}] | Loss: {avg_loss:.4f} | "
            f"Aim Acc: {acc_aim:4.1f}% | Move Acc: {acc_move:4.1f}% | Atk Acc: {acc_atk:4.1f}% | Time: {elapsed:.2f}s",
            flush=True,
        )

    total_time = time.perf_counter() - start_total
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n[+] Training successfully completed in {total_time:.1f}s!", flush=True)
    print(f"[+] Pretrained weights saved to: {os.path.abspath(save_path)}", flush=True)
    print("[+] You can now launch run_rl.bat to continue learning with RL on top of these weights!\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Pretrain Minecraft Branching Q-Network on gameplay sessions")
    parser.add_argument("--data-dirs", nargs="+", default=["train_videos", "out_vid"], help="Directories with recordings")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="models/rl_pvp_model.pth", help="Path to save weights")
    args = parser.parse_args()

    train(
        data_dirs=args.data_dirs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
