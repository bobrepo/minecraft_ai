"""Fast GPU Trainer for MinecraftPvPCNN using recorded combat sessions.

Trains the multi-task CNN on RTX 3050 using mixed precision (FP16 autocast)
and data augmentation (horizontal flips with inverted aim, brightness variation).
Saves model weights directly to models/pvp_model.pth.
"""

import glob
import os
import random
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import MinecraftPvPCNN
from vision_detector import VisionDetector


class AugmentedPvPDataset(Dataset):
    """Dataset extracting combat frames with automatic vision labels and augmentations."""

    def __init__(self, video_files, resolution=(640, 480), max_frames_per_video=1000):
        self.samples = []
        self.width, self.height = resolution
        detector = VisionDetector(resolution)

        print(f"[+] Loading combat sessions from {len(video_files)} video file(s)...", flush=True)

        for vpath in video_files:
            cap = cv2.VideoCapture(vpath)
            frame_count = 0
            while cap.isOpened() and frame_count < max_frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

                # Ground truth from vision detector
                det = detector.detect(frame)
                has_target = 1.0 if det["has_target"] else 0.0
                dx = float(np.clip(det["dx"] * 0.55, -150.0, 150.0))
                dy = float(np.clip(det["dy"] * 0.55, -100.0, 100.0))
                w = 1.0 if det["has_target"] else 0.0
                attack = 1.0 if det["in_attack_range"] else 0.0

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.samples.append((rgb, (dx, dy, w, 0.0, 0.0, 0.0, attack, has_target)))
                frame_count += 1

            cap.release()

        print(f"[+] Extracted {len(self.samples)} combat frames ready for GPU training.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb, labels = self.samples[idx]
        dx, dy, w, s, a, d, attack, has_target = labels

        # Data Augmentation: 50% chance horizontal flip
        if random.random() > 0.5:
            rgb = np.ascontiguousarray(np.fliplr(rgb))
            dx = -dx  # Invert horizontal aim delta
            a, d = d, a  # Swap left and right strafing

        # Data Augmentation: subtle random brightness jitter
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            rgb = np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Convert to Tensor (3, H, W) normalized to [0, 1]
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        aim_target = torch.tensor([dx, dy], dtype=torch.float32)
        move_target = torch.tensor([w, s, a, d], dtype=torch.float32)
        attack_target = torch.tensor([attack], dtype=torch.float32)
        target_target = torch.tensor([has_target], dtype=torch.float32)

        return img_tensor, aim_target, move_target, attack_target, target_target


def train(video_dir: str = "out_vid", epochs: int = 12, batch_size: int = 16, lr: float = 3e-4, save_path: str = "models/pvp_model.pth"):
    """Train MinecraftPvPCNN using mixed precision on GPU."""
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    if not video_files:
        print(f"[!] No recorded videos found in '{video_dir}'. Play Minecraft with the bot or record with run_capture.bat first!", flush=True)
        return

    dataset = AugmentedPvPDataset(video_files)
    if len(dataset) < 10:
        print("[!] Not enough frames to train. Play and record more combat sessions.", flush=True)
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MinecraftPvPCNN().to(device)
    # Load previous weights if available to continuously learn
    if os.path.exists(save_path):
        try:
            model.load_state_dict(torch.load(save_path, map_location=device))
            print(f"[+] Loaded existing model weights from: {save_path} (continual learning)", flush=True)
        except Exception:
            pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    print(f"\n========================================================", flush=True)
    print(f" TRAINING MINECRAFT PVP CNN ON {device.type.upper()}", flush=True)
    print(f" Samples: {len(dataset)} | Epochs: {epochs} | Batch: {batch_size}", flush=True)
    print(f"========================================================\n", flush=True)

    model.train()
    start_all = time.perf_counter()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        t0 = time.perf_counter()

        for imgs, aim_tgt, move_tgt, atk_tgt, tgt_tgt in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            aim_tgt = aim_tgt.to(device, non_blocking=True)
            move_tgt = move_tgt.to(device, non_blocking=True)
            atk_tgt = atk_tgt.to(device, non_blocking=True)
            tgt_tgt = tgt_tgt.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred_aim, pred_move, pred_atk, pred_tgt = model(imgs)
                loss_aim = mse_loss(pred_aim, aim_tgt)
                loss_move = bce_loss(pred_move, move_tgt)
                loss_atk = bce_loss(pred_atk, atk_tgt)
                loss_tgt = bce_loss(pred_tgt, tgt_tgt)
                loss = loss_aim + loss_move + loss_atk + loss_tgt

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / len(dataloader)
        print(f" Epoch [{epoch:02d}/{epochs:02d}] | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s", flush=True)

    total_time = time.perf_counter() - start_all
    model.save_model(save_path)
    print(f"\n[+] Training complete in {total_time:.1f}s! Saved weights to: {os.path.abspath(save_path)}", flush=True)


if __name__ == "__main__":
    train()
