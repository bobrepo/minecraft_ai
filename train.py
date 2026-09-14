"""Self-supervised trainer for MinecraftPvPCNN using recorded gameplay videos.

Iterates through recorded videos in out_vid/, uses VisionDetector to generate
ground-truth labels (aim dx/dy, movement W/A/S/D, and attack punch), and trains
the neural network weights using PyTorch on your GPU.
"""

import glob
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import MinecraftPvPCNN
from vision_detector import VisionDetector


class VideoPvPDataset(Dataset):
    """Dataset that extracts frames and auto-generated combat labels from recorded videos."""

    def __init__(self, video_paths, resolution=(640, 480), max_frames_per_video=600):
        self.samples = []
        self.width, self.height = resolution
        detector = VisionDetector(resolution)

        print(f"[+] Loading and auto-labeling videos from {len(video_paths)} file(s)...", flush=True)

        for vpath in video_paths:
            cap = cv2.VideoCapture(vpath)
            frame_idx = 0
            while cap.isOpened() and frame_idx < max_frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

                # Generate label from vision detector
                det = detector.detect(frame)
                has_target = 1.0 if det["has_target"] else 0.0
                dx = float(np.clip(det["dx"] * 0.22, -40.0, 40.0))
                dy = float(np.clip(det["dy"] * 0.22, -40.0, 40.0))
                w = 1.0 if det["has_target"] else 0.0
                attack = 1.0 if det["in_attack_range"] else 0.0

                # Store RGB frame and target vectors
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.samples.append((rgb, (dx, dy, w, 0.0, 0.0, 0.0, attack, has_target)))
                frame_idx += 1

            cap.release()

        print(f"[+] Extracted {len(self.samples)} labeled frames for training.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb, labels = self.samples[idx]
        # Tensor: (3, H, W) normalized [0, 1]
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        dx, dy, w, s, a, d, attack, has_target = labels
        aim_target = torch.tensor([dx, dy], dtype=torch.float32)
        move_target = torch.tensor([w, s, a, d], dtype=torch.float32)
        attack_target = torch.tensor([attack], dtype=torch.float32)
        target_target = torch.tensor([has_target], dtype=torch.float32)

        return img_tensor, aim_target, move_target, attack_target, target_target


def train_model(video_dir="out_vid", epochs=15, batch_size=8, lr=1e-4, save_path="models/pvp_model.pth"):
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    if not video_files:
        print(f"[!] No recorded .mp4 files found in '{video_dir}'. Record gameplay first using run_capture.bat!")
        return

    dataset = VideoPvPDataset(video_files)
    if len(dataset) < 10:
        print("[!] Dataset too small for training. Record longer gameplay video.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MinecraftPvPCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    print(f"\n[+] Starting training on {device} for {epochs} epochs...", flush=True)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        start_time = time.perf_counter()

        for imgs, aim_tgt, move_tgt, atk_tgt, tgt_tgt in dataloader:
            imgs = imgs.to(device)
            aim_tgt = aim_tgt.to(device)
            move_tgt = move_tgt.to(device)
            atk_tgt = atk_tgt.to(device)
            tgt_tgt = tgt_tgt.to(device)

            optimizer.zero_grad()
            pred_aim, pred_move, pred_atk, pred_tgt = model(imgs)

            loss_aim = mse_loss(pred_aim, aim_tgt)
            loss_move = bce_loss(pred_move, move_tgt)
            loss_atk = bce_loss(pred_atk, atk_tgt)
            loss_tgt = bce_loss(pred_tgt, tgt_tgt)

            loss = loss_aim + loss_move + loss_atk + loss_tgt
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        elapsed = time.perf_counter() - start_time
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s", flush=True)

    model.save_model(save_path)
    print(f"\n[+] Training complete! Model saved to '{save_path}'.")


if __name__ == "__main__":
    train_model()
