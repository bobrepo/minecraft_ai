"""Multi-task PyTorch CNN for Minecraft PvP combat decision-making.

Takes visual screen frames (640x480) and outputs:
1. Mouse aim deltas (dx, dy)
2. Directional movement keys (W, S, A, D)
3. Attack punch trigger (Left Click)
4. Target detection confidence
"""

import os
from typing import Any, Dict, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn


class MinecraftPvPCNN(nn.Module):
    """Deep convolutional neural network for real-time Minecraft PvP decisions."""

    def __init__(self, max_mouse_delta: float = 40.0):
        super().__init__()
        self.max_mouse_delta = max_mouse_delta

        # Convolutional Feature Extractor
        self.backbone = nn.Sequential(
            # Stage 1: Initial downsampling
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 2: Feature learning
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 3: High-level visual patterns (player silhouette, nametag, hitbox)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        # Shared Decision Trunk
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # Head 1: Continuous 3D Mouse Aiming (dx, dy)
        self.aim_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Tanh(),  # Normalized to [-1.0, 1.0]
        )

        # Head 2: Movement Keys (W, S, A, D)
        self.movement_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),  # Multi-label output for W, S, A, D
        )

        # Head 3: Left Click Punch / Attack
        self.attack_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Head 4: Target Detection Confidence
        self.target_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (Batch, 3, Height, Width) normalized to [0, 1].

        Returns:
            aim_delta: (Batch, 2) in range [-max_mouse_delta, max_mouse_delta]
            movement_logits: (Batch, 4) for W, S, A, D
            attack_logit: (Batch, 1)
            target_logit: (Batch, 1)
        """
        features = self.backbone(x)
        latent = self.trunk(features)

        aim_delta = self.aim_head(latent) * self.max_mouse_delta
        movement_logits = self.movement_head(latent)
        attack_logit = self.attack_head(latent)
        target_logit = self.target_head(latent)

        return aim_delta, movement_logits, attack_logit, target_logit

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray, device: torch.device) -> Dict[str, Any]:
        """Convenience method to run inference directly on a raw OpenCV frame.

        Args:
            frame_bgr: BGR uint8 NumPy array from WindowCapture.
            device: torch.device ('cuda' or 'cpu').

        Returns:
            Dictionary with parsed action predictions.
        """
        # Preprocessing: convert BGR to RGB, normalize to [0, 1]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(device)

        aim_delta, movement_logits, attack_logit, target_logit = self.forward(tensor)

        # Parse continuous aim
        dx, dy = aim_delta[0].cpu().numpy()

        # Parse movement probabilities
        move_probs = torch.sigmoid(movement_logits[0]).cpu().numpy()
        w = bool(move_probs[0] > 0.5)
        s = bool(move_probs[1] > 0.5)
        a = bool(move_probs[2] > 0.5)
        d = bool(move_probs[3] > 0.5)

        # Parse attack and target confidence
        attack_prob = torch.sigmoid(attack_logit[0, 0]).item()
        target_conf = torch.sigmoid(target_logit[0, 0]).item()

        return {
            "dx": float(dx),
            "dy": float(dy),
            "w": w,
            "s": s,
            "a": a,
            "d": d,
            "attack": bool(attack_prob > 0.5),
            "attack_prob": attack_prob,
            "has_target": bool(target_conf > 0.5),
            "confidence": target_conf,
        }

    def save_model(self, path: str):
        """Save model state dictionary to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[+] Saved model weights to: {path}")

    def load_model(self, path: str, device: torch.device):
        """Load model state dictionary from disk."""
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=device))
            self.to(device)
            self.eval()
            print(f"[+] Loaded model weights from: {path}")
        else:
            print(f"[!] Model file not found at: {path}, using initial weights.")
