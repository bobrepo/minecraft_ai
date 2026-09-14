"""Pure Reinforcement Learning Aim Environment for Minecraft (Stage 1 Curriculum).

In Stage 1, the bot focuses 100% on learning mouse aim:
- Initial curriculum: Horizontal Movements Only (Yaw alignment).
- Extensible to full 2D Aim (Yaw + Pitch) via horizontal_only toggle.
- Crosshair-Centric Outward Scanning for sub-0.5ms target detection.
- Sub-0.01ms asynchronous DirectX/OpenGL window capture.
- Smooth damped mouse dispatch to prevent jerky flicking.
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from fast_aim_detector import FastAimDetector
from input_controller import InputController
from window_capture import AsyncWindowCapture, WindowCapture


class MinecraftAimEnv:
    """Stage 1 Gym-style Pure Aim Environment for Minecraft PvP."""

    # Branch 1: Horizontal Yaw deltas in pixels per tick (Smooth graduations)
    YAW_ACTIONS = [-32.0, -16.0, -6.0, -2.0, 0.0, 2.0, 6.0, 16.0, 32.0]
    # Branch 2: Vertical Pitch deltas in pixels per tick
    PITCH_ACTIONS = [-18.0, -8.0, -2.5, 0.0, 2.5, 8.0, 18.0]

    def __init__(
        self,
        target_hwnd: Optional[int | str] = None,
        resolution: Tuple[int, int] = (640, 480),
        max_episode_steps: int = 200,
        horizontal_only: bool = True,
    ):
        self.width, self.height = resolution
        self.max_episode_steps = max_episode_steps
        self.horizontal_only = horizontal_only
        self.diag = float(np.hypot(self.width, self.height))

        # Hardware capture & input
        try:
            self.cap = AsyncWindowCapture(target_hwnd if target_hwnd is not None else "Minecraft")
        except Exception:
            self.cap = WindowCapture(target_hwnd if target_hwnd is not None else "Minecraft")

        self.detector = FastAimDetector(resolution)
        self.input_ctrl = InputController()

        # State tracking
        self.current_step = 0
        self.prev_dist_px = float(self.diag / 2.0)
        self.prev_dx = float(self.width / 2.0)
        self.prev_target_x = float(self.width / 2.0)
        self.prev_target_y = float(self.height / 2.0)
        self.prev_yaw_act = 0.0
        self.prev_pitch_act = 0.0
        self.lock_streak = 0
        self.target_seen_ticks = 0

    def set_horizontal_only(self, val: bool):
        """Toggle between Horizontal-only aim and full 2D aim."""
        self.horizontal_only = val

    def reset(self) -> np.ndarray:
        """Reset environment episode counter and return initial observation."""
        self.current_step = 0
        self.prev_dist_px = float(self.diag / 2.0)
        self.prev_dx = float(self.width / 2.0)
        self.prev_yaw_act = 0.0
        self.prev_pitch_act = 0.0
        self.lock_streak = 0
        self.target_seen_ticks = 0

        # Capture initial frame
        suc, frame = self.cap.get_frame()
        if not suc or frame is None:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        elif frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        det = self.detector.detect(frame, crosshair_centric=True)
        return self._build_state(det)

    def _build_state(self, det: Dict[str, Any]) -> np.ndarray:
        """Build compact 10-dimensional normalized observation vector."""
        w2 = self.width / 2.0
        h2 = self.height / 2.0

        if det["has_target"]:
            dx = det["dx"]
            dy = det["dy"]
            dist_px = det["dist_px"]
            box_h = det["box_h"]

            # Target pixel velocity across screen
            tx = det["target_x"]
            ty = det["target_y"]
            vx = tx - self.prev_target_x
            vy = ty - self.prev_target_y
            self.prev_target_x = tx
            self.prev_target_y = ty

            self.target_seen_ticks += 1
            if det["in_lock_zone"] or (self.horizontal_only and abs(dx) <= 16.0):
                self.lock_streak += 1
            else:
                self.lock_streak = max(0, self.lock_streak - 1)
        else:
            dx = 0.0
            dy = 0.0
            dist_px = self.diag / 2.0
            box_h = 0.0
            vx = 0.0
            vy = 0.0
            self.lock_streak = 0
            self.target_seen_ticks = 0

        state = np.array(
            [
                float(np.clip(dx / w2, -1.0, 1.0)),
                float(np.clip(dy / h2, -1.0, 1.0)),
                float(np.clip(dist_px / (self.diag / 2.0), 0.0, 1.0)),
                float(np.clip(box_h / self.height, 0.0, 1.0)),
                float(np.clip(vx / 25.0, -1.0, 1.0)),
                float(np.clip(vy / 25.0, -1.0, 1.0)),
                float(np.clip(self.prev_yaw_act / 32.0, -1.0, 1.0)),
                float(np.clip(self.prev_pitch_act / 18.0, -1.0, 1.0)),
                1.0 if det["has_target"] else 0.0,
                float(np.clip(self.lock_streak / 20.0, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

        return state

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one pure aim environment step.

        Args:
            action: Tuple (yaw_index [0-8], pitch_index [0-6]).

        Returns:
            (next_state, reward, done, info)
        """
        self.current_step += 1
        yaw_idx, pitch_idx = action
        yaw_delta = self.YAW_ACTIONS[yaw_idx]
        pitch_delta = self.PITCH_ACTIONS[pitch_idx] if not self.horizontal_only else 0.0

        # 1. Dispatch smooth damped camera rotation via hardware SendInput
        if yaw_delta != 0.0 or pitch_delta != 0.0:
            self.input_ctrl.move_mouse(yaw_delta, pitch_delta, dynamic=True)

        # 2. Capture updated frame (<0.01ms async)
        suc, frame = self.cap.get_frame()
        if not suc or frame is None:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        elif frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # 3. Detect updated target position via crosshair-centric search
        det = self.detector.detect(frame, crosshair_centric=True)
        next_state = self._build_state(det)

        # 4. Compute Dense Alignment Reward
        reward = 0.0
        w2 = self.width / 2.0
        d_max = self.diag / 2.0

        if det["has_target"]:
            dx = det["dx"]
            dy = det["dy"]
            dist_px = det["dist_px"]

            if self.horizontal_only:
                # Stage 1: Dense horizontal alignment reward
                norm_dx = min(1.0, abs(dx) / w2)
                r_align = 4.0 * ((1.0 - norm_dx) ** 1.5)  # 0 to +4.0
                r_progress = 6.0 * ((abs(self.prev_dx) - abs(dx)) / w2)
                r_lock = 4.0 if abs(dx) <= 16.0 else 0.0
                r_streak = min(3.0, self.lock_streak * 0.2)
                d_act = abs(yaw_delta - self.prev_yaw_act)
                r_smooth = -0.01 * (d_act / 32.0)
                reward = float(r_align + r_progress + r_lock + r_streak + r_smooth)
                self.prev_dx = dx
            else:
                # Full 2D aim alignment
                norm_dist = min(1.0, dist_px / d_max)
                r_align = 4.0 * ((1.0 - norm_dist) ** 1.5)
                r_progress = 5.0 * ((self.prev_dist_px - dist_px) / d_max)
                r_lock = 5.0 if det["in_lock_zone"] else 0.0
                r_streak = min(3.0, self.lock_streak * 0.2)
                d_act = abs(yaw_delta - self.prev_yaw_act) + abs(pitch_delta - self.prev_pitch_act)
                r_smooth = -0.015 * (d_act / 40.0)
                reward = float(r_align + r_progress + r_lock + r_streak + r_smooth)

            self.prev_dist_px = dist_px
        else:
            if det.get("is_facing_sky", False):
                reward = -2.5
            else:
                reward = -0.5

        self.prev_yaw_act = yaw_delta
        self.prev_pitch_act = pitch_delta

        # Done condition
        done = self.current_step >= self.max_episode_steps

        info = {
            "detection": det,
            "reward": reward,
            "dist_px": det.get("dist_px", 999.0),
            "in_lock_zone": det.get("in_lock_zone", False) or (self.horizontal_only and abs(det.get("dx", 999.0)) <= 16.0),
            "lock_streak": self.lock_streak,
            "horizontal_only": self.horizontal_only,
            "step": self.current_step,
        }

        return next_state, reward, done, info

    def close(self):
        """Release resources."""
        self.cap.close()
