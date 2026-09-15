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

    # Branch 1: Horizontal Yaw deltas in pixels per tick (17-action granular spectrum from micro-aim to hyper-speed flicks)
    YAW_ACTIONS = [
        -160.0, -110.0, -75.0, -50.0, -32.0, -18.0, -8.0, -3.0,
          0.0,
          3.0,   8.0,  18.0,  32.0,  50.0,  75.0, 110.0, 160.0,
    ]
    # Branch 2: Vertical Pitch deltas in pixels per tick
    PITCH_ACTIONS = [-18.0, -8.0, -2.5, 0.0, 2.5, 8.0, 18.0]

    def __init__(
        self,
        target_hwnd: Optional[int | str] = None,
        resolution: Tuple[int, int] = (640, 480),
        max_episode_steps: int = 200,
        horizontal_only: bool = True,
        min_lock_ticks: int = 10,
    ):
        self.width, self.height = resolution
        self.max_episode_steps = max_episode_steps
        self.horizontal_only = horizontal_only
        self.min_lock_ticks = min_lock_ticks
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
        self.speed_multiplier = 1.0
        self.prev_dist_px = float(self.diag / 2.0)
        self.prev_dx = float(self.width / 2.0)
        self.prev_target_x = float(self.width / 2.0)
        self.prev_target_y = float(self.height / 2.0)
        self.prev_yaw_act = 0.0
        self.prev_pitch_act = 0.0
        self.lock_streak = 0
        self.consecutive_moving_ticks = 0
        self.target_seen_ticks = 0
        self.off_target_ticks = 0
        self.prev_has_target = False

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
        self.consecutive_moving_ticks = 0
        self.target_seen_ticks = 0
        self.off_target_ticks = 0
        self.prev_has_target = False

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

            # Target pixel velocity across screen (uses 1€ filtered velocity if available)
            if "vx" in det and "vy" in det:
                vx = float(det["vx"])
                vy = float(det["vy"])
                self.prev_target_x = det.get("target_x", self.prev_target_x)
                self.prev_target_y = det.get("target_y", self.prev_target_y)
            else:
                tx = det["target_x"]
                ty = det["target_y"]
                vx = tx - self.prev_target_x
                vy = ty - self.prev_target_y
                self.prev_target_x = tx
                self.prev_target_y = ty
        else:
            dx = 0.0
            dy = 0.0
            dist_px = self.diag / 2.0
            box_h = 0.0
            vx = 0.0
            vy = 0.0
            self.target_seen_ticks = 0

        state = np.array(
            [
                float(np.clip(dx / w2, -1.0, 1.0)),
                float(np.clip(dy / h2, -1.0, 1.0)),
                float(np.clip(dist_px / (self.diag / 2.0), 0.0, 1.0)),
                float(np.clip(box_h / self.height, 0.0, 1.0)),
                float(np.clip(vx / 25.0, -1.0, 1.0)),
                float(np.clip(vy / 25.0, -1.0, 1.0)),
                float(np.clip(self.prev_yaw_act / 160.0, -1.0, 1.0)),
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
            action: Tuple (yaw_index [0-16], pitch_index [0-6]).

        Returns:
            (next_state, reward, done, info)
        """
        self.current_step += 1
        yaw_idx, pitch_idx = action
        yaw_delta = self.YAW_ACTIONS[yaw_idx]
        pitch_delta = self.PITCH_ACTIONS[pitch_idx] if not self.horizontal_only else 0.0

        # Motion detection: check if camera is moving
        is_moving = (abs(yaw_delta) > 0.0) or (abs(pitch_delta) > 0.0)
        if is_moving:
            self.consecutive_moving_ticks += 1
        else:
            self.consecutive_moving_ticks = 0

        # Continuous movement check: moving for >= 2 consecutive ticks, OR flicking/sweeping fast
        is_continuously_moving = (self.consecutive_moving_ticks >= 2) or (abs(yaw_delta) > 3.0) or (abs(pitch_delta) > 2.5)

        # 1. Dispatch camera rotation via direct 1:1 hardware SendInput (0ms phase lag)
        eff_yaw = yaw_delta * self.speed_multiplier
        eff_pitch = pitch_delta * self.speed_multiplier
        if eff_yaw != 0.0 or eff_pitch != 0.0:
            self.input_ctrl.move_mouse(eff_yaw, eff_pitch, dynamic=False)

        # 2. Capture updated frame (<0.01ms async)
        suc, frame = self.cap.get_frame()
        if not suc or frame is None:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        elif frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # 3. Detect updated target position via crosshair-centric search
        det = self.detector.detect(frame, crosshair_centric=True)

        # 4. Compute Reward (View Reward + Heavy Procrastination Penalty + Dwell Lock Bonus)
        reward = 0.0
        w2 = self.width / 2.0
        d_max = self.diag / 2.0

        if det["has_target"]:
            dx = det["dx"]
            dy = det["dy"]
            dist_px = det["dist_px"]

            is_locked = bool(det.get("in_lock_zone", False) or det.get("crosshair_locked", False))

            if is_locked:
                self.off_target_ticks = 0
                if is_continuously_moving:
                    # STRICT RULE: No points for continuously moving or sweeping across enemy!
                    # Reset dwell streak so sweeping across enemy cannot accumulate dwell lock time
                    self.lock_streak = 0
                    reward = 0.0
                elif is_moving:
                    # 1-tick micro-adjustment while on target: 0.0 reward while moving
                    reward = 0.0
                else:
                    # Stationary hold on enemy: accumulate dwell time!
                    self.lock_streak += 1
                    if self.lock_streak >= self.min_lock_ticks:
                        # Crosshair held over enemy for sufficient dwell duration
                        r_lock = 8.0
                        r_streak = min(4.0, (self.lock_streak - self.min_lock_ticks) * 0.20)
                        reward = float(r_lock + r_streak)
                    else:
                        # Still acquiring / dwelling, not yet held for long enough: 0.0 points
                        reward = 0.0
            else:
                # Enemy in view, but crosshair NOT on target -> immediate streak reset
                self.lock_streak = 0
                self.off_target_ticks += 1

                # Directional Progress / Regression Check:
                if self.prev_has_target:
                    curr_err = abs(dx) if self.horizontal_only else dist_px
                    prev_err = abs(self.prev_dx) if self.horizontal_only else self.prev_dist_px
                    delta_err = curr_err - prev_err  # > 0 means crosshair moved AWAY from enemy
                else:
                    delta_err = 0.0

                if delta_err > 0.5:
                    # Moving AWAY from target (e.g. enemy is left, bot moved right, or overshot)
                    # Immediate negative penalty proportional to error increase
                    r_away = 1.5 + min(3.5, delta_err * 0.2)
                    reward = float(-r_away)
                elif self.off_target_ticks > 20:
                    # After grace period: heavy penalty for being late & staring without locking on!
                    slack_time = (self.off_target_ticks - 20) / 60.0
                    penalty = 2.0 + min(3.0, slack_time * 2.0)  # -2.0 to -5.0 pts/tick
                    reward = float(-penalty)
                else:
                    # Moving towards target during grace period: strictly 0.0 (no free on-screen points)
                    reward = 0.0

            self.prev_dx = dx
            self.prev_dist_px = dist_px
            self.prev_has_target = True
        else:
            self.lock_streak = 0
            self.off_target_ticks = 0
            self.prev_has_target = False
            reward = -0.5

        next_state = self._build_state(det)

        self.prev_yaw_act = yaw_delta
        self.prev_pitch_act = pitch_delta

        # Done condition
        done = self.current_step >= self.max_episode_steps

        is_locked_state = bool(det.get("in_lock_zone", False) or det.get("crosshair_locked", False))
        info = {
            "detection": det,
            "reward": reward,
            "dist_px": det.get("dist_px", 999.0),
            "in_lock_zone": is_locked_state,
            "lock_streak": self.lock_streak,
            "is_moving": is_moving,
            "is_continuously_moving": is_continuously_moving,
            "horizontal_only": self.horizontal_only,
            "step": self.current_step,
        }

        return next_state, reward, done, info

    def close(self):
        """Release resources."""
        self.cap.close()
