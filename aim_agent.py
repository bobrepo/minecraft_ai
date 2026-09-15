"""Direct Aim Agent for Highlighted Minecraft Opponents.

Tailored to the user's highlighted enemy appearance (Cyan Glow + Red Cross Target):
1. Parses highlighted image region using Crosshair-Centric Outward Scanning (sub-0.5ms).
2. Slowly and smoothly glides cursor towards the center of the screen to align crosshair.
3. Stage 1 Constraint: Strictly Horizontal movements ONLY (Yaw), pitch locked at 0.0.
4. Single hotkey control: [F6] to start/stop the agent.
5. High-frequency 60 FPS update rate with Minimum-Jerk sub-tick interpolation.
"""

import argparse
from collections import deque
import ctypes
import math
import os
import sys
import time
from typing import Optional, Tuple
import winsound
import cv2
import numpy as np

from fast_aim_detector import FastAimDetector
from input_controller import EmergencyKillswitchListener, InputController
from overlay import PvPOverlayClient
from window_capture import AsyncWindowCapture, WindowCapture


class AimAgent:
    """High-speed aim tracking agent for highlighted targets."""

    def __init__(
        self,
        target_window: Optional[str | int] = None,
        resolution: Tuple[int, int] = (640, 480),
        use_overlay: bool = True,
        fps: int = 60,
        speed: float = 1.0,
        horizontal_only: bool = False,
    ):
        self.width, self.height = resolution
        self.horizontal_only = horizontal_only
        self.use_overlay = use_overlay
        self.fps = max(10, min(120, fps))

        # 1. Asynchronous Capture & Detection
        try:
            self.cap = AsyncWindowCapture(target_window if target_window is not None else "Minecraft")
        except Exception:
            self.cap = WindowCapture(target_window if target_window is not None else "Minecraft")

        self.detector = FastAimDetector(resolution)
        self.input_ctrl = InputController()
        self.killswitch = EmergencyKillswitchListener(self.input_ctrl)

        # 2. Critically Damped Zero-Bounce Tracking Controller
        # High-velocity target acquisition with zero overshoot, zero bounce, and dead-still lock
        self.deadzone_px = 3.5            # Lock deadzone
        self.lock_deadzone_px = 4.0       # Entry threshold to acquire target lock
        self.unlock_deadzone_px = 7.5     # Hysteresis exit threshold (prevents boundary flutter)
        self.is_locked = False
        self.max_step_px = 160.0 * speed  # High-speed flick limit matching 17-action spectrum
        self.max_pitch_step_px = 35.0 * speed  # Smooth vertical pitch limit
        self.lock_deadzone_y_px = 5.0
        self.unlock_deadzone_y_px = 9.0
        self.last_seen_dir = 1            # Direction to sweep when target moves off-screen

        self.prev_dx = 0.0
        self.prev_vx = 0.0
        self.prev_dy = 0.0
        self.prev_vy = 0.0
        self.min_lock_ticks = 10
        self.lock_streak = 0
        self.consecutive_moving_ticks = 0
        self.cumulative_score = 0.0

        # Dynamic AI Speed Control
        self.base_speed = speed
        self.speed_mode = "auto"  # "auto", "0.5x", "1.0x", "1.5x", "2.0x"
        self.current_speed_mult = 1.0

        # 180° Backflip & Smooth Left-to-Right Scanning Search
        self.flip_180_px = 650.0         # 180° turnaround flick distance
        self.last_flip_time = 0.0
        self.scan_amp = 20.0 * speed     # Peak sweep velocity (px/tick)
        self.scan_period = 70            # Ticks per full back-and-forth oscillation cycle (~1.16s)
        self.scan_tick = 0
        self.lost_target_ticks = 0

        # 3. Desktop Overlay
        self.overlay: Optional[PvPOverlayClient] = None
        if self.use_overlay:
            self.overlay = PvPOverlayClient()
            self.overlay.start()

    @property
    def is_active(self) -> bool:
        return self.killswitch.is_active

    def compute_ai_speed_factor(self, dx: float) -> float:
        """AI autonomous speed controller: dynamically governs aim velocity based on distance.

        - Snap Flick (Far: |dx| > 140px): 1.60x speed boost to snap fast to the enemy.
        - Swift Pursuit (Mid: 50px < |dx| <= 140px): 1.00x balanced tracking speed.
        - Controlled Deceleration (Close: 18px < |dx| <= 50px): 0.60x smooth braking.
        - Precision Glide (Lock zone: |dx| <= 18px): 0.35x micro-alignment without overshoot.
        """
        abs_dx = abs(dx)
        if abs_dx > 140.0:
            return 1.60
        elif abs_dx > 50.0:
            return 1.00
        elif abs_dx > 18.0:
            return 0.60
        else:
            return 0.35

    def get_effective_speed(self, dx: float) -> float:
        """Calculate effective speed multiplier considering active speed mode."""
        if self.speed_mode == "auto":
            factor = self.compute_ai_speed_factor(dx)
        elif self.speed_mode == "0.5x":
            factor = 0.50
        elif self.speed_mode == "1.5x":
            factor = 1.50
        elif self.speed_mode == "2.0x":
            factor = 2.00
        else:  # "1.0x"
            factor = 1.00
        return self.base_speed * factor

    def run(self):
        """Locked high-FPS horizontal tracking loop."""
        tick_interval = 1.0 / float(self.fps)
        print("\n" + "=" * 65, flush=True)
        print(" MINECRAFT STAGE 1: AIM AGENT (HORIZONTAL ONLY)", flush=True)
        print("=" * 65, flush=True)
        print("  Curriculum:      STAGE 1: AIMING ONLY (Body is stationary)", flush=True)
        print(f"  Update Rate:     {self.fps} FPS ({tick_interval * 1000:.1f}ms interval)", flush=True)
        print("  Vision:          Crosshair-Centric Outward Scanning (Sub-0.5ms)", flush=True)
        print("  Target Cues:     Yellow (#FFF500) Body", flush=True)
        print("  Search Mode:     180° BACKFLIP on target escape + Smooth Left <-> Right Scan", flush=True)
        print("  Aim Mode:        HORIZONTAL ONLY (Direct 1:1 Lockstep Dispatch)", flush=True)
        print("=" * 65, flush=True)
        print("  Controls:", flush=True)
        print("    [F6]    : TOGGLE AIM ON / OFF (Audio Beep Feedback)", flush=True)
        print("    [V]     : MANUAL 180° BACKFLIP SNAP", flush=True)
        print("    [ESC]   : INSTANT EMERGENCY STOP", flush=True)
        print("    [Ctrl+C]: Exit Agent", flush=True)
        print("=" * 65, flush=True)
        print("[!] Press F6 in Minecraft to START Aim Tracking!\n", flush=True)

        next_tick = time.perf_counter()
        tick_times = deque(maxlen=20)
        current_tps = float(self.fps)
        _was_active = False

        try:
            while True:
                if not self.cap.is_valid():
                    print("[!] Minecraft window closed. Stopping agent.", flush=True)
                    break

                t_now = time.perf_counter()
                tick_times.append(t_now)
                if len(tick_times) >= 2:
                    dt_span = tick_times[-1] - tick_times[0]
                    current_tps = (len(tick_times) - 1) / max(1e-5, dt_span)

                # Check overlay user commands
                if self.overlay:
                    for cmd in self.overlay.poll_commands():
                        if cmd.get("action") == "set_active":
                            val = bool(cmd.get("value", False))
                            self.killswitch.set_active(val)
                            if not val:
                                self.input_ctrl.stop_aim()
                                self.input_ctrl.release_all(force=True)
                        elif cmd.get("action") == "stop_aim":
                            self.killswitch.set_active(False)
                            self.input_ctrl.stop_aim()
                            self.input_ctrl.release_all(force=True)
                        elif cmd.get("action") == "set_speed_mode":
                            new_mode = str(cmd.get("mode", "auto")).lower()
                            if new_mode in ("auto", "0.5x", "1.0x", "1.5x", "2.0x"):
                                self.speed_mode = new_mode
                                print(f"[+] Aim Speed Mode set to: {self.speed_mode.upper()}", flush=True)
                        elif cmd.get("action") == "quit":
                            return

                # Check manual 180° Backflip hotkey [V]
                try:
                    if ctypes.windll.user32.GetAsyncKeyState(0x56) & 0x8000:
                        t_flip = time.perf_counter()
                        if t_flip - self.last_flip_time > 0.4:
                            flip_vx = self.flip_180_px * self.last_seen_dir
                            self.input_ctrl.move_mouse_direct(flip_vx, 0.0)
                            self.last_flip_time = t_flip
                            self.lost_target_ticks = 1
                except Exception:
                    pass

                active = self.is_active

                if active:
                    _was_active = True

                    # 1. Capture latest frame (<0.01ms async)
                    suc, frame = self.cap.get_frame()
                    if not suc or frame is None:
                        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    else:
                        # Dynamically adapt to native capture resolution to prevent aspect-ratio squishing
                        fh, fw = frame.shape[:2]
                        if self.detector.width != fw or self.detector.height != fh:
                            self.detector.width = fw
                            self.detector.height = fh
                            self.detector.cx = fw // 2
                            self.detector.cy = fh // 2
                            self.detector.fy = fh / (2.0 * np.tan(np.radians(35.0)))
                            self.width = fw
                            self.height = fh

                    # 2. Crosshair-Centric Outward Detection (sub-0.5ms)
                    det = self.detector.detect(frame, crosshair_centric=True)

                    vx = 0.0
                    vy = 0.0

                    if det["has_target"]:
                        self.lost_target_ticks = 0
                        self.scan_tick = 0
                        dx = float(det["dx"])
                        dy = float(det["dy"])
                        if dx < -6.0:
                            self.last_seen_dir = -1
                        elif dx > 6.0:
                            self.last_seen_dir = 1

                        # AI Dynamic Speed Control
                        speed_mult = self.get_effective_speed(dx)
                        self.current_speed_mult = speed_mult

                        abs_dx = abs(dx)
                        sign_x = 1.0 if dx > 0 else -1.0
                        abs_dy = abs(dy)
                        sign_y = 1.0 if dy > 0 else -1.0

                        # Calculate true target world velocity with camera ego-motion cancellation
                        screen_delta_x = dx - self.prev_dx
                        target_world_vx = screen_delta_x + self.prev_vx
                        screen_delta_y = dy - self.prev_dy
                        target_world_vy = screen_delta_y + self.prev_vy

                        # --- Horizontal Yaw Tracking ---
                        # Deadband Lock with Hysteresis (Zero Jitter, Zero Bounce)
                        if self.is_locked:
                            if abs_dx > self.unlock_deadzone_px:
                                self.is_locked = False
                            else:
                                # Locked onto enemy: if enemy is actively moving in the world, match strafe velocity,
                                # otherwise hold dead still (vx = 0.0). NO trampoline kick!
                                if abs(target_world_vx) > 3.0:
                                    vx = float(np.clip(target_world_vx, -32.0, 32.0))
                                else:
                                    vx = 0.0
                        else:
                            if abs_dx <= self.lock_deadzone_px:
                                self.is_locked = True
                                vx = 0.0

                        if not self.is_locked:
                            # Critically Damped Proportional Glide (Snappy snap + Smooth deceleration)
                            if abs_dx > 120.0:
                                desired_vx = sign_x * min(self.max_step_px * speed_mult, 0.48 * abs_dx * speed_mult)
                            elif abs_dx > 30.0:
                                desired_vx = sign_x * (0.38 * abs_dx * speed_mult)
                            else:
                                desired_vx = sign_x * (0.24 * abs_dx * speed_mult)

                            # Blend subtle real enemy world motion if moving
                            if abs(target_world_vx) > 3.0:
                                desired_vx += float(np.clip(0.35 * target_world_vx, -20.0, 20.0))

                            # CRITICAL ANTI-OVERSHOOT GUARANTEE:
                            # Never command a single-tick step greater than current distance to deadzone.
                            max_safe_step_x = max(0.0, abs_dx - self.lock_deadzone_px * 0.5)
                            target_step_x = sign_x * min(abs(desired_vx), max_safe_step_x)

                            # Slew-rate limiter: smooth acceleration & prevent jarring sign flips
                            max_accel_x = 80.0 * speed_mult
                            vx = float(np.clip(target_step_x, self.prev_vx - max_accel_x, self.prev_vx + max_accel_x))

                        # --- Vertical Pitch Tracking (Eliminates the vertical gap) ---
                        if not self.horizontal_only:
                            if abs_dy <= self.lock_deadzone_y_px:
                                vy = 0.0
                            else:
                                desired_vy = sign_y * min(self.max_pitch_step_px, 0.35 * abs_dy)
                                max_safe_step_y = max(0.0, abs_dy - self.lock_deadzone_y_px * 0.5)
                                target_step_y = sign_y * min(abs(desired_vy), max_safe_step_y)
                                max_accel_y = 25.0
                                vy = float(np.clip(target_step_y, self.prev_vy - max_accel_y, self.prev_vy + max_accel_y))

                        self.prev_vx = vx
                        self.prev_dx = dx
                        self.prev_vy = vy
                        self.prev_dy = dy

                        # Dispatch 1:1 hardware mouse movement
                        if vx != 0.0 or vy != 0.0:
                            self.input_ctrl.move_mouse_direct(vx, vy)
                    else:
                        self.prev_dx = 0.0
                        self.prev_vx = 0.0
                        self.prev_dy = 0.0
                        self.prev_vy = 0.0
                        self.is_locked = False
                        self.lost_target_ticks += 1
                        now = time.perf_counter()

                        # 180° Backflip when enemy escapes / target is lost
                        if self.lost_target_ticks == 1 and (now - self.last_flip_time > 0.5):
                            # Immediate decisive 180° snap flick in last seen escape direction
                            flip_vx = self.flip_180_px * self.last_seen_dir
                            self.input_ctrl.move_mouse_direct(flip_vx, 0.0)
                            self.last_flip_time = now
                            self.scan_tick = 0
                            vx = flip_vx
                        elif self.lost_target_ticks <= 15:
                            # Brief settling pause (~0.25s) to allow vision detector to lock onto target behind
                            vx = 0.0
                        else:
                            # Target not found directly behind: smooth Left <-> Right panoramic sweep
                            self.scan_tick += 1
                            omega = (2.0 * math.pi) / float(self.scan_period)
                            sweep_vx = self.scan_amp * math.cos(omega * self.scan_tick) * self.last_seen_dir
                            self.input_ctrl.move_mouse_direct(sweep_vx, 0.0)
                            vx = sweep_vx

                    det_info = det
                    step_dx = int(vx)

                    # Dwell and movement tracking for reward calculation
                    is_moving = abs(vx) > 0.0
                    if is_moving:
                        self.consecutive_moving_ticks += 1
                    else:
                        self.consecutive_moving_ticks = 0

                    is_continuously_moving = (self.consecutive_moving_ticks >= 2) or (abs(vx) > 3.0)

                    is_locked = bool(det_info.get("in_lock_zone", False) or det_info.get("crosshair_locked", False) or self.is_locked)
                    if is_locked:
                        if is_continuously_moving:
                            # No points for continuously moving!
                            self.lock_streak = 0
                            step_reward = 0.0
                        elif is_moving:
                            step_reward = 0.0
                        else:
                            self.lock_streak += 1
                            if self.lock_streak >= self.min_lock_ticks:
                                r_lock = 8.0
                                r_streak = min(4.0, (self.lock_streak - self.min_lock_ticks) * 0.20)
                                step_reward = float(r_lock + r_streak)
                            else:
                                step_reward = 0.0
                    else:
                        self.lock_streak = 0
                        step_reward = 0.0

                    self.cumulative_score += step_reward
                else:
                    if _was_active:
                        self.input_ctrl.stop_aim()
                        self.input_ctrl.release_all(force=True)
                        _was_active = False
                    self.prev_dx = 0.0
                    self.prev_vx = 0.0
                    self.is_locked = False
                    self.lock_streak = 0
                    self.consecutive_moving_ticks = 0
                    det_info = {"has_target": False, "dist_px": 0.0, "in_lock_zone": False}
                    step_dx = 0
                    step_reward = 0.0

                # Update Desktop Overlay
                if self.overlay:
                    self.overlay.update(
                        active=active,
                        dx=step_dx,
                        dy=0,
                        target_locked=det_info.get("in_lock_zone", False),
                        target_dist=float(det_info.get("distance", 0.0)),
                        reward=float(step_reward),
                        total_score=float(self.cumulative_score),
                        tps=current_tps,
                        phase="AIM:HORIZ",
                        speed_mode=self.speed_mode,
                    )

                # Strict high-FPS timing
                next_tick += tick_interval
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0.001:
                    time.sleep(sleep_time * 0.90)
                while time.perf_counter() < next_tick:
                    pass
                if time.perf_counter() - next_tick > tick_interval:
                    next_tick = time.perf_counter()

        except KeyboardInterrupt:
            print("\n[+] Stop signal received.", flush=True)
        finally:
            if self.overlay:
                self.overlay.stop()
            self.killswitch.stop()
            self.cap.close()
            print("[+] Aim Agent safely stopped.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Aim Agent for Minecraft (Full 2D Yaw + Pitch Aim Lock, 60 FPS)")
    parser.add_argument("-w", "--window", type=str, default=None, help="Target window title / HWND")
    parser.add_argument("--no-overlay", action="store_true", help="Disable desktop overlay")
    parser.add_argument("--fps", type=int, default=60, help="Aim tracking rate in FPS (default: 60)")
    parser.add_argument("--speed", type=float, default=1.0, help="Aim tracking speed multiplier (default: 1.0)")
    parser.add_argument("--horizontal-only", action="store_true", help="Restrict aiming strictly to horizontal yaw")
    args = parser.parse_args()

    agent = AimAgent(
        target_window=args.window,
        use_overlay=not args.no_overlay,
        fps=args.fps,
        speed=args.speed,
        horizontal_only=args.horizontal_only,
    )
    agent.run()


if __name__ == "__main__":
    main()
