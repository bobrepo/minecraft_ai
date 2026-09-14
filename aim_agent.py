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
    ):
        self.width, self.height = resolution
        self.horizontal_only = True
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

        # 2. Smooth PD Controller Parameters (Calibrated for high-FPS tracking)
        dt_scale = 20.0 / float(self.fps)
        self.kp_yaw = 0.42 * dt_scale
        self.kd_yaw = 0.12 * dt_scale
        self.max_step_px = 24.0 * dt_scale  # Max slew speed to ensure slow, controlled gliding

        self.prev_dx = 0.0

        # 3. Desktop Overlay
        self.overlay: Optional[PvPOverlayClient] = None
        if self.use_overlay:
            self.overlay = PvPOverlayClient()
            self.overlay.start()

    @property
    def is_active(self) -> bool:
        return self.killswitch.is_active

    def run(self):
        """Locked high-FPS horizontal tracking loop."""
        tick_interval = 1.0 / float(self.fps)
        print("\n" + "=" * 65, flush=True)
        print(" MINECRAFT STAGE 1: AIM AGENT (HORIZONTAL ONLY)", flush=True)
        print("=" * 65, flush=True)
        print("  Curriculum:      STAGE 1: AIMING ONLY (Body is stationary)", flush=True)
        print(f"  Update Rate:     {self.fps} FPS ({tick_interval * 1000:.1f}ms interval)", flush=True)
        print("  Vision:          Crosshair-Centric Outward Scanning (Sub-0.5ms)", flush=True)
        print("  Target Cues:     Cyan Glow + Red/Purple Crosshair Sensor", flush=True)
        print("  Aim Mode:        HORIZONTAL ONLY (Yaw Gliding)", flush=True)
        print("=" * 65, flush=True)
        print("  Controls:", flush=True)
        print("    [F6]    : TOGGLE AIM ON / OFF (Audio Beep Feedback)", flush=True)
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
                        elif cmd.get("action") == "quit":
                            return

                active = self.is_active

                if active:
                    _was_active = True

                    # 1. Capture latest frame (<0.01ms async)
                    suc, frame = self.cap.get_frame()
                    if not suc or frame is None:
                        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    elif frame.shape[1] != self.width or frame.shape[0] != self.height:
                        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

                    # 2. Crosshair-Centric Outward Detection (sub-0.5ms)
                    det = self.detector.detect(frame, crosshair_centric=True)

                    vx = 0.0

                    if det["has_target"]:
                        dx = det["dx"]

                        # Proportional-Derivative Smooth Tracking
                        d_dx = dx - self.prev_dx
                        raw_vx = self.kp_yaw * dx + self.kd_yaw * d_dx
                        vx = float(np.clip(raw_vx, -self.max_step_px, self.max_step_px))
                        self.prev_dx = dx

                        # 3. Smooth Sub-Tick Mouse Dispatch (Horizontal Yaw only)
                        # If crosshair is already red (locked on enemy) or within deadzone, hold steady
                        is_locked = det.get("crosshair_locked", False)
                        if is_locked or abs(dx) <= 2.0:
                            pass  # Dead-center lock achieved
                        else:
                            self.input_ctrl.move_mouse(vx, 0.0, dynamic=True)
                    else:
                        self.prev_dx = 0.0

                    det_info = det
                    step_dx = int(vx)
                else:
                    if _was_active:
                        self.input_ctrl.release_all(force=True)
                        _was_active = False
                    self.prev_dx = 0.0
                    det_info = {"has_target": False, "dist_px": 0.0, "in_lock_zone": False}
                    step_dx = 0

                # Update Desktop Overlay
                if self.overlay:
                    self.overlay.update(
                        active=active,
                        dx=step_dx,
                        dy=0,
                        target_locked=det_info.get("in_lock_zone", False),
                        target_dist=float(det_info.get("distance", 0.0)),
                        reward=10.0 if det_info.get("in_lock_zone", False) else 0.0,
                        total_score=0.0,
                        tps=current_tps,
                        phase="AIM:HORIZ",
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
    parser = argparse.ArgumentParser(description="Aim Agent for Minecraft (Horizontal Only, 60 FPS)")
    parser.add_argument("-w", "--window", type=str, default=None, help="Target window title / HWND")
    parser.add_argument("--no-overlay", action="store_true", help="Disable desktop overlay")
    parser.add_argument("--fps", type=int, default=60, help="Aim tracking rate in FPS (default: 60)")
    args = parser.parse_args()

    agent = AimAgent(
        target_window=args.window,
        use_overlay=not args.no_overlay,
        fps=args.fps,
    )
    agent.run()


if __name__ == "__main__":
    main()
