"""High-Speed Autonomous Minecraft PvP Combat AI Agent.

Features:
1. Snappy Proportional-Derivative (PD) mouse aim tracking (up to 150+ pixels/tick).
2. Agile PvP movement: Sprint-engaging (W + Ctrl) and Circle-Strafing (A / D alternation).
3. Instantaneous attack punch execution with configurable cooldown or spam-clicking.
4. Auto-recording combat sessions to out_vid/ for continuous model training.
5. Global F6 emergency toggle hotkey.
"""

import argparse
import ctypes
from datetime import datetime
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
import torch

from input_controller import EmergencyKillswitchListener, InputController
from model import MinecraftPvPCNN
from vision_detector import VisionDetector
from window_capture import WindowCapture, is_minecraft_window, list_windows

# Virtual Key Codes
VK_F6 = 0x75
VK_ESCAPE = 0x1B


def select_window_interactively() -> int:
    """Prompt the user to select Minecraft or active window."""
    windows = list_windows()
    if not windows:
        print("[!] No visible application windows detected.", flush=True)
        sys.exit(1)

    # Check if real Minecraft is already open
    for hwnd, title in windows:
        if is_minecraft_window(hwnd, title):
            print(f"[+] Auto-detected Minecraft window: '{title}' (HWND: {hwnd})", flush=True)
            return hwnd

    print("\n" + "=" * 60, flush=True)
    print(" ACTIVE WINDOWS AVAILABLE FOR PVP AGENT", flush=True)
    print("=" * 60, flush=True)
    for idx, (hwnd, title) in enumerate(windows, 1):
        print(f"  [{idx:2d}] {title}", flush=True)
    print("=" * 60, flush=True)

    while True:
        try:
            choice = input(f"\nSelect window [1-{len(windows)}] or search title (e.g. 'Minecraft'): ").strip()
        except EOFError:
            sys.exit(0)

        if not choice:
            continue

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(windows):
                return windows[idx - 1][0]
            continue

        matches = [(h, t) for h, t in windows if choice.lower() in t.lower()]
        if len(matches) == 1:
            print(f"[+] Matched: {matches[0][1]}", flush=True)
            return matches[0][0]
        elif len(matches) > 1:
            for i, (_, t) in enumerate(matches, 1):
                print(f"    ({i}) {t}", flush=True)
            try:
                sub_choice = input(f"Choose match [1-{len(matches)}]: ").strip()
            except EOFError:
                sys.exit(0)
            if sub_choice.isdigit() and 1 <= int(sub_choice) <= len(matches):
                return matches[int(sub_choice) - 1][0]


class PvpAgent:
    """High-speed Minecraft PvP Combat AI Agent."""

    def __init__(
        self,
        target_window: Optional[str | int] = None,
        resolution: tuple = (640, 480),
        aim_kp: float = 0.55,
        aim_kd: float = 0.12,
        max_aim_delta: float = 150.0,
        attack_cooldown: float = 0.625,  # Minecraft 1.9+ sword attack speed (0.625s)
        spam_click: bool = False,
        record_combat: bool = True,
        model_path: Optional[str] = None,
    ):
        self.width, self.height = resolution
        self.aim_kp = aim_kp
        self.aim_kd = aim_kd
        self.max_aim_delta = max_aim_delta
        self.attack_cooldown = 0.05 if spam_click else attack_cooldown
        self.record_combat = record_combat

        # Controllers & Failsafe Listener
        self.input_ctrl = InputController()
        self.killswitch = EmergencyKillswitchListener(self.input_ctrl)
        self.detector = VisionDetector(resolution)

        # PyTorch Neural Network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[+] Initializing MinecraftPvPCNN on {self.device}...", flush=True)
        self.model = MinecraftPvPCNN().to(self.device).eval()
        if model_path and os.path.exists(model_path):
            self.model.load_model(model_path, self.device)

        # Hook to Minecraft window
        if target_window is None:
            hwnd = select_window_interactively()
            self.cap = WindowCapture(hwnd)
        else:
            try:
                self.cap = WindowCapture(int(target_window))
            except ValueError:
                self.cap = WindowCapture(str(target_window))

        print(f"[+] Hooked to window: '{self.cap.window_title}' (HWND: {self.cap.hwnd})", flush=True)

        # State tracking for PD controller
        # State tracking for PD controller & W-Tap
        self.prev_dx: float = 0.0
        self.prev_dy: float = 0.0
        self.last_attack_time: float = 0.0
        self.search_rot_dir: int = 1
        self.ticks_without_target: int = 0
        self.strafe_tick: int = 0
        self.strafe_direction: str = "d"

        # W-Tap state machine (resets sprint after attacks to chain KB hits)
        self.w_tap_ticks: int = 0
        self.is_w_tapping: bool = False

        # Combat session video recorder
        self.video_writer: Optional[cv2.VideoWriter] = None
        if self.record_combat:
            os.makedirs("out_vid", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.rec_path = os.path.join("out_vid", f"combat_session_{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(self.rec_path, fourcc, 20.0, (self.width, self.height))
            print(f"[+] Recording combat session to: {self.rec_path}", flush=True)

    @property
    def is_active(self) -> bool:
        return self.killswitch.is_active

    @is_active.setter
    def is_active(self, val: bool):
        self.killswitch.set_active(val)

    def step(self, frame_bgr: np.ndarray) -> dict:
        """Execute one 20 TPS combat decision step."""
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)

        # Record raw frame to dataset if recorder is active
        if self.video_writer and self.is_active:
            self.video_writer.write(frame_bgr)

        # 1. Vision Perception
        det = self.detector.detect(frame_bgr)

        # 2. Combat Decision Logic
        actions = {
            "w": False, "s": False, "a": False, "d": False,
            "sprint": False, "jump": False, "attack": False,
            "dx": 0.0, "dy": 0.0
        }

        if det["has_target"]:
            self.ticks_without_target = 0
            cur_dx = det["dx"]
            cur_dy = det["dy"]

            # PD Aiming Calculation
            d_dx = cur_dx - self.prev_dx
            d_dy = cur_dy - self.prev_dy

            aim_x = (cur_dx * self.aim_kp) + (d_dx * self.aim_kd)
            aim_y = (cur_dy * self.aim_kp) + (d_dy * self.aim_kd)

            if abs(cur_dx) < 3.0:
                aim_x = 0.0
            if abs(cur_dy) < 3.0:
                aim_y = 0.0

            actions["dx"] = float(np.clip(aim_x, -self.max_aim_delta, self.max_aim_delta))
            actions["dy"] = float(np.clip(aim_y, -self.max_aim_delta * 0.7, self.max_aim_delta * 0.7))

            self.prev_dx = cur_dx
            self.prev_dy = cur_dy

            # W-Tap Sprint Reset Logic:
            # If in attack range and W-tap is active, release W for 2 ticks to reset sprint
            if self.is_w_tapping:
                self.w_tap_ticks += 1
                if self.w_tap_ticks >= 2:
                    self.is_w_tapping = False
                    self.w_tap_ticks = 0
                    actions["w"] = True
                    actions["sprint"] = True
                else:
                    actions["w"] = False
                    actions["sprint"] = False
            else:
                actions["w"] = True
                actions["sprint"] = True

            # Circle-Strafing in melee combat
            if det["in_attack_range"]:
                self.strafe_tick += 1
                if self.strafe_tick % 10 == 0:
                    self.strafe_direction = "a" if self.strafe_direction == "d" else "d"

                if self.strafe_direction == "a":
                    actions["a"] = True
                else:
                    actions["d"] = True

            # Attack Punch Execution (timed cooldown to avoid spam penalty)
            now = time.perf_counter()
            if det["in_attack_range"] and (now - self.last_attack_time >= self.attack_cooldown):
                actions["attack"] = True
                self.last_attack_time = now
                # Trigger W-tap reset after attack strike
                self.is_w_tapping = True
                self.w_tap_ticks = 0

        else:
            self.ticks_without_target += 1
            self.prev_dx = 0.0
            self.prev_dy = 0.0
            self.is_w_tapping = False

            # Fast search rotation: turn around quickly if enemy moved out of view
            if self.ticks_without_target > 4:
                actions["dx"] = 14.0 * self.search_rot_dir
                actions["w"] = False
                actions["sprint"] = False

        # 3. Hardware DirectInput Dispatch
        if self.is_active:
            if actions["dx"] != 0 or actions["dy"] != 0:
                self.input_ctrl.move_mouse(int(actions["dx"]), int(actions["dy"]))

            self.input_ctrl.set_movement(
                w=actions["w"],
                s=actions["s"],
                a=actions["a"],
                d=actions["d"],
                sprint=actions["sprint"],
                jump=actions["jump"],
            )

            if actions["attack"]:
                self.input_ctrl.attack_click()
        else:
            self.input_ctrl.release_all()

        return {"detection": det, "actions": actions, "active": self.is_active}

    def run(self, preview: bool = True):
        """High-precision 20 TPS combat loop."""
        tick_interval = 1.0 / 20.0
        print("\n" + "=" * 60, flush=True)
        print(" MINECRAFT HIGH-SPEED PVP COMBAT AI", flush=True)
        print("=" * 60, flush=True)
        print(f"  Aim Speed:        P-Gain: {self.aim_kp} | Max Delta: {self.max_aim_delta} px/tick", flush=True)
        print(f"  Attack Cooldown:  {self.attack_cooldown:.2f}s (Minecraft 1.9+ anti-spam timed)", flush=True)
        print(f"  Combat Recording: {'ENABLED (' + self.rec_path + ')' if self.video_writer else 'DISABLED'}", flush=True)
        print("=" * 60, flush=True)
        print("  Controls:", flush=True)
        print("    [F6]    : TOGGLE AI ON / OFF (Audio Beep Confirmation)", flush=True)
        print("    [ESC]   : INSTANT EMERGENCY STOP (Pauses and releases all keys)", flush=True)
        print("    [q]     : Quit Agent (in HUD window)", flush=True)
        print("    [Ctrl+C]: Stop Agent in terminal", flush=True)
        print("=" * 60, flush=True)
        print("[!] Press F6 in Minecraft to ACTIVATE high-speed combat!\n", flush=True)

        next_tick = time.perf_counter()

        try:
            while True:
                if not self.cap.is_valid():
                    print("[!] Target window closed. Stopping agent.", flush=True)
                    break

                active = self.is_active

                success, frame = self.cap.get_frame()
                if success and frame is not None:
                    step_result = self.step(frame)

                    if preview:
                        det = step_result["detection"]
                        actions = step_result["actions"]
                        hud = self.detector.draw_hud(frame, det, actions)

                        state_color = (0, 255, 0) if active else (0, 255, 255)
                        state_txt = "AI: FIGHTING [F6: Pause | ESC: STOP]" if active else "AI: PAUSED [Press F6 in MC to Start]"
                        cv2.putText(hud, state_txt, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)

                        # Attack Cooldown Meter
                        now = time.perf_counter()
                        elapsed = now - self.last_attack_time
                        ratio = min(1.0, elapsed / max(0.001, self.attack_cooldown))
                        bars = int(ratio * 10)
                        meter_str = f"Atk Cooldown: [{'|' * bars}{'.' * (10 - bars)}] {int(ratio * 100)}%"
                        meter_color = (0, 255, 0) if ratio >= 0.85 else (0, 0, 255)
                        cv2.putText(hud, meter_str, (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.48, meter_color, 1)

                        # W-tap status badge
                        wtap_txt = "W-TAP RESETTING" if self.is_w_tapping else "SPRINT LOCKED"
                        wtap_col = (0, 255, 255) if self.is_w_tapping else (200, 200, 200)
                        cv2.putText(hud, f"Status: {wtap_txt}", (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.48, wtap_col, 1)

                        cv2.imshow("Minecraft PvP AI - High-Speed Combat HUD (Press 'q' to stop)", hud)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                next_tick += tick_interval
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0.001:
                    time.sleep(sleep_time * 0.95)
                while time.perf_counter() < next_tick:
                    pass

        except KeyboardInterrupt:
            print("\n[+] Stop signal (Ctrl+C) received.", flush=True)
        finally:
            self.killswitch.stop()
            self.input_ctrl.release_all()
            if self.video_writer:
                self.video_writer.release()
                print(f"[+] Combat recording saved: {self.rec_path}", flush=True)
            self.cap.close()
            cv2.destroyAllWindows()
            print("[+] PvP Agent cleanly terminated. All keys released.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="High-Speed Responsive Minecraft PvP Combat AI Agent.")
    parser.add_argument("-w", "--window", type=str, default=None, help="Target window title (default: auto-detect Minecraft)")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model weights (.pth)")
    parser.add_argument("--kp", type=float, default=0.55, help="Aim P-gain sensitivity (default: 0.55)")
    parser.add_argument("--kd", type=float, default=0.12, help="Aim D-gain damping (default: 0.12)")
    parser.add_argument("--max-delta", type=float, default=150.0, help="Max aim delta px/tick (default: 150.0)")
    parser.add_argument("--cooldown", type=float, default=0.35, help="Attack punch cooldown in seconds (default: 0.35s)")
    parser.add_argument("--spam-click", action="store_true", help="Enable rapid spam-clicking (1.8 PvP mode)")
    parser.add_argument("--no-record", action="store_true", help="Disable automatic combat session recording")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview HUD window")

    args = parser.parse_args()
    agent = PvpAgent(
        target_window=args.window,
        resolution=(640, 480),
        aim_kp=args.kp,
        aim_kd=args.kd,
        max_aim_delta=args.max_delta,
        attack_cooldown=args.cooldown,
        spam_click=args.spam_click,
        record_combat=not args.no_record,
        model_path=args.model,
    )
    agent.run(preview=not args.no_preview)


if __name__ == "__main__":
    main()
