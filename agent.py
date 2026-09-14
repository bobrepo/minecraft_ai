"""Autonomous Minecraft PvP Combat AI Agent.

Combines real-time 20 TPS window capture, VisionDetector, and DirectInput
controls (W, A, S, D, mouse aiming, and attack punches) to engage enemy players.
Features a global F6 toggle hotkey so you can start/stop the AI while playing.
"""

import argparse
import ctypes
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
import torch

from input_controller import InputController
from model import MinecraftPvPCNN
from vision_detector import VisionDetector
from window_capture import WindowCapture, list_windows

# Virtual Key Codes
VK_F6 = 0x75
VK_ESCAPE = 0x1B


def is_key_pressed(vk_code: int) -> bool:
    """Check if a physical key is currently pressed globally via Windows API."""
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)


def select_window_interactively() -> int:
    """Prompt the user to select Minecraft or active window."""
    windows = list_windows()
    if not windows:
        print("[!] No visible application windows detected.", flush=True)
        sys.exit(1)

    # Check if Minecraft is already open
    for hwnd, title in windows:
        if "minecraft" in title.lower():
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
            sub_choice = input(f"Choose match [1-{len(matches)}]: ").strip()
            if sub_choice.isdigit() and 1 <= int(sub_choice) <= len(matches):
                return matches[int(sub_choice) - 1][0]


class PvpAgent:
    """Master PvP Combat AI Agent."""

    def __init__(
        self,
        target_window: Optional[str | int] = None,
        resolution: tuple = (640, 480),
        aim_sensitivity: float = 0.22,
        attack_cooldown: float = 0.55,
        model_path: Optional[str] = None,
    ):
        self.width, self.height = resolution
        self.aim_sensitivity = aim_sensitivity
        self.attack_cooldown = attack_cooldown  # Minecraft 1.9+ sword attack recharge ~0.625s

        # Initialize hardware controller and vision detector
        self.input_ctrl = InputController()
        self.detector = VisionDetector(resolution)

        # Initialize PyTorch GPU CNN Brain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[+] Initializing MinecraftPvPCNN on device: {self.device}...", flush=True)
        self.model = MinecraftPvPCNN().to(self.device).eval()
        if model_path and os.path.exists(model_path):
            self.model.load_model(model_path, self.device)

        # Hook to target window
        if target_window is None:
            hwnd = select_window_interactively()
            self.cap = WindowCapture(hwnd)
        else:
            try:
                self.cap = WindowCapture(int(target_window))
            except ValueError:
                self.cap = WindowCapture(str(target_window))

        print(f"[+] Hooked to window: '{self.cap.window_title}' (HWND: {self.cap.hwnd})", flush=True)

        # Combat state variables
        self.is_active: bool = False
        self.last_attack_time: float = 0.0
        self.search_rotation_direction: int = 1
        self.ticks_without_target: int = 0

    def step(self, frame_bgr: np.ndarray) -> dict:
        """Run one 20 TPS decision step (Sense -> Think -> Act)."""
        # Ensure frame matches 640x480
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)

        # 1. Sense: Detect enemy player visual cues
        det = self.detector.detect(frame_bgr)

        # 2. Think: Determine combat actions
        actions = {"w": False, "s": False, "a": False, "d": False, "attack": False, "dx": 0.0, "dy": 0.0}

        if det["has_target"]:
            self.ticks_without_target = 0

            # Proportional Aiming: calculate relative mouse movement
            # Apply dampening / sensitivity multiplier
            raw_dx = det["dx"]
            raw_dy = det["dy"]

            # Smooth deadband: if crosshair is already on target, don't jitter
            deadband = 4.0
            aim_dx = raw_dx * self.aim_sensitivity if abs(raw_dx) > deadband else 0.0
            aim_dy = raw_dy * self.aim_sensitivity if abs(raw_dy) > deadband else 0.0

            # Clamp max single-tick mouse movement
            aim_dx = float(np.clip(aim_dx, -35.0, 35.0))
            aim_dy = float(np.clip(aim_dy, -25.0, 25.0))

            actions["dx"] = aim_dx
            actions["dy"] = aim_dy

            # Movement: Walk forward towards the enemy if not too close
            actions["w"] = True

            # Attack Punch Logic:
            # If target is in reach and centered in crosshair, and attack cooldown is ready:
            now = time.perf_counter()
            time_since_attack = now - self.last_attack_time
            if det["in_attack_range"] and time_since_attack >= self.attack_cooldown:
                actions["attack"] = True
                self.last_attack_time = now

        else:
            self.ticks_without_target += 1
            # Searching behavior: if no enemy visible for > 5 ticks, rotate slowly to scan arena
            if self.ticks_without_target > 5:
                actions["dx"] = 5.0 * self.search_rotation_direction
                actions["w"] = False

        # 3. Act: Dispatch hardware inputs if AI is ACTIVE
        if self.is_active:
            # Apply 3D camera aim
            if actions["dx"] != 0 or actions["dy"] != 0:
                self.input_ctrl.move_mouse(int(actions["dx"]), int(actions["dy"]))

            # Apply directional walking keys
            self.input_ctrl.set_movement(w=actions["w"], s=actions["s"], a=actions["a"], d=actions["d"])

            # Apply attack punch
            if actions["attack"]:
                self.input_ctrl.attack_click()
        else:
            # If paused, ensure no keys are held
            self.input_ctrl.release_all()

        return {"detection": det, "actions": actions, "active": self.is_active}

    def run(self, preview: bool = True):
        """Main real-time 20 TPS loop."""
        tick_interval = 1.0 / 20.0  # 50ms per tick
        print("\n" + "=" * 60, flush=True)
        print(" MINECRAFT PVP COMBAT AI RUNNING", flush=True)
        print("=" * 60, flush=True)
        print("  Controls:", flush=True)
        print("    [F6]   : TOGGLE AI ON / OFF (Emergency Killswitch)", flush=True)
        print("    [q]    : Quit Agent (in preview window)", flush=True)
        print("    [Ctrl+C]: Stop Agent in terminal", flush=True)
        print("=" * 60, flush=True)
        print("[!] AI starts in PAUSED mode. Press F6 to ENABLE combat control!\n", flush=True)

        f6_was_pressed = False
        next_tick = time.perf_counter()

        try:
            while True:
                # Global Hotkey Check (F6 toggle)
                f6_current = is_key_pressed(VK_F6)
                if f6_current and not f6_was_pressed:
                    self.is_active = not self.is_active
                    status = "ACTIVE (FIGHTING)" if self.is_active else "PAUSED"
                    print(f"\n[>>> AI STATUS CHANGED: {status} <<<]\n", flush=True)
                    if not self.is_active:
                        self.input_ctrl.release_all()
                f6_was_pressed = f6_current

                if not self.cap.is_valid():
                    print("[!] Target window closed. Stopping agent.", flush=True)
                    break

                # 1. Capture screen frame
                success, frame = self.cap.get_frame()
                if success and frame is not None:
                    # 2. Run Sense-Think-Act step
                    step_result = self.step(frame)

                    # 3. Render live HUD preview
                    if preview:
                        det = step_result["detection"]
                        actions = step_result["actions"]
                        hud_frame = self.detector.draw_hud(frame, det, actions)

                        # Overlay AI state
                        state_color = (0, 255, 0) if self.is_active else (0, 255, 255)
                        state_txt = "AI: RUNNING [F6 to Pause]" if self.is_active else "AI: PAUSED [Press F6 to Enable]"
                        cv2.putText(hud_frame, state_txt, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

                        cv2.imshow("Minecraft PvP AI - Live Vision HUD (Press 'q' to stop)", hud_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("\n[+] Stop signal received from preview window.", flush=True)
                            break

                # Precise 20 TPS timing
                next_tick += tick_interval
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0.001:
                    time.sleep(sleep_time * 0.95)
                while time.perf_counter() < next_tick:
                    pass

        except KeyboardInterrupt:
            print("\n[+] Stop signal (Ctrl+C) received.", flush=True)
        finally:
            self.input_ctrl.release_all()
            self.cap.close()
            cv2.destroyAllWindows()
            print("[+] PvP Agent cleanly shutdown. All keys released.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Autonomous Minecraft PvP Combat AI Agent.")
    parser.add_argument("-w", "--window", type=str, default=None, help="Target window title (default: auto-detect Minecraft)")
    parser.add_argument("--model", type=str, default=None, help="Path to trained PyTorch model weights (.pth)")
    parser.add_argument("--no-preview", action="store_true", help="Disable live visual HUD preview window")
    parser.add_argument("--sens", type=float, default=0.22, help="Aim sensitivity multiplier (default: 0.22)")
    parser.add_argument("--cooldown", type=float, default=0.55, help="Attack punch cooldown in seconds (default: 0.55s)")

    args = parser.parse_args()
    agent = PvpAgent(
        target_window=args.window,
        resolution=(640, 480),
        aim_sensitivity=args.sens,
        attack_cooldown=args.cooldown,
        model_path=args.model,
    )
    agent.run(preview=not args.no_preview)


if __name__ == "__main__":
    main()
