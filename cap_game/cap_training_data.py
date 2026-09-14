"""cap_training_data: High-precision Minecraft combat, keystroke, and mouse telemetry recorder.

Captures Minecraft at exact 20 TPS (50ms per frame) synchronized with:
  - Video stream (MP4 / 640x480)
  - Synchronized timestamped keystrokes (W, A, S, D, Space, Sprint, Shift, Attack)
  - Hardware relative mouse movement (dx, dy)
Saves matched session pairs into train_videos/:
  - train_videos/session_YYYYMMDD_HHMMSS.mp4
  - train_videos/session_YYYYMMDD_HHMMSS_actions.json
"""

import argparse
import ctypes
from datetime import datetime
import json
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple
import winsound

import cv2
import numpy as np
try:
    from pynput import mouse
    HAS_PYNPUT = True
except ImportError:
    mouse = None
    HAS_PYNPUT = False

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from window_capture import WindowCapture, is_minecraft_window, list_windows

# Dedicated POINT struct - zero reliance on ctypes.wintypes
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


# Windows API Virtual-Key Codes
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_SPACE = 0x20
VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44
VK_F6 = 0x75
VK_ESCAPE = 0x1B


def is_vkey_pressed(vk_code: int) -> bool:
    """Check if physical key is down globally via Windows API."""
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)


def auto_detect_minecraft() -> Optional[int]:
    """Find the real Minecraft window, strictly excluding browser/terminal tabs."""
    windows = list_windows()
    for hwnd, title in windows:
        if is_minecraft_window(hwnd, title):
            return hwnd
    return None


def select_window_interactively() -> int:
    """Prompt user to choose from visible windows if auto-detect misses."""
    windows = list_windows()
    if not windows:
        print("[!] No visible application windows detected.", flush=True)
        sys.exit(1)

    print("\n" + "=" * 60, flush=True)
    print(" ACTIVE WINDOWS AVAILABLE FOR CAPTURE", flush=True)
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
            return matches[0][0]
        elif len(matches) > 1:
            for i, (_, t) in enumerate(matches, 1):
                print(f"    ({i}) {t}", flush=True)
            try:
                sub = input(f"Choose match [1-{len(matches)}]: ").strip()
                if sub.isdigit() and 1 <= int(sub) <= len(matches):
                    return matches[int(sub) - 1][0]
            except EOFError:
                sys.exit(0)


class TrainingDataRecorder:
    """High-speed synchronous 20 TPS screen and timestamped input telemetry recorder."""

    def __init__(
        self,
        target_hwnd: Optional[int] = None,
        output_dir: str = "train_videos",
        target_resolution: Tuple[int, int] = (640, 480),
        fps: float = 20.0,
    ):
        self.output_dir = output_dir
        self.width, self.height = target_resolution
        self.fps = fps
        self.tick_interval = 1.0 / fps

        os.makedirs(self.output_dir, exist_ok=True)

        if target_hwnd is None:
            target_hwnd = auto_detect_minecraft()
            if target_hwnd is None:
                print("[!] Could not auto-detect Minecraft window automatically.", flush=True)
                print("[!] Please select your Minecraft window from the list below:", flush=True)
                target_hwnd = select_window_interactively()

        self.cap = WindowCapture(target_hwnd)
        print(f"[+] Hooked to window: '{self.cap.window_title}' (HWND: {self.cap.hwnd})", flush=True)

        # Dual-source hardware mouse movement tracking
        self._lock = threading.Lock()
        self._pynput_accum_dx: float = 0.0
        self._pynput_accum_dy: float = 0.0
        self._last_pynput_pos: Optional[Tuple[int, int]] = None
        self._last_cursor_pos: Optional[Tuple[int, int]] = None

        if HAS_PYNPUT and mouse is not None:
            self._mouse_listener = mouse.Listener(on_move=self._on_mouse_move)
            self._mouse_listener.daemon = True
            self._mouse_listener.start()
        else:
            self._mouse_listener = None

        self.is_recording = False
        self.stop_requested = False

        # Session storage
        self.telemetry_log: List[Dict] = []
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_path: str = ""
        self.json_path: str = ""
        self.frame_count: int = 0
        self.session_start_time: float = 0.0

    def _on_mouse_move(self, x, y):
        with self._lock:
            if self._last_pynput_pos is not None:
                self._pynput_accum_dx += (x - self._last_pynput_pos[0])
                self._pynput_accum_dy += (y - self._last_pynput_pos[1])
            self._last_pynput_pos = (x, y)

    def _poll_mouse_movement(self) -> Tuple[int, int]:
        """Dual-source relative mouse polling combining GetCursorPos and pynput listener."""
        # 1. Check pynput accumulated delta
        with self._lock:
            pyn_dx = int(round(self._pynput_accum_dx))
            pyn_dy = int(round(self._pynput_accum_dy))
            self._pynput_accum_dx = 0.0
            self._pynput_accum_dy = 0.0

        # 2. Check Windows GetCursorPos delta
        cur_dx, cur_dy = 0, 0
        try:
            pt = POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                if self._last_cursor_pos is not None:
                    cur_dx = pt.x - self._last_cursor_pos[0]
                    cur_dy = pt.y - self._last_cursor_pos[1]
                self._last_cursor_pos = (pt.x, pt.y)
        except Exception:
            pass

        # Use the most responsive non-zero delta
        final_dx = cur_dx if abs(cur_dx) > abs(pyn_dx) else pyn_dx
        final_dy = cur_dy if abs(cur_dy) > abs(pyn_dy) else pyn_dy

        # Filter out extreme cursor recentering jumps (>250px) that happen when Minecraft centers cursor
        if abs(final_dx) > 250:
            final_dx = 0
        if abs(final_dy) > 250:
            final_dy = 0

        return int(final_dx), int(final_dy)

    def start_session(self):
        """Create new timestamped MP4 and JSON session files."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(self.output_dir, f"session_{ts}.mp4")
        self.json_path = os.path.join(self.output_dir, f"session_{ts}_actions.json")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (self.width, self.height))
        self.telemetry_log = []
        self.frame_count = 0
        self.session_start_time = time.time()
        self.is_recording = True
        winsound.Beep(880, 150)
        print(f"\n[● RECORDING STARTED] Saving to: {self.video_path}", flush=True)

    def pause_session(self):
        """Pause active recording."""
        self.is_recording = False
        winsound.Beep(440, 150)
        print("\n[○ RECORDING PAUSED] Press F6 to resume.", flush=True)

    def finalize_session(self):
        """Save video stream and flush telemetry JSON to disk."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        if self.telemetry_log:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self.telemetry_log, f, indent=2)
            print(f"\n[+] Successfully saved {self.frame_count} frames ({self.frame_count / self.fps:.1f}s) to:")
            print(f"    Video:     {os.path.abspath(self.video_path)}")
            print(f"    Telemetry: {os.path.abspath(self.json_path)}", flush=True)

        self.telemetry_log = []
        self.is_recording = False

    def sample_inputs(self) -> Dict:
        """Sample synchronized timestamped keyboard and mouse state."""
        dx, dy = self._poll_mouse_movement()

        w_down = is_vkey_pressed(VK_W)
        s_down = is_vkey_pressed(VK_S)
        a_down = is_vkey_pressed(VK_A)
        d_down = is_vkey_pressed(VK_D)
        ctrl_down = is_vkey_pressed(VK_CONTROL)
        space_down = is_vkey_pressed(VK_SPACE)
        shift_down = is_vkey_pressed(VK_SHIFT)
        lmb_down = is_vkey_pressed(VK_LBUTTON)

        now = time.time()
        elapsed_ms = round((now - self.session_start_time) * 1000.0, 1) if self.session_start_time > 0 else 0.0
        sprint = ctrl_down or (w_down and not s_down and not shift_down)

        return {
            "frame": self.frame_count,
            "timestamp": round(now, 3),
            "elapsed_ms": elapsed_ms,
            "w": w_down,
            "s": s_down,
            "a": a_down,
            "d": d_down,
            "sprint": sprint,
            "jump": space_down,
            "shift": shift_down,
            "attack": lmb_down,
            "dx": int(dx),
            "dy": int(dy),
        }

    def run(self):
        """Main 20 TPS capture loop."""
        print("\n" + "=" * 64, flush=True)
        print("  MINECRAFT GAMEPLAY & TELEMETRY RECORDER (run_cap_training_data)", flush=True)
        print("=" * 64, flush=True)
        print("  Controls:", flush=True)
        print("    [F6]   : START / PAUSE RECORDING (Audio Beep Confirmation)", flush=True)
        print("    [ESC]  : STOP & SAVE CURRENT SESSION TO train_videos\\", flush=True)
        print("    Ctrl+C : Exit Recorder cleanly", flush=True)
        print("=" * 64, flush=True)
        print("[!] Press F6 in Minecraft when you are ready to record!\n", flush=True)

        f6_prev = False
        next_tick = time.perf_counter()

        try:
            while not self.stop_requested:
                if not self.cap.is_valid():
                    print("[!] Target window closed. Exiting recorder.", flush=True)
                    break

                # F6 toggle check
                f6_down = is_vkey_pressed(VK_F6)
                if f6_down and not f6_prev:
                    if not self.is_recording:
                        if self.video_writer is None:
                            self.start_session()
                        else:
                            self.is_recording = True
                            winsound.Beep(880, 150)
                            print("\n[● RECORDING RESUMED]", flush=True)
                    else:
                        self.pause_session()
                f6_prev = f6_down

                # ESC stop check
                if is_vkey_pressed(VK_ESCAPE):
                    if self.video_writer is not None:
                        print("\n[!] ESC pressed: Finalizing session...", flush=True)
                        self.finalize_session()

                # Tick frame capture
                if self.is_recording:
                    success, frame = self.cap.get_frame()
                    if success and frame is not None:
                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

                        actions = self.sample_inputs()
                        self.telemetry_log.append(actions)
                        self.video_writer.write(frame)
                        self.frame_count += 1

                        # Terminal live visualizer
                        act_str = ""
                        if actions["w"]: act_str += "W "
                        if actions["a"]: act_str += "A "
                        if actions["s"]: act_str += "S "
                        if actions["d"]: act_str += "D "
                        if actions["sprint"]: act_str += "[SPRINT] "
                        if actions["jump"]: act_str += "[JUMP] "
                        if actions["attack"]: act_str += "[ATK ⚡] "

                        elapsed = self.frame_count / self.fps
                        sys.stdout.write(
                            f"\r[REC {elapsed:05.1f}s | {self.frame_count:4d} frames] {act_str:<24} | Aim: dx={actions['dx']:+3d}, dy={actions['dy']:+3d}  "
                        )
                        sys.stdout.flush()

                # Precise 20 TPS pacing
                next_tick += self.tick_interval
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0.001:
                    time.sleep(sleep_time * 0.95)
                while time.perf_counter() < next_tick:
                    pass

        except KeyboardInterrupt:
            print("\n[+] Ctrl+C detected.", flush=True)
        finally:
            if self.video_writer is not None:
                self.finalize_session()
            self.cap.close()
            print("[+] Capture session ended cleanly.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Minecraft Gameplay & Input Telemetry Recorder")
    parser.add_argument("--out-dir", type=str, default="train_videos", help="Directory to save recordings")
    parser.add_argument("--fps", type=float, default=20.0, help="Recording tick rate (default 20 TPS)")
    parser.add_argument("--width", type=int, default=640, help="Output video width")
    parser.add_argument("--height", type=int, default=480, help="Output video height")
    args = parser.parse_args()

    rec = TrainingDataRecorder(
        output_dir=args.out_dir,
        target_resolution=(args.width, args.height),
        fps=args.fps,
    )
    rec.run()


if __name__ == "__main__":
    main()
