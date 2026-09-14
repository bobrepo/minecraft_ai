"""cap_game Recorder: High-speed Minecraft gameplay and input telemetry recorder.

Captures Minecraft window at exact 20 TPS (50ms interval) synchronized with
hardware keyboard (WASD, Space, Sprint) and mouse telemetry (dx, dy, left click).
Saves session pairs into train_videos/:
  - train_videos/session_YYYYMMDD_HHMMSS.mp4 (video frames)
  - train_videos/session_YYYYMMDD_HHMMSS_actions.json (per-frame action labels)
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
from pynput import mouse

# Add parent project directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from window_capture import WindowCapture, is_minecraft_window, list_windows

# Virtual Key Codes for Windows API
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


class TelemetryRecorder:
    """Synchronous 20 TPS Screen and Human Input Telemetry Recorder."""

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
                print("[!] Could not auto-detect Minecraft. Ensure Minecraft is running!", flush=True)
                sys.exit(1)

        self.cap = WindowCapture(target_hwnd)
        print(f"[+] Hooked to window: '{self.cap.window_title}' (HWND: {self.cap.hwnd})", flush=True)

        # Mouse relative delta accumulator
        self._lock = threading.Lock()
        self._accum_dx: float = 0.0
        self._accum_dy: float = 0.0
        self._mouse_listener = mouse.Listener(on_move=self._on_mouse_move)
        self._mouse_listener.daemon = True
        self._mouse_listener.start()

        self.is_recording = False
        self.stop_requested = False

        # Session storage
        self.telemetry_log: List[Dict] = []
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_path: str = ""
        self.json_path: str = ""
        self.frame_count: int = 0

    def _on_mouse_move(self, x, y):
        # We track relative delta across frames
        pass

    def _poll_mouse_delta(self) -> Tuple[int, int]:
        """Poll relative cursor movement using Windows API."""
        # Minecraft locks cursor inside window; GetCursorPos provides delta relative to window center
        try:
            pt = ctypes.wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            if hasattr(self, "_last_cursor_pos"):
                dx = pt.x - self._last_cursor_pos[0]
                dy = pt.y - self._last_cursor_pos[1]
            else:
                dx, dy = 0, 0
            self._last_cursor_pos = (pt.x, pt.y)
            return dx, dy
        except Exception:
            return 0, 0

    def start_session(self):
        """Create new timestamped MP4 and JSON session files."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(self.output_dir, f"session_{ts}.mp4")
        self.json_path = os.path.join(self.output_dir, f"session_{ts}_actions.json")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (self.width, self.height))
        self.telemetry_log = []
        self.frame_count = 0
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
            print(f"\n[+] Saved {self.frame_count} frames ({self.frame_count / self.fps:.1f}s) to:")
            print(f"    Video:     {os.path.abspath(self.video_path)}")
            print(f"    Telemetry: {os.path.abspath(self.json_path)}", flush=True)

        self.telemetry_log = []
        self.is_recording = False

    def sample_inputs(self) -> Dict:
        """Sample synchronized keyboard and mouse state for current tick."""
        dx, dy = self._poll_mouse_delta()

        w_down = is_vkey_pressed(VK_W)
        s_down = is_vkey_pressed(VK_S)
        a_down = is_vkey_pressed(VK_A)
        d_down = is_vkey_pressed(VK_D)
        ctrl_down = is_vkey_pressed(VK_CONTROL)
        space_down = is_vkey_pressed(VK_SPACE)
        shift_down = is_vkey_pressed(VK_SHIFT)
        lmb_down = is_vkey_pressed(VK_LBUTTON)

        sprint = ctrl_down or (w_down and not s_down and not shift_down)

        return {
            "frame": self.frame_count,
            "timestamp": round(time.time(), 3),
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
        print("\n" + "=" * 62, flush=True)
        print("  MINECRAFT GAME & TELEMETRY RECORDER (cap_game)", flush=True)
        print("=" * 62, flush=True)
        print("  Controls:", flush=True)
        print("    [F6]   : START / PAUSE RECORDING (Audio Beep Feedback)", flush=True)
        print("    [ESC]  : STOP & FINALIZE CURRENT SESSION", flush=True)
        print("    Ctrl+C : Exit Recorder", flush=True)
        print("=" * 62, flush=True)
        print("[!] Press F6 in Minecraft when you are ready to record!\n", flush=True)

        f6_prev = False
        next_tick = time.perf_counter()

        try:
            while not self.stop_requested:
                if not self.cap.is_valid():
                    print("[!] Minecraft window closed. Exiting recorder.", flush=True)
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
                        # Resize to standard VGA 4:3 dataset size
                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

                        actions = self.sample_inputs()
                        self.telemetry_log.append(actions)
                        self.video_writer.write(frame)
                        self.frame_count += 1

                        # Terminal live feedback
                        act_str = ""
                        if actions["w"]: act_str += "W "
                        if actions["a"]: act_str += "A "
                        if actions["s"]: act_str += "S "
                        if actions["d"]: act_str += "D "
                        if actions["sprint"]: act_str += "[SPRINT] "
                        if actions["jump"]: act_str += "[JUMP] "
                        if actions["attack"]: act_str += "[ATTACK ⚡] "

                        elapsed = self.frame_count / self.fps
                        sys.stdout.write(
                            f"\r[REC {elapsed:05.1f}s | {self.frame_count:4d} frames] {act_str:<25} | dx: {actions['dx']:+3d}, dy: {actions['dy']:+3d}  "
                        )
                        sys.stdout.flush()

                # Precise 20 TPS timer
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

    rec = TelemetryRecorder(
        output_dir=args.out_dir,
        target_resolution=(args.width, args.height),
        fps=args.fps,
    )
    rec.run()


if __name__ == "__main__":
    main()
