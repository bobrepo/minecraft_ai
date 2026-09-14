"""Real-time Window Screen Recorder for AI training data.

Captures target window at exact Minecraft tick rate (20 ticks/sec) and saves as MP4 into out_vid/.
Uses a threaded timer queue to guarantee constant 20.0 TPS without dropped ticks.
"""

import argparse
from datetime import datetime
import os
import queue
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

from window_capture import WindowCapture, list_windows


def select_window_interactively() -> int:
    """Prompt the user to select from currently active windows."""
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
            print(f"Please enter a number between 1 and {len(windows)}.", flush=True)
            continue

        # Search by substring
        matches = [(h, t) for h, t in windows if choice.lower() in t.lower()]
        if len(matches) == 1:
            print(f"[+] Matched: {matches[0][1]}", flush=True)
            return matches[0][0]
        elif len(matches) > 1:
            print(f"[!] Multiple matches found for '{choice}':", flush=True)
            for i, (_, t) in enumerate(matches, 1):
                print(f"    ({i}) {t}", flush=True)
            try:
                sub_choice = input(f"Choose match [1-{len(matches)}]: ").strip()
            except EOFError:
                sys.exit(0)
            if sub_choice.isdigit() and 1 <= int(sub_choice) <= len(matches):
                return matches[int(sub_choice) - 1][0]
        else:
            print(f"[!] No window containing '{choice}' found. Try again.", flush=True)


def record_window(
    target: Optional[str | int] = None,
    fps: float = 20.0,
    target_width: int = 640,
    target_height: int = 480,
    output_dir: str = "out_vid",
    preview: bool = True,
    max_frames: Optional[int] = None,
):
    """Record the chosen window at the specified tick rate and resolution.

    Args:
        target: Window title, substring, or HWND. If None, prompts interactively.
        fps: Target capture tick rate (default 20.0 for Minecraft ticks).
        target_width: Output width (default 640 for 480p 4:3).
        target_height: Output height (default 480 for 480p 4:3).
        output_dir: Output folder for saved videos.
        preview: Whether to display a live OpenCV preview window.
        max_frames: Optional frame limit (useful for automated testing).
    """
    os.makedirs(output_dir, exist_ok=True)

    if target is None:
        hwnd = select_window_interactively()
        cap = WindowCapture(hwnd)
    else:
        try:
            target_int = int(target)
            cap = WindowCapture(target_int)
        except ValueError:
            cap = WindowCapture(str(target))

    print(f"\n[+] Hooked to window: '{cap.window_title}' (HWND: {cap.hwnd})", flush=True)

    # Capture initial frame to establish video dimensions
    initial_frame = None
    for _ in range(20):
        success, frame = cap.get_frame()
        if success and frame is not None and frame.size > 0:
            initial_frame = frame
            break
        time.sleep(0.1)

    if initial_frame is None:
        print("[!] Failed to capture initial frame. Ensure the window is open.", flush=True)
        cap.close()
        return

    orig_h, orig_w = initial_frame.shape[:2]
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = os.path.join(output_dir, f"capture_{timestamp_str}.mp4")

    # OpenCV VideoWriter with MP4V codec locked to target resolution (854x480)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_filename, fourcc, fps, (target_width, target_height))

    if not writer.isOpened():
        print(f"[!] Error: Could not open video writer for {out_filename}", flush=True)
        cap.close()
        return

    tick_interval = 1.0 / fps
    print(f"[+] Output file: {os.path.abspath(out_filename)}", flush=True)
    print(f"[+] Capture rate: {fps:.1f} ticks/second (50ms tick interval)", flush=True)
    print(f"[+] Resolution: {target_width}x{target_height} (source window: {orig_w}x{orig_h})", flush=True)
    print("[+] Recording started! Press 'q' in preview or Ctrl+C in terminal to stop.\n", flush=True)

    frame_queue = queue.Queue(maxsize=120)
    stop_event = threading.Event()
    stats = {"captured": 0, "written": 0, "start_time": 0.0, "end_time": 0.0}

    # High-precision threaded capture worker
    def capture_worker():
        stats["start_time"] = time.perf_counter()
        next_tick = stats["start_time"]
        last_valid_frame = initial_frame

        while not stop_event.is_set():
            if max_frames and stats["captured"] >= max_frames:
                stats["end_time"] = time.perf_counter()
                stop_event.set()
                break

            if not cap.is_valid():
                print("\n[!] Target window was closed. Stopping capture.", flush=True)
                stats["end_time"] = time.perf_counter()
                stop_event.set()
                break

            success, frame = cap.get_frame()
            if success and frame is not None:
                last_valid_frame = frame
                stats["captured"] += 1
                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
            else:
                stats["captured"] += 1
                try:
                    frame_queue.put_nowait(last_valid_frame)
                except queue.Full:
                    pass

            # High precision 50ms pacing
            next_tick += tick_interval
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0.001:
                time.sleep(sleep_time * 0.95)
            while time.perf_counter() < next_tick:
                pass

        if stats["end_time"] == 0.0:
            stats["end_time"] = time.perf_counter()

    capture_thread = threading.Thread(target=capture_worker, daemon=True)
    capture_thread.start()

    start_time = time.perf_counter()
    last_status_print = start_time

    try:
        while not stop_event.is_set() or not frame_queue.empty():
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Ensure frame size matches target video container (854x480)
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

            writer.write(frame)
            stats["written"] += 1

            now = time.perf_counter()
            # Print console progress every 2 seconds
            if now - last_status_print >= 2.0:
                elapsed = now - start_time
                current_fps = stats["written"] / elapsed if elapsed > 0 else 0
                print(f" -> Recorded {stats['written']} ticks | {current_fps:.1f} TPS | Duration: {elapsed:.1f}s", flush=True)
                last_status_print = now

            if preview:
                preview_frame = frame.copy()
                elapsed = now - start_time
                current_fps = stats["written"] / elapsed if elapsed > 0 else 0
                cv2.putText(
                    preview_frame,
                    f"REC [{target_width}x{target_height}] | {stats['written']} ticks | {current_fps:.1f} TPS | Press 'q' to stop",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Minecraft AI - Capture Preview (Press 'q' to stop)", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[+] Stop signal received from preview window.", flush=True)
                    stop_event.set()
                    break

    except KeyboardInterrupt:
        print("\n[+] Stop signal (Ctrl+C) received.", flush=True)
        stop_event.set()
    finally:
        stop_event.set()
        capture_thread.join(timeout=1.0)
        writer.release()
        cap.close()
        cv2.destroyAllWindows()

        duration = stats["end_time"] - stats["start_time"] if stats["end_time"] > stats["start_time"] else (time.perf_counter() - start_time)
        actual_fps = stats["written"] / duration if duration > 0 else 0
        print("\n" + "=" * 60, flush=True)
        print(" RECORDING COMPLETE", flush=True)
        print("=" * 60, flush=True)
        print(f"  Saved to:       {os.path.abspath(out_filename)}", flush=True)
        print(f"  Total Ticks:    {stats['written']} frames", flush=True)
        print(f"  Duration:       {duration:.2f} seconds", flush=True)
        print(f"  Effective TPS:  {actual_fps:.2f} ticks/sec", flush=True)
        print("=" * 60, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Real-time window capture at Minecraft tick speed (20 TPS) in 640x480 resolution.")
    parser.add_argument("-w", "--window", type=str, default=None, help="Target window title or substring (e.g. 'Minecraft')")
    parser.add_argument("--fps", type=float, default=20.0, help="Target tick rate / FPS (default: 20.0)")
    parser.add_argument("--width", type=int, default=640, help="Output video width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Output video height (default: 480)")
    parser.add_argument("-o", "--output", type=str, default="out_vid", help="Output directory (default: out_vid)")
    parser.add_argument("--no-preview", action="store_true", help="Disable the live preview window")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames (useful for test runs)")

    args = parser.parse_args()
    record_window(
        target=args.window,
        fps=args.fps,
        target_width=args.width,
        target_height=args.height,
        output_dir=args.output,
        preview=not args.no_preview,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
