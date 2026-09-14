"""Real-time Window Screen Recorder for AI training data.

Captures target window at Minecraft tick rate (20 ticks/sec) and saves as MP4 into out_vid/.
"""

import argparse
from datetime import datetime
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

from window_capture import WindowCapture, list_windows


def select_window_interactively() -> int:
    """Prompt the user to select from currently active windows."""
    windows = list_windows()
    if not windows:
        print("[!] No visible application windows detected.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" ACTIVE WINDOWS AVAILABLE FOR CAPTURE")
    print("=" * 60)
    for idx, (hwnd, title) in enumerate(windows, 1):
        print(f"  [{idx:2d}] {title}")
    print("=" * 60)

    while True:
        choice = input(f"\nSelect window [1-{len(windows)}] or search title (e.g. 'Minecraft'): ").strip()
        if not choice:
            continue

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(windows):
                return windows[idx - 1][0]
            print(f"Please enter a number between 1 and {len(windows)}.")
            continue

        # Search by substring
        matches = [(h, t) for h, t in windows if choice.lower() in t.lower()]
        if len(matches) == 1:
            print(f"[+] Matched: {matches[0][1]}")
            return matches[0][0]
        elif len(matches) > 1:
            print(f"[!] Multiple matches found for '{choice}':")
            for i, (_, t) in enumerate(matches, 1):
                print(f"    ({i}) {t}")
            sub_choice = input(f"Choose match [1-{len(matches)}]: ").strip()
            if sub_choice.isdigit() and 1 <= int(sub_choice) <= len(matches):
                return matches[int(sub_choice) - 1][0]
        else:
            print(f"[!] No window containing '{choice}' found. Try again.")


def record_window(
    target: Optional[str | int] = None,
    fps: float = 20.0,
    output_dir: str = "out_vid",
    preview: bool = True,
    max_frames: Optional[int] = None,
):
    """Record the chosen window at the specified tick rate.

    Args:
        target: Window title, substring, or HWND. If None, prompts interactively.
        fps: Target capture tick rate (default 20.0 for Minecraft ticks).
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

    print(f"\n[+] Hooked to window: '{cap.window_title}' (HWND: {cap.hwnd})")

    # Capture initial frame to establish dimensions
    success, initial_frame = cap.get_frame()
    attempts = 0
    while not success and attempts < 10:
        time.sleep(0.1)
        success, initial_frame = cap.get_frame()
        attempts += 1

    if not success or initial_frame is None:
        print("[!] Failed to capture initial frame. Ensure the window is visible and not minimized.")
        cap.close()
        return

    height, width = initial_frame.shape[:2]
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = os.path.join(output_dir, f"capture_{timestamp_str}.mp4")

    # OpenCV VideoWriter with MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"[!] Error: Could not open video writer for {out_filename}")
        cap.close()
        return

    tick_interval = 1.0 / fps
    print(f"[+] Output file: {out_filename}")
    print(f"[+] Capture rate: {fps:.1f} ticks/second (50ms tick interval)")
    print(f"[+] Resolution: {width}x{height}")
    print("[+] Recording started! Press 'q' in the preview window or Ctrl+C in terminal to stop.\n")

    frame_count = 0
    start_time = time.perf_counter()
    next_tick = start_time + tick_interval

    try:
        while True:
            if max_frames and frame_count >= max_frames:
                break

            if not cap.is_valid():
                print("\n[!] Target window was minimized or closed. Stopping capture.")
                break

            success, frame = cap.get_frame()
            if success and frame is not None:
                # If window resized slightly, conform to initial video dimensions
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                writer.write(frame)
                frame_count += 1

                if preview:
                    # Create downscaled preview if window is large
                    preview_frame = frame
                    if width > 960:
                        scale = 960 / width
                        preview_frame = cv2.resize(frame, (960, int(height * scale)))

                    # Overlay status on preview
                    elapsed = time.perf_counter() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    cv2.putText(
                        preview_frame,
                        f"REC | {frame_count} ticks | {current_fps:.1f} FPS | Press 'q' to stop",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow("Capture Preview (Press 'q' to stop)", preview_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n[+] Stop signal received from preview window.")
                        break

            # High-precision tick rate governor
            now = time.perf_counter()
            sleep_duration = next_tick - now
            if sleep_duration > 0.001:
                time.sleep(sleep_duration * 0.9)
            while time.perf_counter() < next_tick:
                pass

            next_tick += tick_interval

    except KeyboardInterrupt:
        print("\n[+] Stop signal (Ctrl+C) received.")
    finally:
        total_time = time.perf_counter() - start_time
        writer.release()
        cap.close()
        cv2.destroyAllWindows()

        actual_fps = frame_count / total_time if total_time > 0 else 0
        print("\n" + "=" * 60)
        print(" RECORDING COMPLETE")
        print("=" * 60)
        print(f"  Saved to:       {os.path.abspath(out_filename)}")
        print(f"  Total Ticks:    {frame_count} frames")
        print(f"  Duration:       {total_time:.2f} seconds")
        print(f"  Effective FPS:  {actual_fps:.2f} ticks/sec")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Real-time window capture at Minecraft tick speed (20 TPS).")
    parser.add_argument("-w", "--window", type=str, default=None, help="Target window title or substring (e.g. 'Minecraft')")
    parser.add_argument("--fps", type=float, default=20.0, help="Target tick rate / FPS (default: 20.0)")
    parser.add_argument("-o", "--output", type=str, default="out_vid", help="Output directory (default: out_vid)")
    parser.add_argument("--no-preview", action="store_true", help="Disable the live preview window")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames (useful for test runs)")

    args = parser.parse_args()
    record_window(
        target=args.window,
        fps=args.fps,
        output_dir=args.output,
        preview=not args.no_preview,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
