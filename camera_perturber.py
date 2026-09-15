"""Camera Perturber for Minecraft RL Sparring & Aim Robustness Training.

Applies periodic random horizontal camera shifts to challenge and train the RL agent.
- Starts if user presses [F6] (Training Mode with perturbations).
- Does NOT run if user starts by pressing [F7] (Clean match mode without perturbations).
- Press [ESC] for instant pause / killswitch.
"""

import argparse
import ctypes
import random
import sys
import time
import winsound
from input_controller import InputController


class CameraPerturber:
    """Dispatches strong horizontal camera shifts to train RL recovery when F6 is pressed."""

    def __init__(
        self,
        min_interval: float = 0.5,
        max_interval: float = 1.3,
        strength: float = 600.0,
    ):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.strength = float(strength)
        self.input_ctrl = InputController()
        self.is_active = False
        self.running = True

    def run(self):
        VK_F6 = 0x75
        VK_F7 = 0x76
        VK_ESCAPE = 0x1B
        VK_PAGEUP = 0x21
        VK_PAGEDOWN = 0x22
        VK_ADD = 0x6B
        VK_SUBTRACT = 0x6D

        f6_prev = False
        f7_prev = False
        esc_prev = False
        pup_prev = False
        pdown_prev = False
        add_prev = False
        sub_prev = False

        print("\n" + "=" * 65, flush=True)
        print(" MINECRAFT AI: CAMERA PERTURBER (SPARRING TRAINER)", flush=True)
        print("=" * 65, flush=True)
        print("  Role:       High-displacement horizontal camera shifts for RL training", flush=True)
        print(f"  Strength:   Base strength: {self.strength:.0f}px (Shifts: +/-{self.strength*0.6:.0f}px to +/-{self.strength*1.8:.0f}px)", flush=True)
        print(f"  Interval:   Randomized {self.min_interval:.1f}s - {self.max_interval:.1f}s between shifts", flush=True)
        print("=" * 65, flush=True)
        print("  Controls:", flush=True)
        print("    [F6]        : START / TOGGLE (Training Mode - Perturbations Active)", flush=True)
        print("    [F7]        : STOP / BYPASS  (Clean Mode - Perturber Will NOT Run)", flush=True)
        print("    [Page Up]   : Increase kick distance (+50px)", flush=True)
        print("    [Page Down] : Decrease kick distance (-50px)", flush=True)
        print("    [ESC]       : Emergency Pause All", flush=True)
        print("    [Ctrl+C]    : Exit Script", flush=True)
        print("=" * 65, flush=True)
        print("[!] Ready! Press F6 to train with perturbations, or F7 for clean aim.\n", flush=True)

        next_shift_time = time.perf_counter() + random.uniform(self.min_interval, self.max_interval)

        # Diverse high-displacement horizontal shift multipliers
        multipliers = [-1.8, -1.4, -1.0, -0.6, 0.6, 1.0, 1.4, 1.8]

        try:
            while self.running:
                now = time.perf_counter()

                # Poll Hotkeys
                f6_down = bool(ctypes.windll.user32.GetAsyncKeyState(VK_F6) & 0x8000)
                f7_down = bool(ctypes.windll.user32.GetAsyncKeyState(VK_F7) & 0x8000)
                esc_down = bool(ctypes.windll.user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)
                pup_down = bool(ctypes.windll.user32.GetAsyncKeyState(VK_PAGEUP) & 0x8000)
                pdown_down = bool(ctypes.windll.user32.GetAsyncKeyState(VK_PAGEDOWN) & 0x8000)
                add_down = bool(ctypes.windll.user32.GetAsyncKeyState(VK_ADD) & 0x8000)
                sub_down = bool(ctypes.windll.user32.GetAsyncKeyState(VK_SUBTRACT) & 0x8000)

                # ESC: Instant Pause
                if esc_down and not esc_prev:
                    if self.is_active:
                        self.is_active = False
                        try:
                            winsound.Beep(550, 160)
                        except Exception:
                            pass
                        print("[PERTURBER] PAUSED via ESC.", flush=True)

                # F6: Enable Training Mode (Run Perturber)
                elif f6_down and not f6_prev:
                    self.is_active = not self.is_active
                    try:
                        if self.is_active:
                            winsound.Beep(1200, 100)
                        else:
                            winsound.Beep(600, 120)
                    except Exception:
                        pass
                    state_str = "ACTIVE (Random Shifts Running)" if self.is_active else "PAUSED"
                    print(f"\n[PERTURBER STATE (F6): {state_str}]", flush=True)
                    next_shift_time = now + random.uniform(self.min_interval, self.max_interval)

                # F7: Disable / Bypass (Clean Mode - DO NOT RUN)
                elif f7_down and not f7_prev:
                    if self.is_active:
                        self.is_active = False
                        try:
                            winsound.Beep(600, 120)
                        except Exception:
                            pass
                    print("\n[PERTURBER STATE (F7): DISABLED - Clean Aim Mode (Zero Perturbations)]", flush=True)

                # Increase distance live via Page Up or Numpad +
                elif (pup_down and not pup_prev) or (add_down and not add_prev):
                    self.strength = min(2500.0, self.strength + 100.0)
                    try:
                        winsound.Beep(1400, 80)
                    except Exception:
                        pass
                    print(f"  [+] DISTANCE INCREASED -> Base: {self.strength:.0f}px (Shifts: +/-{self.strength*0.6:.0f}px to +/-{self.strength*1.8:.0f}px)", flush=True)

                # Decrease distance live via Page Down or Numpad -
                elif (pdown_down and not pdown_prev) or (sub_down and not sub_prev):
                    self.strength = max(100.0, self.strength - 100.0)
                    try:
                        winsound.Beep(800, 80)
                    except Exception:
                        pass
                    print(f"  [-] DISTANCE DECREASED -> Base: {self.strength:.0f}px (Shifts: +/-{self.strength*0.6:.0f}px to +/-{self.strength*1.8:.0f}px)", flush=True)

                f6_prev = f6_down
                f7_prev = f7_down
                esc_prev = esc_down
                pup_prev = pup_down
                pdown_prev = pdown_down
                add_prev = add_down
                sub_prev = sub_down

                # Dispatch Strong Horizontal Camera Shift when active
                if self.is_active and now >= next_shift_time:
                    mult = random.choice(multipliers)
                    dx = round(mult * self.strength, 1)

                    # Dynamic duration scaled to displacement to guarantee smooth capture
                    dur = max(0.045, min(0.080, abs(dx) / 12000.0))
                    self.input_ctrl.move_mouse(dx, 0.0, dynamic=True, duration_sec=dur)
                    print(f"  ⚡ [SHIFT] Displaced camera dx={dx:+.0f}px (kick distance: {abs(dx):.0f}px) -> RL recovering...", flush=True)

                    next_shift_time = now + random.uniform(self.min_interval, self.max_interval)

                time.sleep(0.01)  # 10ms loop

        except KeyboardInterrupt:
            print("\n[+] Perturber stopped.", flush=True)
        finally:
            self.input_ctrl.release_all()


def main():
    parser = argparse.ArgumentParser(description="Camera Perturber for RL Training")
    parser.add_argument("--strength", type=float, default=600.0, help="Shift strength in pixels (default: 600.0)")
    parser.add_argument("--preset", type=str, choices=["light", "medium", "strong", "extreme", "insane"], default=None, help="Preset strength (light=300, medium=450, strong=600, extreme=900, insane=1400)")
    parser.add_argument("--min-interval", type=float, default=0.5, help="Min seconds between shifts (default: 0.5)")
    parser.add_argument("--max-interval", type=float, default=1.3, help="Max seconds between shifts (default: 1.3)")
    args = parser.parse_args()

    # Apply preset if specified
    strength = args.strength
    if args.preset == "light":
        strength = 300.0
    elif args.preset == "medium":
        strength = 450.0
    elif args.preset == "strong":
        strength = 600.0
    elif args.preset == "extreme":
        strength = 900.0
    elif args.preset == "insane":
        strength = 1400.0

    perturber = CameraPerturber(
        min_interval=args.min_interval,
        max_interval=args.max_interval,
        strength=strength,
    )
    perturber.run()


if __name__ == "__main__":
    main()
