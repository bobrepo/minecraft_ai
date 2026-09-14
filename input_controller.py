"""DirectInput hardware simulation for Minecraft 3D camera and controls.

Uses Windows SendInput with direct hardware scan codes for 100% DirectX compatibility.
"""

import ctypes
import threading
import time
from typing import Callable, Dict, Optional, Set
import winsound

# Windows API constants
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]


# Flag constants
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

# DirectInput scan codes for Minecraft
SCAN_CODES: Dict[str, int] = {
    "w": 0x11,
    "s": 0x1F,
    "a": 0x1E,
    "d": 0x20,
    "space": 0x39,
    "shift": 0x2A,
    "ctrl": 0x1D,
    "e": 0x12,
    "q": 0x10,
    "1": 0x02,
    "2": 0x03,
    "3": 0x04,
    "4": 0x05,
}


class SubTickMouseThread(threading.Thread):
    """High-frequency (120 Hz) background thread for biological Minimum-Jerk camera movement.

    Interpolates aim deltas smoothly across monitor refresh frames (~8.3ms) rather than jumping
    abruptly once every 50ms combat tick. Uses Flash & Hogan's Minimum-Jerk polynomial and
    a Bresenham-style fractional accumulator to conserve 100% of integer mouse pixels.
    """

    def __init__(self, send_raw_fn: Callable[[int, int], None]):
        super().__init__(daemon=True)
        self.send_raw = send_raw_fn
        self.lock = threading.Lock()
        self.active = False
        self.total_dx = 0.0
        self.total_dy = 0.0
        self.duration = 0.045
        self.start_time = 0.0
        self.prev_p = 0.0
        self.accum_x = 0.0
        self.accum_y = 0.0
        self.running = True

    @staticmethod
    def _min_jerk_poly(s: float) -> float:
        """Flash & Hogan Minimum-Jerk position polynomial: 10s^3 - 15s^4 + 6s^5."""
        return 10.0 * (s**3) - 15.0 * (s**4) + 6.0 * (s**5)

    def submit_aim(self, dx: float, dy: float, duration_sec: float = 0.045):
        """Submit new aim delta to be dispatched over duration_sec via minimum jerk."""
        with self.lock:
            # If an existing trajectory was active, blend any residual fractional pixels
            rem_dx = (self.total_dx * (1.0 - self.prev_p)) if self.active else 0.0
            rem_dy = (self.total_dy * (1.0 - self.prev_p)) if self.active else 0.0

            self.total_dx = float(dx) + rem_dx
            self.total_dy = float(dy) + rem_dy
            self.duration = max(0.015, float(duration_sec))
            self.start_time = time.perf_counter()
            self.prev_p = 0.0
            self.active = (self.total_dx != 0.0 or self.total_dy != 0.0)

    def run(self):
        while self.running:
            with self.lock:
                if self.active:
                    now = time.perf_counter()
                    elapsed = now - self.start_time
                    s = min(1.0, elapsed / self.duration)
                    p = self._min_jerk_poly(s)
                    dp = p - self.prev_p
                    self.prev_p = p

                    self.accum_x += self.total_dx * dp
                    self.accum_y += self.total_dy * dp

                    sx = int(round(self.accum_x))
                    sy = int(round(self.accum_y))
                    self.accum_x -= sx
                    self.accum_y -= sy

                    if sx != 0 or sy != 0:
                        self.send_raw(sx, sy)

                    if s >= 1.0:
                        self.active = False
            time.sleep(0.007)  # ~140 Hz sub-tick update interval

    def stop(self):
        self.running = False


class InputController:
    """Controls Minecraft keyboard and 3D camera mouse inputs safely."""

    def __init__(self):
        self._pressed_keys: Set[str] = set()
        self._mouse_down: bool = False
        # Minimum-Jerk Sub-Tick Thread running at ~140 Hz
        self.mouse_thread = SubTickMouseThread(self._send_mouse_raw)
        self.mouse_thread.start()

    def press_key(self, key: str):
        """Send keydown event with hardware scan code."""
        key_lower = key.lower()
        if key_lower not in SCAN_CODES:
            return

        code = SCAN_CODES[key_lower]
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYDOWN, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(INPUT_KEYBOARD), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        self._pressed_keys.add(key_lower)

    def release_key(self, key: str):
        """Send keyup event with hardware scan code."""
        key_lower = key.lower()
        if key_lower not in SCAN_CODES:
            return

        code = SCAN_CODES[key_lower]
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(INPUT_KEYBOARD), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        self._pressed_keys.discard(key_lower)

    def set_movement(self, w: bool = False, s: bool = False, a: bool = False, d: bool = False, sprint: bool = False, jump: bool = False):
        """Set directional movement keys, sprinting, and jumping in one call."""
        # W / S forward/backward
        if w and not s:
            self.press_key("w")
            self.release_key("s")
        elif s and not w:
            self.press_key("s")
            self.release_key("w")
        else:
            self.release_key("w")
            self.release_key("s")

        # A / D strafing
        if a and not d:
            self.press_key("a")
            self.release_key("d")
        elif d and not a:
            self.press_key("d")
            self.release_key("a")
        else:
            self.release_key("a")
            self.release_key("d")

        # Sprinting (holding Ctrl)
        if sprint and w:
            self.press_key("ctrl")
        else:
            self.release_key("ctrl")

        # Jumping (Space)
        if jump:
            self.press_key("space")
        else:
            self.release_key("space")

    def _send_mouse_raw(self, dx: int, dy: int):
        """Low-level Windows SendInput dispatch."""
        if dx == 0 and dy == 0:
            return
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(INPUT_MOUSE), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def move_mouse(self, dx: float | int, dy: float | int, dynamic: bool = True):
        """Rotate first-person 3D camera with biological Minimum-Jerk sub-tick smoothing.

        Dispatches mouse deltas smoothly across monitor frames via a 140 Hz background
        thread to eliminate camera tearing, skipping, and jitter in Minecraft.
        """
        dx_val = float(dx)
        dy_val = float(dy)
        if dx_val == 0.0 and dy_val == 0.0:
            return

        # Clamp vertical deflection per tick to prevent over-pitching into zenith or nadir
        dy_val = max(-65.0, min(65.0, dy_val))

        if dynamic and hasattr(self, "mouse_thread") and self.mouse_thread.is_alive():
            # Submit to Minimum-Jerk Sub-Tick Thread (duration 45ms matches 50ms combat tick)
            self.mouse_thread.submit_aim(dx_val, dy_val, duration_sec=0.045)
        else:
            self._send_mouse_raw(int(round(dx_val)), int(round(dy_val)))

    def left_down(self):
        """Press left mouse button (punch/attack start)."""
        if self._mouse_down:
            return
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(INPUT_MOUSE), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        self._mouse_down = True

    def left_up(self):
        """Release left mouse button."""
        if not self._mouse_down:
            return
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(INPUT_MOUSE), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        self._mouse_down = False

    def attack_click(self):
        """Fast instantaneous attack punch without sleeping/blocking."""
        self.left_down()
        self.left_up()

    def release_all(self, force: bool = False):
        """Safety failsafe: release keys and mouse buttons pressed by the AI.

        Args:
            force: If True, flushes all known scan codes even if untracked (e.g. on emergency stop).
                   If False, only releases keys and mouse buttons that the AI actually pressed.
        """
        extra = ctypes.c_ulong(0)

        # 1. Keyboard release: only send KEYUP if keys were pressed or force is requested
        keys_to_flush = set()
        if force:
            keys_to_flush = set(SCAN_CODES.values())
        elif self._pressed_keys:
            keys_to_flush = {SCAN_CODES[k] for k in self._pressed_keys if k in SCAN_CODES}

        for code in keys_to_flush:
            ii_ = Input_I()
            ii_.ki = KeyBdInput(0, code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(INPUT_KEYBOARD), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        self._pressed_keys.clear()

        # 2. Mouse release: CRITICAL! NEVER send MOUSEEVENTF_LEFTUP unless the AI actually pressed it!
        # Sending synthetic mouse-up when mouse wasn't down causes Windows to drop/trigger clicks!
        if self._mouse_down:
            ii_m = Input_I()
            ii_m.mi = MouseInput(0, 0, 0, MOUSEEVENTF_LEFTUP | MOUSEEVENTF_RIGHTUP, 0, ctypes.pointer(extra))
            x_m = Input(ctypes.c_ulong(INPUT_MOUSE), ii_m)
            ctypes.windll.user32.SendInput(1, ctypes.pointer(x_m), ctypes.sizeof(x_m))
            self._mouse_down = False
        elif force:
            self._mouse_down = False

        if hasattr(self, "mouse_thread"):
            with self.mouse_thread.lock:
                self.mouse_thread.active = False


class EmergencyKillswitchListener:
    """High-frequency background daemon thread monitoring emergency stop hotkeys.

    Guarantees that the AI can ALWAYS be stopped instantly with 100% reliability,
    even when Minecraft has raw 3D mouse capture or the main thread is processing.

    Hotkeys:
    - [F6]  : Toggle AI Active / Paused state.
    - [ESC] : Instant Emergency Stop (pauses immediately and releases all inputs).
    """

    def __init__(self, input_ctrl: InputController, on_state_change: Optional[Callable[[bool], None]] = None):
        self.input_ctrl = input_ctrl
        self.on_state_change = on_state_change
        self.is_active: bool = False
        self._running: bool = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def set_active(self, active: bool):
        """Programmatically update active state."""
        prev = self.is_active
        self.is_active = active
        if prev and not active:
            self.input_ctrl.release_all(force=True)

    def _monitor_loop(self):
        VK_F6 = 0x75
        VK_ESCAPE = 0x1B
        f6_prev = False
        esc_prev = False

        while self._running:
            try:
                f6_down = bool(ctypes.windll.user32.GetAsyncKeyState(VK_F6) & 0x8000)
                esc_down = bool(ctypes.windll.user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)

                # ESC: Instant Emergency Kill / Pause
                if esc_down and not esc_prev:
                    if self.is_active:
                        self.is_active = False
                        self.input_ctrl.release_all(force=True)
                        try:
                            winsound.Beep(550, 160)
                        except Exception:
                            pass
                        print("\n[EMERGENCY STOP TRIGGERED (ESC): AI PAUSED & ALL INPUTS RELEASED]", flush=True)
                        if self.on_state_change:
                            self.on_state_change(False)

                # F6: Toggle Active / Paused
                elif f6_down and not f6_prev:
                    self.is_active = not self.is_active
                    if not self.is_active:
                        self.input_ctrl.release_all(force=True)
                    try:
                        if self.is_active:
                            winsound.Beep(1200, 100)
                        else:
                            winsound.Beep(600, 140)
                    except Exception:
                        pass
                    status = ">>> ACTIVE (FIGHTING) <<<" if self.is_active else "PAUSED"
                    print(f"\n[AI STATE TOGGLE (F6): {status}]", flush=True)
                    if self.on_state_change:
                        self.on_state_change(self.is_active)

                f6_prev = f6_down
                esc_prev = esc_down
            except Exception:
                pass
            time.sleep(0.005)  # 5ms continuous polling

    def stop(self):
        """Stop listener and ensure all inputs are released."""
        self._running = False
        self.is_active = False
        self.input_ctrl.release_all()

