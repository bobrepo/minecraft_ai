"""DirectInput hardware simulation for Minecraft 3D camera and controls.

Uses Windows SendInput with direct hardware scan codes for 100% DirectX compatibility.
"""

import ctypes
import time
from typing import Dict, Set

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


class InputController:
    """Controls Minecraft keyboard and 3D camera mouse inputs safely."""

    def __init__(self):
        self._pressed_keys: Set[str] = set()
        self._mouse_down: bool = False

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

    def move_mouse(self, dx: int, dy: int):
        """Rotate first-person 3D camera by relative pixel deltas."""
        if dx == 0 and dy == 0:
            return
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(INPUT_MOUSE), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

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

    def release_all(self):
        """Safety failsafe: release all currently pressed keys and mouse buttons."""
        for key in list(self._pressed_keys):
            self.release_key(key)
        self._pressed_keys.clear()
        if self._mouse_down:
            self.left_up()
