"""High-speed real-time window capture module for Windows.

Uses Windows Win32 API, DWM, and MSS for low-latency client-area screen capture.
"""

import ctypes
from typing import Dict, List, Optional, Tuple
import numpy as np

# Set process DPI awareness so window coordinates match physical screen pixels
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import mss
import win32con
import win32gui

user32 = ctypes.windll.user32
dwmapi = ctypes.windll.dwmapi

# Attach thread to interactive desktop if running in a separate service/runner desktop
try:
    h_default_desk = user32.OpenDesktopW("default", 0, False, 0x0040 | 0x0001 | 0x0080)
    if h_default_desk:
        user32.SetThreadDesktop(h_default_desk)
except Exception:
    h_default_desk = None


# Known internal/system windows to exclude
IGNORE_TITLES = {
    "Default IME",
    "MSCTFIME UI",
    "CiceroUIWndFrame",
    "Program Manager",
    "DesktopWindowXamlSource",
    "PopupHost",
    "Windows Input Experience",
    "NVIDIA GeForce Overlay",
    "NVIDIA GeForce Overlay DT",
    "Hidden Window",
    "SystemResourceNotifyWindow",
    "MediaContextNotificationWindow",
    "Settings",
}


def _is_cloaked(hwnd: int) -> bool:
    """Check if window is cloaked (e.g., hidden UWP app or inactive virtual desktop)."""
    try:
        cloaked = ctypes.c_int(0)
        dwmapi.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
        return cloaked.value != 0
    except Exception:
        return False


def list_windows(include_minimized: bool = True) -> List[Tuple[int, str]]:
    """Enumerate visible application windows.

    Args:
        include_minimized: If True, also includes minimized windows.

    Returns:
        List of tuples containing (hwnd, window_title).
    """
    windows = []

    def enum_handler(hwnd, _):
        try:
            if not win32gui.IsWindow(hwnd):
                return True

            title = win32gui.GetWindowText(hwnd).strip()
            if not title:
                return True

            if title in IGNORE_TITLES or title.startswith("GDI+ Window"):
                return True

            if _is_cloaked(hwnd):
                return True

            is_iconic = bool(win32gui.IsIconic(hwnd))
            is_visible = bool(win32gui.IsWindowVisible(hwnd))

            if not is_visible and not is_iconic:
                return True

            if not include_minimized and is_iconic:
                return True

            display_title = f"{title} [Minimized]" if is_iconic else title
            windows.append((hwnd, display_title))
        except Exception:
            pass
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    if h_default_desk:
        user32.EnumDesktopWindows(h_default_desk, WNDENUMPROC(enum_handler), 0)
    else:
        win32gui.EnumWindows(enum_handler, None)

    return windows


class WindowCapture:
    """Captures a specific window's client area at high frame rates using MSS."""

    def __init__(self, target: Optional[str | int] = None):
        """Initialize capture for a window by title (substring) or HWND.

        Args:
            target: Window title substring or integer HWND. If None, uses active foreground window.
        """
        self.sct = mss.mss()
        self.hwnd: Optional[int] = None
        self.window_title: str = ""

        if target is None:
            self.hwnd = win32gui.GetForegroundWindow()
            self.window_title = win32gui.GetWindowText(self.hwnd)
        elif isinstance(target, int):
            self.hwnd = target
            self.window_title = win32gui.GetWindowText(self.hwnd)
        else:
            self.hwnd = self._find_window_by_title(target)
            if self.hwnd is None:
                raise ValueError(f"Could not find window matching title: '{target}'")
            self.window_title = win32gui.GetWindowText(self.hwnd)

        # Restore window if minimized
        if self.hwnd and win32gui.IsIconic(self.hwnd):
            print(f"[+] Restoring minimized window: '{self.window_title}'...")
            win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)

    def _find_window_by_title(self, query: str) -> Optional[int]:
        query_lower = query.lower()
        active_windows = list_windows(include_minimized=True)

        # First try exact match
        for hwnd, title in active_windows:
            clean_title = title.replace(" [Minimized]", "").strip()
            if clean_title.lower() == query_lower:
                return hwnd

        # Then try substring match
        for hwnd, title in active_windows:
            clean_title = title.replace(" [Minimized]", "").strip()
            if query_lower in clean_title.lower():
                return hwnd

        return None

    def is_valid(self) -> bool:
        """Check if target window is still valid and open."""
        if not self.hwnd or not win32gui.IsWindow(self.hwnd):
            return False
        return True

    def get_client_rect(self) -> Optional[Dict[str, int]]:
        """Get the current screen coordinates of the window's client area."""
        if not self.is_valid():
            return None

        try:
            # If minimized, client rect will be 0
            if win32gui.IsIconic(self.hwnd):
                return None

            # Client area dimensions
            _, _, width, height = win32gui.GetClientRect(self.hwnd)
            if width <= 0 or height <= 0:
                # Fallback to GetWindowRect if client rect is unavailable
                rect = win32gui.GetWindowRect(self.hwnd)
                width = rect[2] - rect[0]
                height = rect[3] - rect[1]
                if width <= 0 or height <= 0:
                    return None
                return {
                    "top": int(rect[1]),
                    "left": int(rect[0]),
                    "width": int(width),
                    "height": int(height),
                }

            # Convert (0, 0) of client area to screen coordinates
            screen_x, screen_y = win32gui.ClientToScreen(self.hwnd, (0, 0))

            return {
                "top": int(screen_y),
                "left": int(screen_x),
                "width": int(width),
                "height": int(height),
            }
        except Exception:
            return None

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture one frame of the target window.

        Returns:
            Tuple of (success: bool, frame: np.ndarray in BGR format or None).
        """
        rect = self.get_client_rect()
        if rect is None:
            return False, None

        try:
            # Fast capture via MSS
            raw_shot = self.sct.grab(rect)
            frame = np.frombuffer(raw_shot.raw, dtype=np.uint8).reshape((raw_shot.height, raw_shot.width, 4))
            frame_bgr = np.ascontiguousarray(frame[:, :, :3])
            return True, frame_bgr
        except Exception:
            return False, None

    def close(self):
        """Release MSS resources."""
        if self.sct:
            self.sct.close()
