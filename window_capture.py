"""High-speed real-time window capture module for Windows.

Supports direct Window DC capture (PrintWindow) and MSS with coordinate clamping.
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
import win32ui

user32 = ctypes.windll.user32
dwmapi = ctypes.windll.dwmapi

# Attach thread to interactive desktop if running in a separate service/runner desktop
try:
    h_default_desk = user32.OpenDesktopW("default", 0, False, 0x0040 | 0x0001 | 0x0080)
    if h_default_desk:
        user32.SetThreadDesktop(h_default_desk)
except Exception:
    h_default_desk = None


# Known internal/system windows to exclude from selection
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
    """Check if window is cloaked (hidden UWP app or inactive virtual desktop)."""
    try:
        cloaked = ctypes.c_int(0)
        dwmapi.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
        return cloaked.value != 0
    except Exception:
        return False


def list_windows(include_minimized: bool = True) -> List[Tuple[int, str]]:
    """Enumerate visible application windows.

    Args:
        include_minimized: If True, includes minimized application windows.

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
    """Captures a specific window at high frame rates."""

    def __init__(self, target: Optional[str | int] = None):
        """Initialize capture for a window by title (substring) or HWND."""
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

        # Restore and activate window if minimized
        if self.hwnd and win32gui.IsIconic(self.hwnd):
            print(f"[+] Restoring minimized window: '{self.window_title}'...", flush=True)
            win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            try:
                win32gui.SetForegroundWindow(self.hwnd)
            except Exception:
                pass

    def _find_window_by_title(self, query: str) -> Optional[int]:
        query_lower = query.lower()
        active_windows = list_windows(include_minimized=True)

        for hwnd, title in active_windows:
            clean = title.replace(" [Minimized]", "").strip().lower()
            if clean == query_lower:
                return hwnd

        for hwnd, title in active_windows:
            clean = title.replace(" [Minimized]", "").strip().lower()
            if query_lower in clean:
                return hwnd

        return None

    def is_valid(self) -> bool:
        """Check if target window is still open and valid."""
        if not self.hwnd or not win32gui.IsWindow(self.hwnd):
            return False
        return True

    def _capture_printwindow(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture directly from the window's DC using PrintWindow (captures window even if covered)."""
        if not self.is_valid():
            return False, None

        try:
            left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
            w = right - left
            h = bottom - top
            if w <= 10 or h <= 10:
                return False, None

            hwnd_dc = win32gui.GetWindowDC(self.hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()

            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
            save_dc.SelectObject(save_bitmap)

            # PW_RENDERFULLCONTENT = 2
            result = user32.PrintWindow(self.hwnd, save_dc.GetSafeHdc(), 2)
            if not result:
                result = user32.PrintWindow(self.hwnd, save_dc.GetSafeHdc(), 1)

            if result:
                bmpinfo = save_bitmap.GetInfo()
                bmpstr = save_bitmap.GetBitmapBits(True)
                img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
                frame = np.ascontiguousarray(img[:, :, :3])
            else:
                frame = None

            # Clean up Windows GDI resources
            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwnd_dc)

            if frame is not None and frame.size > 0:
                return True, frame
            return False, None
        except Exception:
            return False, None

    def _capture_mss(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture via MSS screen crop with coordinate boundary clamping."""
        if not self.is_valid():
            return False, None

        try:
            _, _, w, h = win32gui.GetClientRect(self.hwnd)
            if w <= 10 or h <= 10:
                rect = win32gui.GetWindowRect(self.hwnd)
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                screen_x, screen_y = rect[0], rect[1]
            else:
                screen_x, screen_y = win32gui.ClientToScreen(self.hwnd, (0, 0))

            # Clamp coordinates to primary screen bounds to prevent BitBlt errors
            mon = self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]
            mon_left = mon["left"]
            mon_top = mon["top"]
            mon_right = mon_left + mon["width"]
            mon_bottom = mon_top + mon["height"]

            clamped_left = max(mon_left, int(screen_x))
            clamped_top = max(mon_top, int(screen_y))
            clamped_right = min(mon_right, int(screen_x + w))
            clamped_bottom = min(mon_bottom, int(screen_y + h))

            clamped_w = clamped_right - clamped_left
            clamped_h = clamped_bottom - clamped_top

            if clamped_w <= 10 or clamped_h <= 10:
                return False, None

            box = {
                "top": clamped_top,
                "left": clamped_left,
                "width": clamped_w,
                "height": clamped_h,
            }

            raw_shot = self.sct.grab(box)
            frame = np.frombuffer(raw_shot.raw, dtype=np.uint8).reshape((raw_shot.height, raw_shot.width, 4))
            return True, np.ascontiguousarray(frame[:, :, :3])
        except Exception:
            return False, None

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture one frame of the target window.

        Tries PrintWindow first (avoids obstruction by other windows),
        falling back to clamped MSS capture if PrintWindow is unsupported.
        """
        success, frame = self._capture_printwindow()
        if success and frame is not None and not np.all(frame == 0):
            return True, frame

        # Fallback to clamped MSS
        return self._capture_mss()

    def close(self):
        """Release capture resources."""
        if self.sct:
            try:
                self.sct.close()
            except Exception:
                pass
