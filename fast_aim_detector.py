"""Ultra-fast real-time opponent detector for Stage 1 Pure Aim RL.

Specialized for Minecraft PvP with Crosshair-Centric Outward Scanning:
1. Cyan-highlighted enemy body detection (BGRr=[223,255,0]) — primary cue only.
2. Red crosshair lock sensor: center pixel turns RED when crosshair is over enemy.
3. Sky mask to detect if looking at empty sky.

Crosshair-Centric scanning probes the central region of interest first,
reducing detection latency to sub-0.5ms.

Crosshair color states (from actual game capture):
  - SEARCHING:   Center pixel is near-black/transparent [14, 7, 6] — white '+' arms offset from center
  - RED_LOCKED:  Center pixel turns bright RED [33, 1, 255] BGR — crosshair over cyan enemy body
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


class FastAimDetector:
    """Ultra-fast (<0.4ms) crosshair-centric multi-cue enemy detector."""

    def __init__(self, resolution: Tuple[int, int] = (640, 480)):
        self.width, self.height = resolution
        self.cx = self.width // 2
        self.cy = self.height // 2

        # 1. Highlighted Enemy: Cyan-green glowing body
        #    Actual pixel from image: BGR=[223, 255, 0] (B=223, G=255, R=0)
        #    Range with tolerance for lighting variation:
        self.lower_cyan_bgr = np.array([170, 210, 0], dtype=np.uint8)
        self.upper_cyan_bgr = np.array([255, 255, 30], dtype=np.uint8)

        # 2. Sky color boundary (to detect if looking at empty sky)
        self.lower_sky = np.array([170, 110, 0], dtype=np.uint8)
        self.upper_sky = np.array([255, 255, 175], dtype=np.uint8)

        # Minecraft 70 deg vertical FOV focal length at current height
        self.fy = self.height / (2.0 * np.tan(np.radians(35.0)))

    def _detect_in_roi(
        self,
        roi: np.ndarray,
        offset_x: int,
        offset_y: int,
        ch_x: int,
        ch_y: int,
    ) -> Optional[Tuple[float, float, float, float, float, str]]:
        """Scan a specific sub-region (ROI) for the cyan-highlighted enemy body."""
        rh, rw = roi.shape[:2]
        if rh < 10 or rw < 10:
            return None

        # --- Cyan Highlighted Enemy ---
        # Actual BGR from image analysis: [223, 255, 0]  (B=223, G=255, R=0)
        cyan_mask = cv2.inRange(roi, self.lower_cyan_bgr, self.upper_cyan_bgr)
        n_cyan = cv2.countNonZero(cyan_mask)
        if n_cyan > 60:
            cnts, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c) > 50:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    # Aim at horizontal center, upper-center (30% from top = torso/chest area)
                    tx = offset_x + bx + bw / 2.0
                    ty = offset_y + by + bh * 0.30
                    return tx, ty, float(bw), float(bh), 1.0, "highlight_cyan"

        return None

    def check_crosshair_lock(self, frame: np.ndarray) -> Tuple[bool, str]:
        """Instant sub-0.001ms crosshair color probe to determine if crosshair is directly over enemy.

        Ground truth colors from actual game capture:
        - SEARCHING (not over enemy): Center pixel is near-black / transparent
          BGR center ≈ [14, 7, 6] — the crosshair gap (dark transparent hole between white arms)
          White arms offset from center: BGR ≈ [241, 248, 249]
        - RED_LOCKED (over enemy): Center pixel turns bright RED
          BGR ≈ [33, 1, 255] → R=255, G=1, B=33 — crosshair changes color over cyan body
        """
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        # Sample a 5x5 patch around the crosshair center
        patch = frame[max(0, cy - 2) : min(h, cy + 3), max(0, cx - 2) : min(w, cx + 3)]
        if patch.size == 0:
            return False, "UNKNOWN"

        # Check for RED crosshair lock: R > 200, G < 50, B < 80
        # (in BGR array: channel 2 = R, channel 1 = G, channel 0 = B)
        red_pixels = (patch[:, :, 2] > 200) & (patch[:, :, 0] < 80) & (patch[:, :, 1] < 50)
        if np.any(red_pixels):
            return True, "RED_LOCKED"

        # Not locked: crosshair center is near-black (transparent gap between white arms)
        return False, "SEARCHING"

    def detect(self, frame: np.ndarray, crosshair_centric: bool = True) -> Dict[str, Any]:
        """Detect opponent head & body target with crosshair-centric outward scanning."""
        h, w = frame.shape[:2]
        ch_x = w // 2
        ch_y = h // 2

        # Instant crosshair color lock probe (<0.001ms)
        crosshair_locked, crosshair_color = self.check_crosshair_lock(frame)

        # Fast subsampled SIMD sky dominance check (0.3ms)
        sub_frame = frame[::4, ::4]
        sky_mask = cv2.inRange(sub_frame, self.lower_sky, self.upper_sky)
        total_sky_ratio = float(cv2.mean(sky_mask)[0]) / 255.0
        lower_sky = sky_mask[int(sky_mask.shape[0] * 0.45) :, :]
        lower_sky_ratio = float(cv2.mean(lower_sky)[0]) / 255.0 if lower_sky.size > 0 else 0.0
        is_facing_sky = (total_sky_ratio > 0.65) or (lower_sky_ratio > 0.38)

        if is_facing_sky:
            return {
                "has_target": False,
                "target_x": float(ch_x),
                "target_y": float(ch_y),
                "dx": 0.0,
                "dy": 0.0,
                "box_w": 0.0,
                "box_h": 0.0,
                "distance": 99.0,
                "confidence": 0.0,
                "cue": "none",
                "tier": "none",
                "is_facing_sky": True,
                "in_lock_zone": False,
                "crosshair_locked": False,
                "crosshair_color": crosshair_color,
            }

        res: Optional[Tuple[float, float, float, float, float, str]] = None
        tier = "none"

        if crosshair_centric:
            # Tier 1: Inner Crosshair Focus Zone (+/- 160px) - sub-0.2ms
            hw1, hh1 = 160, 160
            x1_1, y1_1 = max(0, ch_x - hw1), max(0, ch_y - hh1)
            x2_1, y2_1 = min(w, ch_x + hw1), min(h, ch_y + hh1)
            res = self._detect_in_roi(frame[y1_1:y2_1, x1_1:x2_1], x1_1, y1_1, ch_x, ch_y)
            tier = "inner"

            # Tier 2: Mid Zone (+/- 320px x +/- 240px)
            if res is None:
                hw2, hh2 = 320, 240
                x1_2, y1_2 = max(0, ch_x - hw2), max(0, ch_y - hh2)
                x2_2, y2_2 = min(w, ch_x + hw2), min(h, ch_y + hh2)
                res = self._detect_in_roi(frame[y1_2:y2_2, x1_2:x2_2], x1_2, y1_2, ch_x, ch_y)
                tier = "mid"

        # Tier 3: Full Combat ROI Fallback (5% to 85% height)
        if res is None:
            y_top = int(h * 0.05)
            y_bot = int(h * 0.85)
            res = self._detect_in_roi(frame[y_top:y_bot, :], 0, y_top, ch_x, ch_y)
            tier = "full"

        if res is not None:
            tx, ty, bw, bh, conf, cue = res
            dx = float(tx - ch_x)
            dy = float(ty - ch_y)
            dist_px = float(np.hypot(dx, dy))

            # Calibrated 3D distance Z = (fy * 1.8) / box_h
            dist_est = float(np.clip((self.fy * 1.8) / max(10.0, bh), 0.5, 30.0))

            # Dead-center lock zone: crosshair is within +/- 22px of target or red crosshair color matches
            in_lock_zone = crosshair_locked or ((abs(dx) <= 22.0) and (abs(dy) <= 22.0))

            return {
                "has_target": True,
                "target_x": tx,
                "target_y": ty,
                "dx": dx,
                "dy": dy,
                "dist_px": dist_px,
                "box_w": bw,
                "box_h": bh,
                "distance": round(dist_est, 2),
                "confidence": conf,
                "cue": cue,
                "tier": tier,
                "is_facing_sky": False,
                "in_lock_zone": in_lock_zone,
                "crosshair_locked": crosshair_locked,
                "crosshair_color": crosshair_color,
            }

        return {
            "has_target": False,
            "target_x": float(ch_x),
            "target_y": float(ch_y),
            "dx": 0.0,
            "dy": 0.0,
            "dist_px": 999.0,
            "box_w": 0.0,
            "box_h": 0.0,
            "distance": 99.0,
            "confidence": 0.0,
            "cue": "none",
            "tier": "none",
            "is_facing_sky": False,
            "in_lock_zone": crosshair_locked,
            "crosshair_locked": crosshair_locked,
            "crosshair_color": crosshair_color,
        }
