"""Real-time computer vision detector for Minecraft PvP enemies.

Detects player hitboxes (F3+B red eye-lines & white wireframe), nametags, and player silhouettes.
Computes crosshair aiming deltas (dx, dy) and combat reach distance.
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


class VisionDetector:
    """Detects enemy players in 640x480 (or arbitrary resolution) Minecraft frames."""

    def __init__(self, target_resolution: Tuple[int, int] = (640, 480)):
        self.width, self.height = target_resolution
        self.crosshair_x = self.width // 2
        self.crosshair_y = self.height // 2

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze a frame and detect enemy target position and aim offsets.

        Args:
            frame: BGR image from Minecraft capture.

        Returns:
            Dictionary containing target location, dx/dy aiming deltas, range, and confidence.
        """
        h, w = frame.shape[:2]
        ch_x = w // 2
        ch_y = h // 2

        # 1. Mask out player HUD / Hotbar area (bottom 18% of screen) and top bar (5%)
        active_mask = np.ones((h, w), dtype=bool)
        active_mask[int(h * 0.82):, :] = False  # Ignore hearts / hotbar
        active_mask[:int(h * 0.04), :] = False  # Ignore top margin

        # Ignore small region directly around own crosshair to prevent crosshair self-detection
        ch_r = 10
        active_mask[ch_y - ch_r:ch_y + ch_r, ch_x - ch_r:ch_x + ch_r] = False

        target_x: Optional[float] = None
        target_y: Optional[float] = None
        box_w: float = 0.0
        box_h: float = 0.0
        confidence: float = 0.0
        det_type: str = "none"

        # --- Detection Strategy 1: F3+B Red Eye-Level Line ---
        # The F3+B red eye-line is bright red with a distinct horizontal aspect ratio (width >> height)
        red_mask = (frame[:, :, 2] > 170) & (frame[:, :, 1] < 70) & (frame[:, :, 0] < 70) & active_mask
        if np.any(red_mask):
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask.astype(np.uint8))
            best_line_idx = -1
            max_line_width = 0

            for i in range(1, num_labels):
                bx, by, bw, bh, area = stats[i]
                aspect = bw / float(bh) if bh > 0 else 0
                # A horizontal line has high aspect ratio and reasonable width
                if aspect >= 3.5 and bw >= 15 and bh <= 35:
                    if bw > max_line_width:
                        max_line_width = bw
                        best_line_idx = i

            if best_line_idx != -1:
                cx, cy = centroids[best_line_idx]
                bx, by, bw, bh, _ = stats[best_line_idx]
                target_x = float(cx)
                # Aim at upper chest (slightly below eyes)
                estimated_body_h = bw * 2.5
                target_y = float(cy + estimated_body_h * 0.20)
                box_w = float(bw)
                box_h = float(estimated_body_h)
                confidence = 0.95
                det_type = "f3b_eyeline"

        # --- Detection Strategy 2: White Hitbox Wireframe (F3+B) ---
        if target_x is None:
            white_mask = (frame[:, :, 0] > 230) & (frame[:, :, 1] > 230) & (frame[:, :, 2] > 230) & active_mask
            white_u8 = (white_mask.astype(np.uint8)) * 255

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(white_u8, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_box = None
            max_score = 0.0

            for c in contours:
                bx, by, bw, bh = cv2.boundingRect(c)
                # Player hitbox or nametag bounding box
                if (bw > 20 and bh > 30) or (bw > 40 and bh > 12):
                    dist_to_center = np.hypot((bx + bw / 2) - ch_x, (by + bh / 2) - ch_y)
                    score = (bw * bh) / (1.0 + dist_to_center * 0.3)
                    if score > max_score:
                        max_score = score
                        best_box = (bx, by, bw, bh)

            if best_box is not None:
                bx, by, bw, bh = best_box
                target_x = float(bx + bw / 2.0)
                target_y = float(by + bh / 2.0)
                box_w = float(bw)
                box_h = float(bh)
                confidence = 0.80
                det_type = "hitbox_wireframe"

        # If no target found
        if target_x is None:
            return {
                "has_target": False,
                "target_x": float(ch_x),
                "target_y": float(ch_y),
                "dx": 0.0,
                "dy": 0.0,
                "box_w": 0.0,
                "box_h": 0.0,
                "confidence": 0.0,
                "type": "none",
                "in_attack_range": False,
            }

        # Calculate crosshair aiming deltas
        dx = float(target_x - ch_x)
        dy = float(target_y - ch_y)

        # Attack reach check:
        # 1. Target is near crosshair (within aiming cone)
        # 2. Target apparent size indicates within ~3.0 block reach
        dist_to_crosshair = np.hypot(dx, dy)
        in_crosshair_cone = dist_to_crosshair < (w * 0.18)
        close_enough = box_w > (w * 0.10) or box_h > (h * 0.18)
        in_attack_range = in_crosshair_cone and close_enough

        return {
            "has_target": True,
            "target_x": target_x,
            "target_y": target_y,
            "dx": dx,
            "dy": dy,
            "box_w": box_w,
            "box_h": box_h,
            "confidence": confidence,
            "type": det_type,
            "in_attack_range": in_attack_range,
        }

    def draw_hud(self, frame: np.ndarray, det: Dict[str, Any], actions: Optional[Dict[str, bool]] = None) -> np.ndarray:
        """Draw interactive HUD overlay on frame for visualization and debugging."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        ch_x = w // 2
        ch_y = h // 2

        # Draw crosshair indicator
        cv2.drawMarker(vis, (ch_x, ch_y), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)

        if det["has_target"]:
            tx = int(det["target_x"])
            ty = int(det["target_y"])

            # Color: Green if in attack range, Yellow if aiming/tracking
            color = (0, 255, 0) if det["in_attack_range"] else (0, 255, 255)

            # Draw target tracking circle and aim line
            cv2.circle(vis, (tx, ty), 12, color, 2)
            cv2.line(vis, (ch_x, ch_y), (tx, ty), color, 2)

            # Draw target bounding box if dimensions available
            bw = int(det["box_w"])
            bh = int(det["box_h"])
            if bw > 0 and bh > 0:
                x1 = max(0, int(tx - bw / 2))
                y1 = max(0, int(ty - bh / 2))
                cv2.rectangle(vis, (x1, y1), (x1 + bw, y1 + bh), color, 2)

            # Status label
            status_text = "LOCK: IN RANGE [PUNCH]" if det["in_attack_range"] else "LOCK: AIMING & PURSUING"
            cv2.putText(vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(
                vis,
                f"Aim Delta: dx={det['dx']:.1f}, dy={det['dy']:.1f} | Conf: {det['confidence']*100:.0f}%",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        else:
            cv2.putText(vis, "STATUS: SEARCHING FOR ENEMY...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Draw active controls HUD
        if actions:
            hud_str = f"Keys: {'W' if actions.get('w') else '-'} {'A' if actions.get('a') else '-'} {'S' if actions.get('s') else '-'} {'D' if actions.get('d') else '-'} | Punch: {'YES' if actions.get('attack') else '-'}"
            cv2.putText(vis, hud_str, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return vis
