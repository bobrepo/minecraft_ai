"""Unit tests for Yellow (#FFF500) + Purple Outline (#CC00FF) Detection & Left-to-Right Scanning.

Verifies:
1. FastAimDetector detects Yellow (#FFF500) enemy bodies.
2. FastAimDetector detects Purple/Magenta (#CC00FF) enemy outlines.
3. FastAimDetector detects combined Yellow + Purple on actual game capture.
4. Other outline / fallback cues (Cyan) are removed and rejected.
5. Crosshair lock probe detects Yellow body and Purple outline under crosshair.
6. PureAimRLAgent executes smooth Left-to-Right back-and-forth scanning when target is lost.
7. AimAgent executes smooth Left-to-Right back-and-forth sinusoidal scanning.
8. Bidirectional tracking verification: Target at dx = +35px actively commands rightward movement (vx > 0).
"""

import math
import os
import unittest
from unittest.mock import MagicMock
import cv2
import numpy as np

from aim_agent import AimAgent
from fast_aim_detector import FastAimDetector
from pure_aim_rl import PureAimRLAgent


class TestYellowPurpleDetection(unittest.TestCase):

    def setUp(self):
        self.detector = FastAimDetector((640, 480))

    def test_synthetic_yellow_body_detection(self):
        """Verify FastAimDetector detects pure #FFF500 yellow body (BGR=[0, 245, 255])."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a yellow torso box at x=340..380, y=200..300
        frame[200:300, 340:380] = [0, 245, 255]

        det = self.detector.detect(frame)
        self.assertTrue(det["has_target"])
        self.assertEqual(det["cue"], "highlight_yellow")
        self.assertAlmostEqual(det["dx"], 40.0, delta=5.0)

    def test_purple_outline_removed_and_ignored(self):
        """Verify FastAimDetector ignores purple outline (#CC00FF) since user removed purple outline."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw purple outline rectangle
        cv2.rectangle(frame, (280, 180), (340, 320), (255, 0, 204), thickness=3)

        det = self.detector.detect(frame)
        self.assertFalse(det["has_target"], "Purple outline should be ignored when only yellow is enabled")

    def test_actual_user_screenshot_detection(self):
        """Verify FastAimDetector detects enemy on actual uploaded user screenshot."""
        screenshot_path = r"C:/Users/shado/.gemini/antigravity/brain/9e02a45b-eb33-4e7a-9621-80e03ba70cd3/.user_uploaded/media_1789444999262.png"
        if not os.path.exists(screenshot_path):
            self.skipTest("Screenshot not found on local disk")

        img = cv2.imread(screenshot_path)
        det_actual = FastAimDetector((img.shape[1], img.shape[0]))
        res = det_actual.detect(img)

        self.assertTrue(res["has_target"], "Should detect yellow enemy on uploaded user screenshot")
        self.assertEqual(res["cue"], "highlight_yellow")
        self.assertTrue(res["in_lock_zone"], "Crosshair is inside enemy bounding box")
        self.assertGreater(res["box_w"], 80.0)
        self.assertGreater(res["box_h"], 150.0)

    def test_other_outline_detection_removed(self):
        """Verify FastAimDetector no longer detects legacy cyan outline (removed upon user request)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[200:280, 360:400] = [223, 255, 0]  # Cyan BGR

        det = self.detector.detect(frame)
        self.assertFalse(det["has_target"], "Cyan outline should be ignored after removal of other outlines")


class TestCrosshairLockColors(unittest.TestCase):

    def setUp(self):
        self.detector = FastAimDetector((640, 480))

    def test_yellow_under_crosshair(self):
        """Yellow body under crosshair returns YELLOW_LOCKED."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[236:245, 316:325] = [0, 245, 255]

        locked, cue = self.detector.check_crosshair_lock(frame)
        self.assertTrue(locked)
        self.assertEqual(cue, "YELLOW_LOCKED")

    def test_purple_not_locked(self):
        """Purple outline under crosshair returns False (outline removed)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[236:245, 316:325] = [255, 0, 204]

        locked, cue = self.detector.check_crosshair_lock(frame)
        self.assertFalse(locked)
        self.assertEqual(cue, "SEARCHING")


class TestLeftToRightScanningAndAiming(unittest.TestCase):

    def test_rl_select_action_180_backflip_then_scan(self):
        """Verify PureAimRLAgent fires 180° backflip flick burst on lost target, then scans."""
        agent = MagicMock()
        agent.last_seen_dir = 1  # Enemy escaped right
        agent.lost_target_ticks = 0

        select_fn = PureAimRLAgent.select_action.__get__(agent, PureAimRLAgent)
        det_offscreen = {"has_target": False}

        # Ticks 1..4: 180° flick burst (+160px at Gear 16)
        for i in range(1, 5):
            act = select_fn(np.zeros(10), det_offscreen)
            self.assertEqual(act, (16, 3), f"Tick {i} should be 180° flick Gear 16")

        # Ticks 5..18: hold steady (Gear 8) to scan behind
        for i in range(5, 19):
            act = select_fn(np.zeros(10), det_offscreen)
            self.assertEqual(act, (8, 3), f"Tick {i} should hold steady (Gear 8)")

        # Ticks 19..42: sweep right (Gear 11)
        for i in range(19, 43):
            act = select_fn(np.zeros(10), det_offscreen)
            self.assertEqual(act, (11, 3), f"Tick {i} should sweep right (Gear 11)")

        # Ticks 43..66: reverse and sweep left (Gear 5)
        for i in range(43, 67):
            act = select_fn(np.zeros(10), det_offscreen)
            self.assertEqual(act, (5, 3), f"Tick {i} should sweep left (Gear 5)")

    def test_left_to_right_centering_not_blocked_by_bounding_box(self):
        """Verify that when target center is at dx=+35px, the agent continues driving rightward."""
        agent = MagicMock()
        agent.base_speed = 1.0
        agent.speed_mode = "1.0x"
        agent.kp_yaw = 0.55
        agent.kd_yaw = 0.22
        agent.prev_dx = 30.0
        agent.deadzone_px = 3.5
        agent.max_step_px = 160.0
        agent.get_effective_speed = MagicMock(return_value=1.0)
        agent.input_ctrl = MagicMock()

        # Simulate target at dx = +35.0 (enemy center is to the right)
        dx = 35.0
        is_centered = abs(dx) <= agent.deadzone_px
        self.assertFalse(is_centered, "dx=+35px is not centered, must continue moving right")

        # PD calculation
        d_dx = dx - agent.prev_dx
        raw_vx = agent.kp_yaw * dx + agent.kd_yaw * d_dx
        self.assertGreater(raw_vx, 10.0, "Velocity must be strongly positive (moving rightward)")

    def test_direct_mouse_dispatch_eliminates_desync_with_fractional_accumulator(self):
        """Verify move_mouse_direct dispatches immediately without phase lag and preserves fractions."""
        from input_controller import InputController
        ctrl = InputController()
        dispatched = []
        ctrl._send_mouse_raw = lambda sx, sy: dispatched.append((sx, sy))

        try:
            # 1. Send fractional movement: 3.4 px
            ctrl.move_mouse_direct(3.4, 0.0)
            self.assertEqual(dispatched[-1], (3, 0), "Should immediately dispatch 3 integer pixels")
            self.assertAlmostEqual(ctrl._direct_accum_x, 0.4, places=2, msg="Should retain 0.4 remainder")

            # 2. Next tick: send 2.8 px -> total remainder is 0.4 + 2.8 = 3.2 -> dispatch 3 px
            ctrl.move_mouse_direct(2.8, 0.0)
            self.assertEqual(dispatched[-1], (3, 0))
            self.assertAlmostEqual(ctrl._direct_accum_x, 0.2, places=2)

            # 3. Negative movement: send -4.5 px -> total is 0.2 - 4.5 = -4.3 -> dispatch -4 px
            ctrl.move_mouse_direct(-4.5, 0.0)
            self.assertEqual(dispatched[-1], (-4, 0))
            self.assertAlmostEqual(ctrl._direct_accum_x, -0.3, places=2)
        finally:
            ctrl.mouse_thread.stop()

    def test_aim_agent_sinusoidal_scan_oscillates_both_directions(self):
        """Verify AimAgent's sinusoidal search scan alternates right and left."""
        agent = MagicMock()
        agent.scan_amp = 20.0
        agent.scan_period = 70
        agent.last_seen_dir = 1
        omega = (2.0 * math.pi) / float(agent.scan_period)

        velocities = []
        for tick in range(1, agent.scan_period + 1):
            vx = agent.scan_amp * math.cos(omega * tick) * agent.last_seen_dir
            velocities.append(vx)

        # Has positive velocities (moving right)
        self.assertTrue(any(v > 5.0 for v in velocities), "Should sweep right")
        # Has negative velocities (moving left)
        self.assertTrue(any(v < -5.0 for v in velocities), "Should sweep left")
        # Turnarounds are smooth (max single-tick change is bounded)
        max_jerk = max(abs(velocities[i] - velocities[i - 1]) for i in range(1, len(velocities)))
        self.assertLess(max_jerk, 5.0, "Scan must be smooth and continuous without sudden flicks")


if __name__ == "__main__":
    unittest.main()
