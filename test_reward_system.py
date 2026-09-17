"""Unit tests for MinecraftAimEnv Reward System.

Verifies:
1. Immediate positive reward on Tick 1 when locked on yellow (#FFF500).
2. Streak bonus progression starting from Tick 1 (+0.2 per tick up to +12.0 max).
3. Immediate demerit (-2.0 pts/tick) when off yellow target, even during active aiming.
4. Heavy demerit (-2.5 to -5.0 pts/tick) when moving away from target.
5. Continuous demerit (-2.0 pts/tick) when target is not in view (scanning/flipping).
6. Streak reset upon losing target.
7. Parity between AimAgent score tracking and RL environment rewards.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from aim_env import MinecraftAimEnv


class MockAimEnv(MinecraftAimEnv):
    """Headless MockAimEnv with isolated inputs and simulated detections."""

    def __init__(self, min_lock_ticks: int = 1):
        self.width = 640
        self.height = 480
        self.max_episode_steps = 200
        self.horizontal_only = True
        self.min_lock_ticks = min_lock_ticks
        self.diag = float(np.hypot(self.width, self.height))

        # Mock hardware dependencies
        self.cap = MagicMock()
        self.cap.get_frame.return_value = (True, np.zeros((self.height, self.width, 3), dtype=np.uint8))
        self.detector = MagicMock()
        self.input_ctrl = MagicMock()

        # State tracking
        self.current_step = 0
        self.speed_multiplier = 1.0
        self.prev_dist_px = float(self.diag / 2.0)
        self.prev_dx = float(self.width / 2.0)
        self.prev_target_x = float(self.width / 2.0)
        self.prev_target_y = float(self.height / 2.0)
        self.prev_yaw_act = 0.0
        self.prev_pitch_act = 0.0
        self.lock_streak = 0
        self.consecutive_moving_ticks = 0
        self.target_seen_ticks = 0
        self.off_target_ticks = 0
        self.prev_has_target = False


def make_det(has_target=True, in_center=True, dx=0.0, dy=0.0, dist_px=0.0, height_zone="CENTER", height_ratio=0.42):
    return {
        "has_target": has_target,
        "target_x": 320.0 + dx,
        "target_y": 240.0 + dy,
        "dx": float(dx),
        "dy": float(dy),
        "dist_px": float(dist_px),
        "box_w": 40.0,
        "box_h": 60.0,
        "target_width": 40.0,
        "target_height": 60.0,
        "distance": 3.0,
        "height_ratio": height_ratio,
        "height_zone": height_zone,
        "in_center": in_center,
        "in_lock_zone": in_center,
        "crosshair_locked": in_center,
    }


class TestRewardSystem(unittest.TestCase):

    def setUp(self):
        self.env = MockAimEnv(min_lock_ticks=1)
        self.env.reset()

    def test_immediate_reward_on_tick_1_when_locked_in_center(self):
        """Crosshair in center of yellow body must award positive points starting immediately from Tick 1."""
        self.env.detector.detect.return_value = make_det(has_target=True, in_center=True, height_zone="CENTER")

        _, reward, _, info = self.env.step((8, 3))  # Hold still (0.0 px delta)
        self.assertEqual(info["lock_streak"], 1)
        self.assertAlmostEqual(reward, 8.0, places=2)
        self.assertTrue(info["in_lock_zone"])
        self.assertTrue(info["in_center"])

    def test_streak_progression_starting_from_tick_1(self):
        """Streak bonus must accumulate +0.20 per tick from tick 1 upwards."""
        self.env.detector.detect.return_value = make_det(has_target=True, in_center=True, height_zone="CENTER")

        # Tick 1: 8.0
        _, r1, _, info1 = self.env.step((8, 3))
        self.assertAlmostEqual(r1, 8.0, places=2)
        self.assertEqual(info1["lock_streak"], 1)

        # Tick 2: 8.2
        _, r2, _, info2 = self.env.step((8, 3))
        self.assertAlmostEqual(r2, 8.2, places=2)
        self.assertEqual(info2["lock_streak"], 2)

        # Tick 3: 8.4
        _, r3, _, info3 = self.env.step((8, 3))
        self.assertAlmostEqual(r3, 8.4, places=2)
        self.assertEqual(info3["lock_streak"], 3)

        # Dwell to max streak (+4.0 max bonus => 12.0 total)
        for _ in range(25):
            _, reward, _, _ = self.env.step((8, 3))
        self.assertAlmostEqual(reward, 12.0, places=2)

    def test_no_score_when_aiming_at_feet(self):
        """Aiming at feet (height_zone='FEET') must NOT give score; it must deduct points (-2.0)."""
        self.env.detector.detect.return_value = make_det(has_target=True, in_center=False, height_zone="FEET", height_ratio=0.85)

        _, reward, _, info = self.env.step((8, 3))
        self.assertEqual(reward, -2.0, "Must deduct points when aiming at feet instead of center")
        self.assertEqual(info["lock_streak"], 0)
        self.assertFalse(info["in_center"])

    def test_no_score_when_aiming_at_head(self):
        """Aiming at head (height_zone='HEAD') must NOT give score; it must deduct points (-2.0)."""
        self.env.detector.detect.return_value = make_det(has_target=True, in_center=False, height_zone="HEAD", height_ratio=0.10)

        _, reward, _, info = self.env.step((8, 3))
        self.assertEqual(reward, -2.0, "Must deduct points when aiming at head instead of center")
        self.assertEqual(info["lock_streak"], 0)
        self.assertFalse(info["in_center"])

    def test_strict_demerit_when_off_yellow(self):
        """Any tick off yellow must deduct points (-2.0), even if performing an action towards target."""
        # Initial step: target visible on right (dx=50.0), not locked
        self.env.detector.detect.return_value = make_det(has_target=True, in_center=False, dx=50.0, dist_px=50.0)

        # Taking action moving towards target (yaw action +32px)
        _, reward, _, info = self.env.step((12, 3))
        self.assertEqual(reward, -2.0, "Must deduct 2.0 points when off target")
        self.assertEqual(info["lock_streak"], 0)
        self.assertFalse(info["in_lock_zone"])

    def test_heavy_demerit_when_moving_away_from_yellow(self):
        """Moving away from target must incur severe demerits (-2.5 to -5.0)."""
        # Step 1: enemy at dx = 50.0
        self.env.detector.detect.return_value = make_det(has_target=True, in_center=False, dx=50.0, dist_px=50.0)
        self.env.step((8, 3))

        # Step 2: bot turned left (-50px) instead of right, dx increased to 100.0 (delta_err = +50.0)
        self.env.detector.detect.return_value = make_det(has_target=True, in_center=False, dx=100.0, dist_px=100.0)
        _, reward, _, _ = self.env.step((3, 3))
        self.assertLess(reward, -2.5, "Moving away must result in penalty steeper than -2.5")
        self.assertGreaterEqual(reward, -5.0)

    def test_demerit_when_target_not_visible(self):
        """When target is not in view at all, demerit (-2.0) is deducted continuously."""
        self.env.detector.detect.return_value = make_det(has_target=False, in_center=False)

        _, reward, _, info = self.env.step((8, 3))
        self.assertEqual(reward, -2.0)
        self.assertEqual(info["lock_streak"], 0)

    def test_instant_streak_reset_on_target_loss(self):
        """Losing target after acquiring high streak must immediately reset streak and deduct points."""
        self.env.detector.detect.return_value = make_det(has_target=True, in_center=True, height_zone="CENTER")

        # Build streak
        for _ in range(5):
            self.env.step((8, 3))
        self.assertEqual(self.env.lock_streak, 5)

        # Lose target
        self.env.detector.detect.return_value = make_det(has_target=True, in_center=False, dx=35.0, dist_px=35.0)
        _, reward, _, info = self.env.step((8, 3))
        self.assertEqual(info["lock_streak"], 0)
        self.assertLess(reward, 0.0)

    def test_wild_sweep_across_target_gets_demerit(self):
        """A wild flick (+75px or +160px) sweeping past enemy must reset streak and deduct points."""
        self.env.detector.detect.return_value = make_det(has_target=True, in_center=True, height_zone="CENTER")

        # Step 1: locked
        self.env.step((8, 3))
        self.assertEqual(self.env.lock_streak, 1)

        # Step 2: wild flick (+75px/tick)
        _, reward, _, info = self.env.step((14, 3))
        self.assertEqual(reward, -2.0)
        self.assertEqual(info["lock_streak"], 0)

    def test_cursor_on_yellow_awards_points_even_if_bbox_distorted(self):
        """When cursor is directly on yellow (#FFF500), points must be awarded even if bbox is wide or distorted."""
        det = {
            "has_target": True,
            "target_x": 453.0,
            "target_y": 226.0,
            "dx": 133.0,
            "dy": -14.0,
            "dist_px": 134.0,
            "raw_dx": 133.0,
            "raw_dy": -14.0,
            "vx": 0.0,
            "vy": 0.0,
            "box_w": 323.0,
            "box_h": 418.0,
            "target_height": 418.0,
            "target_width": 323.0,
            "height_ratio": 0.45,
            "height_zone": "CENTER",
            "in_center": False,
            "in_lock_zone": True,
            "crosshair_locked": True,
            "crosshair_color": "YELLOW_LOCKED",
            "distance": 1.48,
            "confidence": 1.0,
            "cue": "highlight_yellow",
            "tier": "full",
            "is_facing_sky": False,
        }
        self.env.detector.detect.return_value = det

        # Tick 1: Must award positive lock reward (+8.0)
        _, r1, _, info1 = self.env.step((8, 3))
        self.assertAlmostEqual(r1, 8.0, places=2, msg="Direct yellow contact must award +8.0 pts on tick 1")
        self.assertEqual(info1["lock_streak"], 1)
        self.assertTrue(info1["in_lock_zone"])
        self.assertTrue(info1["crosshair_locked"])

        # Tick 2: Streak accumulation (+8.2)
        _, r2, _, info2 = self.env.step((8, 3))
        self.assertAlmostEqual(r2, 8.2, places=2, msg="Direct yellow contact must accumulate streak bonus")
        self.assertEqual(info2["lock_streak"], 2)


class TestAimAgentRewardLogic(unittest.TestCase):

    def test_aim_agent_reward_parity(self):
        """Simulate AimAgent immediate yellow rewards and off-target demerits."""
        min_lock_ticks = 1
        lock_streak = 0
        cumulative_score = 0.0

        def process_tick(in_lock: bool):
            nonlocal lock_streak, cumulative_score
            if in_lock:
                lock_streak += 1
                r_lock = 8.0
                r_streak = min(4.0, max(0.0, lock_streak - min_lock_ticks) * 0.20)
                step_reward = float(r_lock + r_streak)
            else:
                lock_streak = 0
                step_reward = -2.0

            cumulative_score += step_reward
            return step_reward, lock_streak, cumulative_score

        # 1. First tick on yellow: immediate +8.0 reward
        r1, s1, total1 = process_tick(in_lock=True)
        self.assertEqual(r1, 8.0)
        self.assertEqual(s1, 1)
        self.assertEqual(total1, 8.0)

        # 2. Second tick on yellow: +8.2 reward
        r2, s2, total2 = process_tick(in_lock=True)
        self.assertEqual(r2, 8.2)
        self.assertEqual(s2, 2)
        self.assertEqual(total2, 16.2)

        # 3. Off yellow: -2.0 demerit
        r3, s3, total3 = process_tick(in_lock=False)
        self.assertEqual(r3, -2.0)
        self.assertEqual(s3, 0)
        self.assertEqual(total3, 14.2)

        # 4. Continues off yellow: another -2.0 demerit
        r4, s4, total4 = process_tick(in_lock=False)
        self.assertEqual(r4, -2.0)
        self.assertEqual(s4, 0)
        self.assertEqual(total4, 12.2)


class TestHeightSensingAndCenterLock(unittest.TestCase):

    def test_detector_height_sensing_and_center_lock(self):
        """Test FastAimDetector with synthetic image to verify height zones and center lock."""
        from fast_aim_detector import FastAimDetector
        import cv2

        detector = FastAimDetector((640, 480))
        cx, cy = 320, 240

        # Create 640x480 black image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Target centered on crosshair: box at x: [300, 340], y: [200, 300] (bh=100, bw=40)
        # Torso center is around 200 + 42 = 242. Crosshair is at (320, 240) -> right in torso center!
        frame[200:300, 300:340] = [0, 245, 255]  # Yellow #FFF500 (BGR)

        det = detector.detect(frame)
        self.assertTrue(det["has_target"])
        self.assertEqual(det["box_h"], 100.0)
        self.assertEqual(det["box_w"], 40.0)
        self.assertEqual(det["height_zone"], "CENTER")
        self.assertTrue(det["in_center"], "Must be in center when crosshair is at torso center")
        self.assertTrue(det["in_lock_zone"])

        # 2. Target shifted up so crosshair points at FEET: box at y: [140, 240]
        # Crosshair cy=240 is at the very bottom (feet) of the box
        frame_feet = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_feet[140:240, 300:340] = [0, 245, 255]
        det_feet = detector.detect(frame_feet)
        self.assertTrue(det_feet["has_target"])
        self.assertIn(det_feet["height_zone"], ("FEET", "BELOW"))
        self.assertFalse(det_feet["in_center"], "Crosshair at feet must NOT be in center")

        # 3. Target shifted down so crosshair points at HEAD: box at y: [238, 338]
        # Crosshair cy=240 is near the very top (head) of the box
        frame_head = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_head[238:338, 300:340] = [0, 245, 255]
        det_head = detector.detect(frame_head)
        self.assertTrue(det_head["has_target"])
        self.assertEqual(det_head["height_zone"], "HEAD")
        self.assertFalse(det_head["in_center"], "Crosshair at head must NOT be in center")


if __name__ == "__main__":
    unittest.main()
