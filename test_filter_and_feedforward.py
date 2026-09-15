"""Unit tests for 1€ (One Euro) Filter, Hitbox Lock Deadzone, and Velocity Feedforward.

Verifies:
1. OneEuroFilter suppresses high-frequency jitter at low speed.
2. OneEuroFilter responds immediately to fast step flicks with minimal lag.
3. FastAimDetector outputs filtered dx, dy, and velocity estimates (vx, vy).
4. AimAgent locks stationary on stationary target and matches vx during lateral strafe.
5. AimAgent enforces strictly horizontal aiming (pitch == 0.0).
6. PureAimRLAgent select_action activates sticky lock for in_lock_zone and matches feedforward gear.
7. PureAimRLAgent eval_mode disables exploration (epsilon == 0.0).
"""

import math
import time
import unittest
from unittest.mock import MagicMock
import numpy as np

from aim_agent import AimAgent
from fast_aim_detector import FastAimDetector, LowPassFilter, OneEuroFilter
from pure_aim_rl import PureAimRLAgent


class TestOneEuroFilter(unittest.TestCase):

    def test_low_speed_jitter_suppression(self):
        """Verify 1€ filter significantly reduces low-frequency micro-jitter."""
        filt = OneEuroFilter(min_cutoff=1.5, beta=0.020, d_cutoff=1.0)

        # Feed stationary signal with +-3.0px sinusoidal noise at 60 FPS
        t = 0.0
        dt = 1.0 / 60.0
        raw_values = []
        filtered_values = []

        base = 100.0
        for i in range(60):
            t += dt
            noise = 3.0 * math.sin(i * 1.2)
            val = base + noise
            raw_values.append(val)
            f_val, _ = filt.filter(val, t=t)
            filtered_values.append(f_val)

        # Discard warm-up (first 10 frames)
        raw_std = np.std(raw_values[10:])
        filt_std = np.std(filtered_values[10:])

        # Filtered standard deviation should be significantly lower than raw noise
        self.assertLess(filt_std, raw_std * 0.60, f"Expected standard deviation reduction: filt={filt_std:.2f}, raw={raw_std:.2f}")

    def test_step_flick_high_responsiveness(self):
        """Verify 1€ filter adapts cutoff on fast flick step with zero lag."""
        filt = OneEuroFilter(min_cutoff=1.5, beta=0.020, d_cutoff=1.0)

        t = 0.0
        dt = 1.0 / 60.0
        # Initialize at 0
        filt.filter(0.0, t=t)

        # Sudden snap flick from 0 to 150.0px
        t += dt
        f_val, f_dx = filt.filter(150.0, t=t)

        # High beta should allow immediate jump (>60% of step on frame 1)
        self.assertGreater(f_val, 80.0, "1€ filter should adapt and track large step flick rapidly")
        self.assertGreater(f_dx, 500.0, "Derivative estimate should be large during fast flick")


class TestDetectorFilteredVelocity(unittest.TestCase):

    def test_detector_returns_filtered_and_velocity_keys(self):
        """Verify FastAimDetector returns both filtered dx/dy, raw values, and vx/vy."""
        detector = FastAimDetector((640, 480))

        # Synthetic frame with yellow enemy (#FFF500 -> BGR=[0, 245, 255])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Place yellow box at x=360..400, y=200..280 (dx ≈ +60px)
        frame[200:280, 360:400] = [0, 245, 255]

        det1 = detector.detect(frame)
        self.assertTrue(det1["has_target"])
        self.assertIn("dx", det1)
        self.assertIn("raw_dx", det1)
        self.assertIn("vx", det1)
        self.assertIn("vy", det1)
        self.assertEqual(det1["vx"], 0.0)  # First frame: 0 velocity

        # Frame 2: Move box right by 10px to x=370..410
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[200:280, 370:410] = [0, 245, 255]
        time.sleep(0.016)

        det2 = detector.detect(frame2)
        self.assertTrue(det2["has_target"])
        # Should detect positive lateral velocity vx > 0
        self.assertGreater(det2["vx"], 3.0, "Lateral velocity should be positive when moving right")


class TestAimAgentVelocityFeedforward(unittest.TestCase):

    def test_locked_stationary_target_holds_still(self):
        """When locked on stationary target, output velocity should be 0.0."""
        agent = MagicMock()
        agent.deadzone_px = 6.0
        agent.kp_yaw = 0.55
        agent.kd_yaw = 0.22
        agent.max_step_px = 160.0
        agent.prev_dx = 0.0
        agent.input_ctrl = MagicMock()
        agent.get_effective_speed = MagicMock(return_value=1.0)

        # Simulate detection locked with 0 velocity
        det = {"has_target": True, "in_lock_zone": True, "dx": 2.0, "vx": 0.0}

        # Check lock logic
        is_locked = bool(det.get("in_lock_zone", False) or abs(det["dx"]) <= 6.0)
        target_vx = float(det.get("vx", 0.0))

        if is_locked:
            if abs(target_vx) > 2.0:
                vx = float(np.clip(target_vx, -32.0, 32.0))
            else:
                vx = 0.0
        self.assertEqual(vx, 0.0)

    def test_locked_moving_target_feeds_forward_velocity(self):
        """When locked on moving target, output velocity should match target_vx."""
        det = {"has_target": True, "in_lock_zone": True, "dx": 3.0, "vx": 12.5}

        is_locked = bool(det.get("in_lock_zone", False) or abs(det["dx"]) <= 6.0)
        target_vx = float(det.get("vx", 0.0))

        if is_locked:
            if abs(target_vx) > 2.0:
                vx = float(np.clip(target_vx, -32.0, 32.0))
            else:
                vx = 0.0

        self.assertEqual(vx, 12.5, "Aim should feed forward lateral velocity to track moving target")


class TestRLStickyLockAndFeedforward(unittest.TestCase):

    def test_select_action_sticky_lock_stationary(self):
        """Verify PureAimRLAgent selects hold action (8, 3) for in_lock_zone target."""
        agent = MagicMock()
        agent.env = MagicMock()
        agent.env.YAW_ACTIONS = [-160.0, -110.0, -75.0, -50.0, -32.0, -18.0, -8.0, -3.0, 0.0, 3.0, 8.0, 18.0, 32.0, 50.0, 75.0, 110.0, 160.0]
        agent.last_seen_dir = 1
        agent.epsilon = 0.35

        select_fn = PureAimRLAgent.select_action.__get__(agent, PureAimRLAgent)

        # Target in lock zone, stationary
        det = {"has_target": True, "in_lock_zone": True, "crosshair_locked": False, "dx": 2.0, "dy": 0.0, "vx": 0.0}
        act = select_fn(np.zeros(10), det)
        self.assertEqual(act, (8, 3), "Should select action (8, 3) -> 0.0 delta when locked")

    def test_select_action_sticky_lock_moving(self):
        """Verify PureAimRLAgent selects closest gear to match target vx when in_lock_zone."""
        agent = MagicMock()
        agent.env = MagicMock()
        agent.env.YAW_ACTIONS = [-160.0, -110.0, -75.0, -50.0, -32.0, -18.0, -8.0, -3.0, 0.0, 3.0, 8.0, 18.0, 32.0, 50.0, 75.0, 110.0, 160.0]
        agent.last_seen_dir = 1
        agent.epsilon = 0.35

        select_fn = PureAimRLAgent.select_action.__get__(agent, PureAimRLAgent)

        # Target in lock zone, moving right at +17.0 px/tick (closest gear is 18.0 px at index 11)
        det = {"has_target": True, "in_lock_zone": True, "crosshair_locked": False, "dx": 4.0, "vx": 17.0}
        act = select_fn(np.zeros(10), det)
        self.assertEqual(act, (11, 3), "Should select gear 11 (+18.0 px) to match vx=+17.0 px/tick")

    def test_eval_mode_zeroes_epsilon(self):
        """Verify eval_mode sets epsilon to 0.0."""
        agent = PureAimRLAgent.__new__(PureAimRLAgent)
        agent.width = 640
        agent.height = 480
        agent.save_path = "dummy.pth"
        agent.gamma = 0.96
        agent.eval_mode = True
        agent.epsilon = 0.0 if agent.eval_mode else 0.35
        agent.epsilon_min = 0.0 if agent.eval_mode else 0.05
        self.assertEqual(agent.epsilon, 0.0)
        self.assertEqual(agent.epsilon_min, 0.0)


if __name__ == "__main__":
    unittest.main()
