"""Unit tests for MinecraftAimEnv Reward System.

Verifies:
1. No points for sweeping / flicking past the target.
2. No points for continuously moving / wiggling across target.
3. No points for continuous slow drift across target.
4. No points during initial dwell period (< 10 ticks).
5. Full reward unlocked once crosshair dwells steadily (>= 10 ticks).
6. Immediate streak reset upon losing target.
7. Micro-adjustment handling (1 tick at <= 3.0px preserves streak, 0 points on moving tick).
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from aim_env import MinecraftAimEnv


class MockAimEnv(MinecraftAimEnv):
    """Headless MockAimEnv with isolated inputs and simulated detections."""

    def __init__(self, min_lock_ticks: int = 10):
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


def make_det(has_target=True, in_lock_zone=True, dx=0.0, dy=0.0, dist_px=0.0):
    return {
        "has_target": has_target,
        "target_x": 320.0 + dx,
        "target_y": 240.0 + dy,
        "dx": float(dx),
        "dy": float(dy),
        "dist_px": float(dist_px),
        "box_w": 40.0,
        "box_h": 60.0,
        "distance": 3.0,
        "in_lock_zone": in_lock_zone,
        "crosshair_locked": in_lock_zone,
    }


class TestRewardSystem(unittest.TestCase):

    def setUp(self):
        self.env = MockAimEnv(min_lock_ticks=10)
        self.env.reset()

    def test_sweep_across_target_gets_zero_reward(self):
        """A fast flick (e.g. +75px) across target must yield 0 reward and streak 0."""
        # yaw_idx 14 = +75.0 px/tick (flick)
        self.env.detector.detect.return_value = make_det(has_target=True, in_lock_zone=True)

        _, reward, _, info = self.env.step((14, 3))
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["lock_streak"], 0)
        self.assertTrue(info["is_continuously_moving"])

    def test_oscillation_across_target_gets_zero_reward(self):
        """Wiggling back and forth across target (+18, -18, +18) must yield 0 reward."""
        self.env.detector.detect.return_value = make_det(has_target=True, in_lock_zone=True)

        actions = [(11, 3), (5, 3), (11, 3), (5, 3)]  # +18, -18, +18, -18
        for act in actions:
            _, reward, _, info = self.env.step(act)
            self.assertEqual(reward, 0.0)
            self.assertEqual(info["lock_streak"], 0)

    def test_continuous_slow_drift_gets_zero_reward(self):
        """Drifting (+3px) every tick should trigger continuous move on tick 2 and reset streak."""
        self.env.detector.detect.return_value = make_det(has_target=True, in_lock_zone=True)

        # Tick 1: 1 tick move -> moving, reward 0.0
        _, r1, _, info1 = self.env.step((9, 3))  # +3.0 px
        self.assertEqual(r1, 0.0)
        self.assertEqual(info1["lock_streak"], 0)

        # Tick 2: 2nd consecutive move -> is_continuously_moving = True
        _, r2, _, info2 = self.env.step((9, 3))  # +3.0 px
        self.assertEqual(r2, 0.0)
        self.assertEqual(info2["lock_streak"], 0)
        self.assertTrue(info2["is_continuously_moving"])

    def test_dwell_time_requirement(self):
        """Holding stationary on target gives 0 reward for ticks 1-9, then awards points at tick 10+."""
        self.env.detector.detect.return_value = make_det(has_target=True, in_lock_zone=True)

        # Ticks 1 to 9: holding still (yaw_idx 8 = 0.0 px)
        for tick in range(1, 10):
            _, reward, _, info = self.env.step((8, 3))
            self.assertEqual(reward, 0.0, f"Expected 0.0 reward on tick {tick} before dwell satisfied")
            self.assertEqual(info["lock_streak"], tick)

        # Tick 10: Dwell threshold reached!
        _, r10, _, info10 = self.env.step((8, 3))
        self.assertEqual(info10["lock_streak"], 10)
        self.assertAlmostEqual(r10, 8.0, places=2)

        # Tick 11: Streak bonus kicks in (+0.20 per tick)
        _, r11, _, info11 = self.env.step((8, 3))
        self.assertEqual(info11["lock_streak"], 11)
        self.assertAlmostEqual(r11, 8.2, places=2)

    def test_instant_streak_reset_on_target_loss(self):
        """Losing target after acquiring high streak must immediately reset lock_streak to 0."""
        self.env.detector.detect.return_value = make_det(has_target=True, in_lock_zone=True)

        # Build streak past threshold
        for _ in range(12):
            self.env.step((8, 3))
        self.assertEqual(self.env.lock_streak, 12)

        # Lose lock
        self.env.detector.detect.return_value = make_det(has_target=True, in_lock_zone=False, dx=30.0)
        _, reward, _, info = self.env.step((8, 3))
        self.assertEqual(info["lock_streak"], 0)
        self.assertLessEqual(reward, 0.0)

    def test_micro_adjustment_handling(self):
        """A 1-tick 3px adjustment gives 0 reward while moving, but preserves streak if stationary next tick."""
        self.env.detector.detect.return_value = make_det(has_target=True, in_lock_zone=True)

        # Dwell for 12 ticks
        for _ in range(12):
            self.env.step((8, 3))
        self.assertEqual(self.env.lock_streak, 12)

        # 1-tick micro-adjustment (+3.0px, yaw_idx 9) while target remains in lock zone
        _, r_move, _, info_move = self.env.step((9, 3))
        self.assertEqual(r_move, 0.0, "Moving tick must have 0 reward")
        self.assertEqual(info_move["lock_streak"], 12, "1-tick micro-adjustment should not reset streak")

        # Next tick: hold stationary again
        _, r_hold, _, info_hold = self.env.step((8, 3))
        self.assertEqual(info_hold["lock_streak"], 13)
        self.assertAlmostEqual(r_hold, 8.6, places=2)


class TestAimAgentRewardLogic(unittest.TestCase):

    def test_aim_agent_reward_parity(self):
        """Simulate AimAgent dwell reward and continuous moving suppression logic."""
        min_lock_ticks = 10
        lock_streak = 0
        consecutive_moving_ticks = 0
        cumulative_score = 0.0

        def process_tick(vx: float, in_lock: bool):
            nonlocal lock_streak, consecutive_moving_ticks, cumulative_score
            is_moving = abs(vx) > 0.0
            if is_moving:
                consecutive_moving_ticks += 1
            else:
                consecutive_moving_ticks = 0

            is_continuously_moving = (consecutive_moving_ticks >= 2) or (abs(vx) > 3.0)

            if in_lock:
                if is_continuously_moving:
                    lock_streak = 0
                    step_reward = 0.0
                elif is_moving:
                    step_reward = 0.0
                else:
                    lock_streak += 1
                    if lock_streak >= min_lock_ticks:
                        r_lock = 8.0
                        r_streak = min(4.0, (lock_streak - min_lock_ticks) * 0.20)
                        step_reward = float(r_lock + r_streak)
                    else:
                        step_reward = 0.0
            else:
                lock_streak = 0
                step_reward = 0.0

            cumulative_score += step_reward
            return step_reward, lock_streak

        # 1. Sweeping fast through target: 0.0 reward, streak 0
        r, s = process_tick(vx=25.0, in_lock=True)
        self.assertEqual(r, 0.0)
        self.assertEqual(s, 0)

        # 2. Continuous wiggling: 0.0 reward, streak 0
        r, s = process_tick(vx=-10.0, in_lock=True)
        self.assertEqual(r, 0.0)
        self.assertEqual(s, 0)

        # 3. Holding still on target: 0 reward for first 9 ticks
        for i in range(1, 10):
            r, s = process_tick(vx=0.0, in_lock=True)
            self.assertEqual(r, 0.0)
            self.assertEqual(s, i)

        # 4. Tick 10: dwell threshold met!
        r, s = process_tick(vx=0.0, in_lock=True)
        self.assertEqual(r, 8.0)
        self.assertEqual(s, 10)


if __name__ == "__main__":
    unittest.main()
