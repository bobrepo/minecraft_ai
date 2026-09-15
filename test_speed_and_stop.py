"""Unit tests for Data Preservation, AI Aim Speed Control, and Instant Stop Feature.

Verifies:
1. Training data safe save and rolling backup preservation.
2. Immediate aim stop via SubTickMouseThread.halt() and input_ctrl.stop_aim().
3. AI dynamic distance-adaptive speed calculation in AimAgent.
4. Aim speed modes ("auto", "0.5x", "1.0x", "1.5x", "2.0x") and live mode switching.
5. MinecraftAimEnv speed multiplier scaling.
"""

import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch

from aim_agent import AimAgent
from aim_env import MinecraftAimEnv
from input_controller import InputController, SubTickMouseThread
from pure_aim_rl import BranchingDuelingQNet, PureAimRLAgent


class TestDataPreservation(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pth")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_safe_save_checkpoint_creates_atomic_file_and_backup(self):
        """Verify save_checkpoint performs atomic save and creates rolling backup."""
        # Create a mock agent with net
        agent = MagicMock()
        agent.save_path = self.model_path
        agent.train_lock = MagicMock()
        agent.train_lock.__enter__ = MagicMock()
        agent.train_lock.__exit__ = MagicMock()

        q_net = BranchingDuelingQNet(state_dim=10, yaw_dim=17, pitch_dim=7)
        agent.q_net = q_net

        # Bind save_checkpoint method
        save_fn = PureAimRLAgent.save_checkpoint.__get__(agent, PureAimRLAgent)

        # 1. First save: creates file
        save_fn(self.model_path)
        self.assertTrue(os.path.exists(self.model_path))
        state1 = torch.load(self.model_path, map_location="cpu")
        self.assertEqual(state1["yaw_head.2.weight"].shape, torch.Size([17, 64]))

        # 2. Second save: creates _prev_backup.pth and updates main file
        time.sleep(0.01)
        save_fn(self.model_path)
        backup_path = self.model_path.replace(".pth", "_prev_backup.pth")
        self.assertTrue(os.path.exists(backup_path), "Backup file should be created on subsequent save")
        self.assertEqual(os.path.getsize(backup_path), os.path.getsize(self.model_path))


class TestInstantAimStop(unittest.TestCase):

    def test_halt_immediately_zeroes_subtick_thread(self):
        """Verify halt() clears active status, velocities, and accumulators."""
        dispatched = []

        def mock_send(sx, sy):
            dispatched.append((sx, sy))

        thread = SubTickMouseThread(mock_send)
        thread.submit_aim(160.0, 0.0, duration_sec=0.1)

        self.assertTrue(thread.active)
        self.assertEqual(thread.total_dx, 160.0)

        # Call halt()
        thread.halt()
        self.assertFalse(thread.active)
        self.assertEqual(thread.total_dx, 0.0)
        self.assertEqual(thread.total_dy, 0.0)
        self.assertEqual(thread.accum_x, 0.0)
        self.assertEqual(thread.accum_y, 0.0)

    def test_input_ctrl_stop_aim(self):
        """Verify input_ctrl.stop_aim() flushes sub-tick thread."""
        ctrl = InputController()
        try:
            ctrl.move_mouse(100.0, 0.0, dynamic=True, duration_sec=0.2)
            ctrl.stop_aim()
            with ctrl.mouse_thread.lock:
                self.assertFalse(ctrl.mouse_thread.active)
                self.assertEqual(ctrl.mouse_thread.total_dx, 0.0)
        finally:
            ctrl.release_all(force=True)
            ctrl.mouse_thread.stop()


class TestAIAimSpeedControl(unittest.TestCase):

    def test_dynamic_ai_speed_scaling_curve(self):
        """Verify AI computes high speed for large distance and decelerates for close distances."""
        # Create Mock AimAgent
        agent = MagicMock()
        agent.base_speed = 1.0
        agent.speed_mode = "auto"

        compute_speed = AimAgent.compute_ai_speed_factor.__get__(agent, AimAgent)
        get_speed = AimAgent.get_effective_speed.__get__(agent, AimAgent)

        # Far distance: snap flick boost
        s_far = compute_speed(250.0)
        self.assertGreaterEqual(s_far, 1.5, "Far target should trigger snap flick speed boost")

        # Mid distance: swift tracking
        s_mid = compute_speed(80.0)
        self.assertEqual(s_mid, 1.0)

        # Close distance: controlled braking
        s_close = compute_speed(30.0)
        self.assertLess(s_close, 1.0)
        self.assertEqual(s_close, 0.60)

        # Lock zone: micro-glide precision
        s_lock = compute_speed(5.0)
        self.assertLess(s_lock, 0.50)
        self.assertEqual(s_lock, 0.35)

        # Monotonicity: farther distance should have >= speed than closer distance
        self.assertGreater(s_far, s_mid)
        self.assertGreater(s_mid, s_close)
        self.assertGreater(s_close, s_lock)

    def test_speed_mode_presets(self):
        """Verify speed presets (0.5x, 1.0x, 1.5x, 2.0x) override dynamic mode."""
        agent = MagicMock()
        agent.base_speed = 1.0
        agent.compute_ai_speed_factor = AimAgent.compute_ai_speed_factor.__get__(agent, AimAgent)
        get_speed = AimAgent.get_effective_speed.__get__(agent, AimAgent)

        for mode, expected in [("0.5x", 0.5), ("1.0x", 1.0), ("1.5x", 1.5), ("2.0x", 2.0)]:
            agent.speed_mode = mode
            self.assertEqual(get_speed(100.0), expected)


class TestEnvSpeedMultiplier(unittest.TestCase):

    def test_env_scales_action_by_speed_multiplier(self):
        """Verify MinecraftAimEnv correctly scales dispatched mouse deltas by speed_multiplier."""
        env = MagicMock()
        env.width = 640
        env.height = 480
        env.horizontal_only = True
        env.YAW_ACTIONS = [-160.0, -110.0, -75.0, -50.0, -32.0, -18.0, -8.0, -3.0, 0.0, 3.0, 8.0, 18.0, 32.0, 50.0, 75.0, 110.0, 160.0]
        env.PITCH_ACTIONS = [-18.0, -8.0, -2.5, 0.0, 2.5, 8.0, 18.0]
        env.current_step = 0
        env.consecutive_moving_ticks = 0
        env.speed_multiplier = 1.5  # 1.5x speed
        env.input_ctrl = MagicMock()
        env.cap = MagicMock()
        env.cap.get_frame.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        env.detector = MagicMock()
        env.detector.detect.return_value = {"has_target": False, "dist_px": 999.0}
        env._build_state = MagicMock(return_value=np.zeros(10))
        env.max_episode_steps = 200

        step_fn = MinecraftAimEnv.step.__get__(env, MinecraftAimEnv)

        # Action: yaw_idx 14 (+75.0 px)
        step_fn((14, 3))

        # Expected dispatched yaw = 75.0 * 1.5 = 112.5 px
        env.input_ctrl.move_mouse.assert_called_once()
        args, kwargs = env.input_ctrl.move_mouse.call_args
        dispatched_yaw = args[0]
        self.assertAlmostEqual(dispatched_yaw, 112.5, places=1)


if __name__ == "__main__":
    unittest.main()
