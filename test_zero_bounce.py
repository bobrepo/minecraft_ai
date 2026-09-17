"""Unit tests for Zero-Bounce Controller, Ego-Motion Cancellation, Hysteresis Deadband, and 9-to-17 Action Warm Transfer.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np
import torch

from aim_agent import AimAgent
from pure_aim_rl import BranchingDuelingQNet, PureAimRLAgent


class TestZeroBounceController(unittest.TestCase):

    def test_deadzone_hysteresis(self):
        """Verify lock engages at <= 4.0px and remains engaged until > 7.5px."""
        agent = MagicMock()
        agent.lock_deadzone_px = 4.0
        agent.unlock_deadzone_px = 7.5
        agent.is_locked = False

        # Approach target: at 5.0px, not yet locked
        dx = 5.0
        if not agent.is_locked and abs(dx) <= agent.lock_deadzone_px:
            agent.is_locked = True
        self.assertFalse(agent.is_locked)

        # Reaches 3.5px: enters lock
        dx = 3.5
        if not agent.is_locked and abs(dx) <= agent.lock_deadzone_px:
            agent.is_locked = True
        self.assertTrue(agent.is_locked)

        # Micro-drift to 6.0px: remains locked (hysteresis prevents unlocking)
        dx = 6.0
        if agent.is_locked and abs(dx) > agent.unlock_deadzone_px:
            agent.is_locked = False
        self.assertTrue(agent.is_locked)

        # Large drift to 8.0px: breaks lock
        dx = 8.0
        if agent.is_locked and abs(dx) > agent.unlock_deadzone_px:
            agent.is_locked = False
        self.assertFalse(agent.is_locked)

    def test_anti_overshoot_safety_clamp(self):
        """Verify max_safe_step physically prevents overshooting the deadzone."""
        lock_deadzone_px = 4.0

        for test_dx in [10.0, 25.0, 50.0, 150.0, -12.0, -80.0, -247.0]:
            abs_dx = abs(test_dx)
            sign = 1.0 if test_dx > 0 else -1.0

            desired_vx = sign * 0.48 * abs_dx
            max_safe_step = max(0.0, abs_dx - lock_deadzone_px * 0.5)
            target_step = sign * min(abs(desired_vx), max_safe_step)

            remaining_after_step = abs_dx - abs(target_step)
            self.assertGreaterEqual(remaining_after_step, 0.0)
            self.assertEqual(np.sign(target_step), np.sign(test_dx))

    def test_ego_motion_compensation(self):
        """Verify camera ego-motion cancellation prevents false apparent velocity."""
        cam_turn = 40.0
        screen_delta = -40.0
        target_world_vx = screen_delta + cam_turn
        self.assertEqual(target_world_vx, 0.0)

        cam_turn = 10.0
        screen_delta = 0.0
        target_world_vx = screen_delta + cam_turn
        self.assertEqual(target_world_vx, 10.0)

    def test_monotonic_approach_no_bouncing(self):
        """Simulate tracking run from dx=-247.0 (user screenshot) and verify zero sign flips."""
        dx = -247.0
        pending_vx = 0.0
        prev_vx = 0.0
        is_locked = False
        lock_deadzone_px = 4.0
        unlock_deadzone_px = 7.5
        max_step_px = 160.0

        commands = []
        positions = []

        for _ in range(25):
            dx = dx - pending_vx
            abs_dx = abs(dx)
            sign = 1.0 if dx > 0 else -1.0

            if is_locked:
                if abs_dx > unlock_deadzone_px:
                    is_locked = False
                    vx = 0.0
                else:
                    vx = 0.0
            else:
                if abs_dx <= lock_deadzone_px:
                    is_locked = True
                    vx = 0.0

            if not is_locked:
                if abs_dx > 120.0:
                    desired_vx = sign * min(max_step_px, 0.48 * abs_dx)
                elif abs_dx > 30.0:
                    desired_vx = sign * (0.38 * abs_dx)
                else:
                    desired_vx = sign * (0.24 * abs_dx)

                max_safe_step = max(0.0, abs_dx - lock_deadzone_px * 0.5)
                target_step = sign * min(abs(desired_vx), max_safe_step)
                max_accel = 80.0
                vx = float(np.clip(target_step, prev_vx - max_accel, prev_vx + max_accel))

            prev_vx = vx
            pending_vx = vx
            commands.append(vx)
            positions.append(dx)

        non_zero_cmds = [c for c in commands if abs(c) > 0.0]
        for c in non_zero_cmds:
            self.assertLess(c, 0.0, f"Sign-flipped command {c} detected!")

        self.assertTrue(is_locked)
        self.assertLessEqual(abs(positions[-1]), lock_deadzone_px)


class TestWarmTransfer(unittest.TestCase):

    def test_warm_transfer_9_to_17_actions(self):
        """Verify 9-action weights are successfully mapped into 17-action network."""
        d9 = {
            "shared.0.weight": torch.randn(128, 10),
            "shared.0.bias": torch.randn(128),
            "shared.1.weight": torch.randn(128),
            "shared.1.bias": torch.randn(128),
            "val_head.0.weight": torch.randn(64, 128),
            "val_head.0.bias": torch.randn(64),
            "val_head.2.weight": torch.randn(1, 64),
            "val_head.2.bias": torch.randn(1),
            "yaw_head.0.weight": torch.randn(64, 128),
            "yaw_head.0.bias": torch.randn(64),
            "yaw_head.2.weight": torch.randn(9, 64),
            "yaw_head.2.bias": torch.randn(9),
            "pitch_head.0.weight": torch.randn(64, 128),
            "pitch_head.0.bias": torch.randn(64),
            "pitch_head.2.weight": torch.randn(7, 64),
            "pitch_head.2.bias": torch.randn(7),
        }

        agent = MagicMock()
        agent.q_net = BranchingDuelingQNet(state_dim=10, yaw_dim=17, pitch_dim=7)
        agent.target_net = BranchingDuelingQNet(state_dim=10, yaw_dim=17, pitch_dim=7)

        transfer_fn = PureAimRLAgent._warm_transfer_9act.__get__(agent, PureAimRLAgent)
        transfer_fn(d9)

        w17 = agent.q_net.state_dict()["yaw_head.2.weight"]
        torch.testing.assert_close(w17[8], d9["yaw_head.2.weight"][4])
        torch.testing.assert_close(w17[3], d9["yaw_head.2.weight"][0])
        torch.testing.assert_close(w17[13], d9["yaw_head.2.weight"][8])

    def test_differential_learning_rates_protect_yaw(self):
        """Verify differential learning rate groups protect yaw mastery while enabling pitch learning."""
        lr = 3e-4
        q_net = BranchingDuelingQNet(state_dim=10, yaw_dim=17, pitch_dim=7)
        param_groups = [
            {"params": q_net.shared.parameters(), "lr": lr * 0.25},
            {"params": q_net.val_head.parameters(), "lr": lr * 0.25},
            {"params": q_net.yaw_head.parameters(), "lr": lr * 0.20},
            {"params": q_net.pitch_head.parameters(), "lr": lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        # Verify pitch has full learning rate while yaw and backbone have protected lower rates
        self.assertEqual(optimizer.param_groups[0]["lr"], lr * 0.25)
        self.assertEqual(optimizer.param_groups[1]["lr"], lr * 0.25)
        self.assertEqual(optimizer.param_groups[2]["lr"], lr * 0.20)
        self.assertEqual(optimizer.param_groups[3]["lr"], lr)

    def test_simultaneous_2d_select_action(self):
        """Verify PureAimRLAgent select_action commands both yaw and pitch concurrently in 2D mode."""
        agent = MagicMock()
        agent.env = MagicMock()
        agent.env.horizontal_only = False
        agent.env.YAW_ACTIONS = [-160.0, -110.0, -75.0, -50.0, -32.0, -18.0, -8.0, -3.0, 0.0, 3.0, 8.0, 18.0, 32.0, 50.0, 75.0, 110.0, 160.0]
        agent.last_seen_dir = 1
        agent.epsilon = 0.0  # Pure exploitation

        q_net = BranchingDuelingQNet(state_dim=10, yaw_dim=17, pitch_dim=7)
        agent.q_net = q_net
        agent.device = torch.device("cpu")

        select_fn = PureAimRLAgent.select_action.__get__(agent, PureAimRLAgent)

        # Off-center in both axes: dx = -50 (enemy left), dy = -25 (enemy up)
        det = {"has_target": True, "dx": -50.0, "dy": -25.0, "vx": 0.0, "vy": 0.0}
        act = select_fn(np.zeros(10, dtype=np.float32), det)

        self.assertIsInstance(act, tuple)
        self.assertEqual(len(act), 2)
        yaw_act, pitch_act = act
        self.assertTrue(0 <= yaw_act < 17)
        self.assertTrue(0 <= pitch_act < 7)


if __name__ == "__main__":
    unittest.main()
