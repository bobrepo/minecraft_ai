"""Visual Reward Engine for Minecraft PvP Reinforcement Learning.

Computes real-time scalar rewards based on:
1. Aim Alignment: Crosshair centered on target.
2. Optimal Spacing: Ideal ~3-block reach distance (height ~180-260px).
3. Evasive Dodging: Lateral circle-strafing (A / D) in combat.
4. Hit Detection via Red Hurt-Tint:
   - Normal Sweep Hit: +10.0
   - Critical Falling Hit: +25.0
   - Knockback Sprint Hit: +40.0
   - Whiff / Miss Swing Penalty: -2.0
"""

from typing import Any, Dict, Tuple
import numpy as np


class PvPRewardEngine:
    """Calculates reinforcement learning rewards from visual perception and player state."""

    def __init__(self, resolution: Tuple[int, int] = (640, 480)):
        self.width, self.height = resolution
        self.crosshair_x = self.width // 2
        self.crosshair_y = self.height // 2

        # State history
        self.jump_tick_counter = 999  # Ticks since last Space jump
        self.attack_tick_counter = 999
        self.hurt_cooldown_counter = 0
        self.last_target_height = 0.0

    def record_action(self, jump: bool = False, attack: bool = False):
        """Update jump and attack timing counters."""
        if jump:
            self.jump_tick_counter = 0
        else:
            self.jump_tick_counter += 1

        if attack:
            self.attack_tick_counter = 0
        else:
            self.attack_tick_counter += 1

        if self.hurt_cooldown_counter > 0:
            self.hurt_cooldown_counter -= 1

    def is_falling(self) -> bool:
        """Check if player is currently in falling phase of jump (ticks 5 to 12).

        In Minecraft (20 TPS), a jump lasts ~12 ticks.
        Ticks 0-4: Ascending.
        Ticks 5-11: Falling with downward velocity -> CRITICAL HIT WINDOW!
        """
        return 5 <= self.jump_tick_counter <= 11

    def detect_hurt_tint(self, frame: np.ndarray, det: Dict[str, Any]) -> bool:
        """Detect if the enemy entity is flashing red from taking damage."""
        if not det["has_target"]:
            return False

        tx = int(det["target_x"])
        ty = int(det["target_y"])
        bw = max(20, int(det["box_w"] * 0.7))
        bh = max(30, int(det["box_h"] * 0.6))

        x1 = max(0, tx - bw // 2)
        y1 = max(0, ty - bh // 2)
        x2 = min(self.width, tx + bw // 2)
        y2 = min(self.height, ty + bh // 2)

        if x2 <= x1 or y2 <= y1:
            return False

        roi = frame[y1:y2, x1:x2]
        r = roi[:, :, 2].astype(np.float32)
        g = roi[:, :, 1].astype(np.float32)
        b = roi[:, :, 0].astype(np.float32)

        # Hurt tint: Red channel spikes significantly above Green and Blue
        hurt_mask = (r > 150) & (r > (g + 45)) & (r > (b + 45))
        hurt_ratio = np.mean(hurt_mask)

        # > 15% of torso pixels flashing red indicates a confirmed damage strike
        return float(hurt_ratio) > 0.15

    def compute_reward(self, frame: np.ndarray, det: Dict[str, Any], actions: Dict[str, Any]) -> Dict[str, Any]:
        """Compute scalar reward and event breakdown for the current tick.

        Args:
            frame: Current 640x480 screen frame.
            det: VisionDetector dictionary.
            actions: Dictionary of current actions (w, s, a, d, sprint, jump, attack, dx, dy).

        Returns:
            Dictionary with total reward and individual reward components.
        """
        r_aim = 0.0
        r_dist = 0.0
        r_dodge = 0.0
        r_hit = 0.0
        hit_type = "none"

        if det["has_target"]:
            # 1. Aim Centering Reward
            # Peaks at +2.0 when crosshair is dead center; decays with distance
            dist_to_ch = np.hypot(det["dx"], det["dy"])
            r_aim = 2.0 * max(0.0, 1.0 - (dist_to_ch / 140.0))

            # 2. Optimal Spacing (~3 blocks ideal distance)
            # At 3 blocks, enemy height on 640x480 is ~180px to 260px
            target_h = det["box_h"]
            self.last_target_height = target_h

            if 170 <= target_h <= 270:
                r_dist = 2.0  # Sweet spot reach distance!
            elif target_h < 120:
                r_dist = -0.5  # Too far away
            elif target_h > 330:
                r_dist = -0.5  # Too close / crowded

            # 3. Evasive Dodging / Circle-Strafing
            # Reward lateral motion (A or D) while engaged in combat
            if det["in_attack_range"] and (actions.get("a") or actions.get("d")):
                r_dodge = 0.5

            # 4. Hit Detection via Red Hurt-Tint
            # Triggered if attack was executed within last 3 ticks and enemy flashes red
            enemy_damaged = self.detect_hurt_tint(frame, det)

            if enemy_damaged and self.hurt_cooldown_counter == 0 and self.attack_tick_counter <= 4:
                self.hurt_cooldown_counter = 8  # Ignore consecutive frames of same hurt flash

                if self.is_falling():
                    r_hit = 25.0  # Critical Falling Hit!
                    hit_type = "critical_hit"
                elif actions.get("sprint") and actions.get("w"):
                    r_hit = 40.0  # Knockback Sprint Hit!
                    hit_type = "knockback_hit"
                else:
                    r_hit = 10.0  # Normal Sweep Hit!
                    hit_type = "sweep_hit"

            # Whiff penalty: attacking when enemy is not within 3-block reach
            elif actions.get("attack") and not det["in_attack_range"]:
                r_hit = -2.0
                hit_type = "whiff"

        else:
            # Searching penalty for spinning when nothing is visible
            r_aim = -0.1
            if actions.get("attack"):
                r_hit = -2.0
                hit_type = "whiff"

        total_reward = float(r_aim + r_dist + r_dodge + r_hit)

        return {
            "reward": total_reward,
            "r_aim": float(r_aim),
            "r_dist": float(r_dist),
            "r_dodge": float(r_dodge),
            "r_hit": float(r_hit),
            "hit_type": hit_type,
            "is_falling": self.is_falling(),
        }
