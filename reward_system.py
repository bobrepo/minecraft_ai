"""Visual Reward Engine & Mechanics Rubric for Minecraft PvP Reinforcement Learning.

Minecraft 1.9–1.21 Combat Mechanics & RL Reward Rubric:
====================================================================================================
Event / Action               | Condition                                 | RL Reward   | Description
-----------------------------+-------------------------------------------+-------------+--------------------------------------------
Knockback (KB) Hit           | Sprint hit with reset ready (1st W hit)   | +40.0 pts   | Highest reward; initiates combo & pushes back
Max-Reach Distance Hit Bonus | Hit landed from 2.6-3.0 blocks distance   | +15.0 pts   | Out-spacing bonus added on top of any hit
Critical Hit                 | Hit landed while falling (ticks 5-11 post | +25.0 pts   | 150% damage + golden star particles
Sweep Hit                    | Grounded hit / consecutive sprint hit     | +10.0 pts   | Base damage sweep hit
W-Tap Sprint Reset           | Release W >=2 ticks then re-engage        |  +5.0 pts   | Resets sprint counter for subsequent KB hit
Distance Attack Swing        | Attack swing initiated at 2.6-3.0 blocks  |  +2.5 pts   | Reward for attacking with spacing discipline
Overcrowded Attack Penalty   | Attack swing while crowded (<1.5 blocks)  |  -2.0 pts   | Penalizes face-hugging inside enemy hitbox
Spam Attack Penalty          | Attack when weapon cooldown < 85%         |  -8.0 pts   | Heavy penalty for spam-clicking without timing
Whiff / Miss Swing           | Attack when target not in 3-block reach   |  -3.0 pts   | Penalizes swinging at empty air
Optimal 3-Block Spacing      | Enemy box height 170-270 px               |  +2.0 /tick | Ideal melee reach distance spacing
Spacing Violation            | Box height <120px or >320px               |  -0.5 /tick | Too far away or crowded inside enemy
Aim Alignment Tracking       | Crosshair within 140px radius of target   |0.0-+2.0/tick| Continuous crosshair tracking on enemy
Evasive Circle-Strafing      | Lateral movement (A or D) in combat       |  +0.5 /tick | Circle-strafing to dodge incoming attacks
====================================================================================================
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


class PvPRewardEngine:
    """Calculates reinforcement learning rewards from visual perception and player state."""

    def __init__(self, resolution: Tuple[int, int] = (640, 480)):
        self.width, self.height = resolution
        self.crosshair_x = self.width // 2
        self.crosshair_y = self.height // 2

        # Jump & Falling Physics Tracker (Minecraft 20 TPS physics: ~12 ticks jump cycle)
        self.jump_tick_counter: int = 999  # Ticks since Space jump

        # Weapon Cooldown Tracker (Sword: 1.6 hits/sec -> ~12.5 ticks recharge)
        self.attack_tick_counter: int = 999  # Ticks since last attack execution
        self.weapon_recharge_ticks: int = 12  # 12 ticks for sword full recharge (0.60s)

        # W-Tap Sprint Reset State Machine
        self.w_held_prev: bool = False
        self.w_release_ticks: int = 999
        self.sprint_reset_ready: bool = True  # Initial sprint starts with KB hit ready
        self.consecutive_sprint_hits: int = 0

        # Hurt-flash debounce
        self.hurt_cooldown_counter: int = 0
        self.last_target_height: float = 0.0

    def get_attack_cooldown_charge(self) -> float:
        """Return weapon attack recharge percentage in range [0.0, 1.0].

        Swords recharge in 12 ticks (0.60s). At >= 0.85 (>= 10 ticks),
        the hit deals full damage and knockback. Under 0.85, it is a spam click.
        """
        return min(1.0, self.attack_tick_counter / float(self.weapon_recharge_ticks))

    def record_action(self, w: bool = False, jump: bool = False, attack: bool = False) -> Dict[str, Any]:
        """Update physics counters and W-tap state machine for the current tick.

        Returns:
            Dictionary containing state flags:
            - 'w_tap_reset': True if player successfully executed a W-tap reset
            - 'spam_attack': True if player attacked prematurely (<85% cooldown)
            - 'charge': Current attack meter charge percentage [0.0, 1.0]
        """
        flags = {
            "w_tap_reset": False,
            "spam_attack": False,
            "charge": self.get_attack_cooldown_charge(),
        }

        # 1. Jump Physics Counter
        if jump:
            self.jump_tick_counter = 0
        else:
            self.jump_tick_counter += 1

        # 2. W-Tap Sprint Reset State Machine:
        # Releasing W for >= 2 ticks and pressing it again resets the sprint,
        # allowing the next attack to deliver a high-knockback hit!
        if w:
            if not self.w_held_prev and self.w_release_ticks >= 2:
                self.sprint_reset_ready = True
                self.consecutive_sprint_hits = 0
                flags["w_tap_reset"] = True
            self.w_held_prev = True
            self.w_release_ticks = 0
        else:
            self.w_held_prev = False
            self.w_release_ticks += 1
            if self.w_release_ticks >= 2:
                self.sprint_reset_ready = True
                self.consecutive_sprint_hits = 0

        # 3. Weapon Cooldown & Anti-Spam Check
        if attack:
            charge = self.get_attack_cooldown_charge()
            if charge < 0.85:
                # Premature attack (< 85% recharge) -> SPAM PENALTY!
                flags["spam_attack"] = True
            self.attack_tick_counter = 0
        else:
            self.attack_tick_counter += 1

        # 4. Hurt flash cooldown
        if self.hurt_cooldown_counter > 0:
            self.hurt_cooldown_counter -= 1

        flags["charge"] = self.get_attack_cooldown_charge()
        return flags

    def is_falling(self) -> bool:
        """Check if player is currently in falling phase of jump (ticks 5 to 11).

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

    def compute_reward(
        self,
        frame: np.ndarray,
        det: Dict[str, Any],
        actions: Dict[str, Any],
        action_flags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute scalar reward and detailed event breakdown for the current tick.

        Args:
            frame: Current 640x480 screen frame.
            det: VisionDetector dictionary.
            actions: Dictionary of current actions (w, s, a, d, sprint, jump, attack, dx, dy).
            action_flags: Optional dictionary returned by record_action().

        Returns:
            Dictionary with total reward and individual reward components.
        """
        if action_flags is None:
            action_flags = {}

        r_aim = 0.0
        r_dist = 0.0
        r_dodge = 0.0
        r_wtap = 0.0
        r_spam = 0.0
        r_dist_atk = 0.0
        r_hit = 0.0
        hit_type = "none"

        # 1. Anti-Spam Click Penalty
        if action_flags.get("spam_attack"):
            r_spam = -8.0
            hit_type = "spam_penalty"

        if det["has_target"]:
            # 2. Aim Centering Reward
            # Peaks at +2.0 when crosshair is dead center; decays with distance
            dist_to_ch = np.hypot(det["dx"], det["dy"])
            r_aim = 2.0 * max(0.0, 1.0 - (dist_to_ch / 140.0))

            # 3. Optimal Spacing (~3 blocks ideal distance)
            # At 3 blocks, enemy height on 640x480 is ~170px to 270px
            target_h = det["box_h"]
            self.last_target_height = target_h

            if 170 <= target_h <= 270:
                r_dist = 2.0  # Sweet spot reach distance!
            elif target_h < 120:
                r_dist = -0.5  # Too far away
            elif target_h > 320:
                r_dist = -0.5  # Too close / crowded

            # 4. Evasive Dodging / Circle-Strafing
            # Reward lateral motion (A or D) while engaged in combat
            if det["in_attack_range"] and (actions.get("a") or actions.get("d")):
                r_dodge = 0.5

            # 5. W-Tap Reset Reward: rewarding player for resetting sprint during combat
            if action_flags.get("w_tap_reset") and det["in_attack_range"]:
                r_wtap = 5.0

            # 6. Distance Attack Reward: reward swinging from safe maximum reach (~2.6 - 3.0 blocks)
            # In 640x480, height between 150px and 220px represents maximum melee reach
            if actions.get("attack") and det["in_attack_range"]:
                if 150 <= target_h <= 220:
                    r_dist_atk = 2.5  # Reward disciplined spacing when attacking
                elif target_h > 310:
                    r_dist_atk = -2.0  # Penalty for face-hugging / overcrowded attacks

            # 7. Hit Detection via Red Hurt-Tint
            # Triggered if attack was executed within last 4 ticks and enemy flashes red
            enemy_damaged = self.detect_hurt_tint(frame, det)

            if enemy_damaged and self.hurt_cooldown_counter == 0 and self.attack_tick_counter <= 4:
                self.hurt_cooldown_counter = 8  # Debounce consecutive frames of same flash

                # Check if hit was landed from maximum reach distance
                is_distance_hit = (150 <= target_h <= 220)
                dist_hit_bonus = 15.0 if is_distance_hit else 0.0

                if self.is_falling():
                    r_hit = 25.0 + dist_hit_bonus  # Critical Falling Hit (+ distance bonus)
                    hit_type = "dist_critical_hit" if is_distance_hit else "critical_hit"
                elif actions.get("sprint") and actions.get("w"):
                    if self.sprint_reset_ready or self.consecutive_sprint_hits == 0:
                        r_hit = 40.0 + dist_hit_bonus  # Knockback Sprint Hit (+ distance bonus)
                        hit_type = "dist_knockback_hit" if is_distance_hit else "knockback_hit"
                        self.sprint_reset_ready = False
                        self.consecutive_sprint_hits += 1
                    else:
                        r_hit = 10.0 + dist_hit_bonus  # Subsequent Sweep Hit (+ distance bonus)
                        hit_type = "dist_sweep_hit" if is_distance_hit else "sweep_hit"
                else:
                    r_hit = 10.0 + dist_hit_bonus  # Normal Sweep Hit (+ distance bonus)
                    hit_type = "dist_sweep_hit" if is_distance_hit else "sweep_hit"

            # Whiff penalty: attacking when enemy is not within 3-block reach
            elif actions.get("attack") and not det["in_attack_range"]:
                r_hit = -3.0
                if hit_type == "none":
                    hit_type = "whiff"

        else:
            # Searching penalty for spinning when nothing is visible
            r_aim = -0.1
            if actions.get("attack"):
                r_hit = -3.0
                if hit_type == "none":
                    hit_type = "whiff"

        total_reward = float(r_aim + r_dist + r_dodge + r_wtap + r_spam + r_dist_atk + r_hit)

        return {
            "reward": total_reward,
            "r_aim": float(r_aim),
            "r_dist": float(r_dist),
            "r_dodge": float(r_dodge),
            "r_wtap": float(r_wtap),
            "r_spam": float(r_spam),
            "r_dist_atk": float(r_dist_atk),
            "r_hit": float(r_hit),
            "hit_type": hit_type,
            "is_falling": self.is_falling(),
            "sprint_reset_ready": self.sprint_reset_ready,
            "cooldown_charge": self.get_attack_cooldown_charge(),
        }
