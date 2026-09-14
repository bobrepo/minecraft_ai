"""Visual Reward Engine & Mechanics Rubric for Minecraft PvP Reinforcement Learning.

Minecraft 1.9–1.21 Combat Mechanics & Pure Aim-Centric Reward Rubric:
====================================================================================================
Event / Action               | Condition                                 | RL Reward   | Description
-----------------------------+-------------------------------------------+-------------+--------------------------------------------
Dead-Center Crosshair Lock   | Target center <= 35px from crosshair      | +15.0 /tick | Pure Aim (HIGH): keeps crosshair locked on enemy
On-Body Aim Tracking         | Target distance 35px - 80px               | +8.0-+15.0  | Pure Aim (HIGH): smooth gradient tracking on body
In-Frame Pursuit Aim         | Target distance 80px - 180px              | +2.0-+8.0   | Pure Aim (HIGH): smooth guidance toward center
Predictive Target Intercept  | Re-acquires target after trajectory loss  |  +6.0 pts   | Pure Aim (HIGH): looking at & re-acquiring target
Predictive Search Guidance   | Steers toward predicted target location   |  +1.0 /tick | Pure Aim: encourages turning toward fleeing enemy
Off-Target Look Away Penalty | Distance > 180px                          |  -1.0 /tick | NEGATIVE: punishes looking away from enemy
Target Lost / Blind Search   | Enemy not in view                         |  -1.0 /tick | NEGATIVE: punishes losing sight of enemy
Spam Attack Penalty          | Attack when weapon cooldown < 85%         |  -2.0 pts   | NEGATIVE: penalizes spam-clicking
Anti-Bunny-Hop Jump Spam     | Repeated mid-air jump / uncharged jump    |  -2.0 pts   | NEGATIVE: penalizes jump spam
Whiff / Miss Swing Penalty   | Attack when target not in reach           |  -2.0 pts   | NEGATIVE: penalizes swinging at empty air
Overcrowded Spacing Penalty  | Box height > 320px                        |  -0.5 /tick | NEGATIVE: penalizes face-hugging inside hitbox
Too Far Spacing Penalty      | Box height < 120px                        |  -0.5 /tick | NEGATIVE: penalizes drifting away from combat
W-Tap / Strafe / Hits        | Any movement or hit events                |   0.0 pts   | NO POSITIVE REWARD (only looking is high)
====================================================================================================
"""

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np


class TargetTrajectoryPredictor:
    """Predicts target trajectory, maintains spatial memory map, and extrapolates positions when target leaves view."""

    def __init__(self, resolution: Tuple[int, int] = (640, 480)):
        self.width, self.height = resolution
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        self.vx = 0.0
        self.vy = 0.0
        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None
        self.last_seen_x: Optional[float] = None
        self.last_seen_y: Optional[float] = None
        self.last_seen_h: float = 0.0
        self.ticks_lost: int = 999
        self.confidence: float = 0.0
        self.predicted_pos: Optional[Tuple[float, float]] = None
        self.was_tracking: bool = False

    def update(self, det: Dict[str, Any]) -> Dict[str, Any]:
        if det.get("has_target", False):
            cur_x = float(det.get("target_x", self.cx + float(det.get("dx", 0.0))))
            cur_y = float(det.get("target_y", self.cy + float(det.get("dy", 0.0))))

            # Check if target was re-acquired along predicted trajectory
            reacquired_predicted = False
            if 0 < self.ticks_lost <= 20 and self.predicted_pos is not None:
                pred_dist = math.hypot(cur_x - self.predicted_pos[0], cur_y - self.predicted_pos[1])
                # Within 140px of predicted position indicates successful predictive intercept
                if pred_dist < 140.0:
                    reacquired_predicted = True

            # Calculate smoothed velocity with exponential moving average (EMA)
            if self.was_tracking and self.prev_x is not None and self.prev_y is not None:
                raw_vx = cur_x - self.prev_x
                raw_vy = cur_y - self.prev_y
                self.vx = 0.45 * raw_vx + 0.55 * self.vx
                self.vy = 0.45 * raw_vy + 0.55 * self.vy
            else:
                self.vx = 0.0
                self.vy = 0.0

            self.last_seen_x = cur_x
            self.last_seen_y = cur_y
            self.last_seen_h = float(det.get("box_h", 0.0))
            self.prev_x = cur_x
            self.prev_y = cur_y
            self.ticks_lost = 0
            self.confidence = 1.0
            self.was_tracking = True
            self.predicted_pos = (cur_x + self.vx, cur_y + self.vy)

            return {
                "is_predicting": False,
                "has_target": True,
                "reacquired_predicted": reacquired_predicted,
                "vx": self.vx,
                "vy": self.vy,
                "speed": float(math.hypot(self.vx, self.vy)),
                "pred_pos": (cur_x, cur_y),
                "pred_dx": float(det["dx"]),
                "pred_dy": float(det["dy"]),
                "confidence": 1.0,
                "ticks_lost": 0,
                "direction": self._get_direction_name(det["dx"], det["dy"]),
            }
        else:
            self.was_tracking = False
            self.ticks_lost += 1
            self.prev_x = None
            self.prev_y = None

            # Confidence decays linearly over 18 ticks (~0.90s)
            self.confidence = max(0.0, 1.0 - (self.ticks_lost / 18.0))

            if self.ticks_lost <= 18 and self.last_seen_x is not None and self.last_seen_y is not None:
                # Damped linear trajectory extrapolation
                damping = max(0.2, 1.0 - (self.ticks_lost * 0.04))
                extrap_x = self.last_seen_x + (self.vx * self.ticks_lost * damping)
                extrap_y = self.last_seen_y + (self.vy * self.ticks_lost * damping)
                self.predicted_pos = (extrap_x, extrap_y)
                pred_dx = extrap_x - self.cx
                pred_dy = extrap_y - self.cy

                return {
                    "is_predicting": True,
                    "has_target": False,
                    "reacquired_predicted": False,
                    "vx": self.vx,
                    "vy": self.vy,
                    "speed": float(math.hypot(self.vx, self.vy)),
                    "pred_pos": (extrap_x, extrap_y),
                    "pred_dx": float(pred_dx),
                    "pred_dy": float(pred_dy),
                    "confidence": float(self.confidence),
                    "ticks_lost": self.ticks_lost,
                    "direction": self._get_direction_name(pred_dx, pred_dy),
                }
            else:
                self.predicted_pos = None
                return {
                    "is_predicting": False,
                    "has_target": False,
                    "reacquired_predicted": False,
                    "vx": 0.0,
                    "vy": 0.0,
                    "speed": 0.0,
                    "pred_pos": None,
                    "pred_dx": 0.0,
                    "pred_dy": 0.0,
                    "confidence": 0.0,
                    "ticks_lost": self.ticks_lost,
                    "direction": "CENTER",
                }

    def get_current_state(self) -> Dict[str, Any]:
        """Return the current trajectory state without incrementing counters."""
        if self.ticks_lost == 0 and self.last_seen_x is not None and self.last_seen_y is not None:
            return {
                "is_predicting": False,
                "has_target": True,
                "vx": self.vx,
                "vy": self.vy,
                "confidence": 1.0,
                "ticks_lost": 0,
                "pred_dx": self.last_seen_x - self.cx,
                "pred_dy": self.last_seen_y - self.cy,
                "direction": self._get_direction_name(self.last_seen_x - self.cx, self.last_seen_y - self.cy),
            }
        elif self.ticks_lost <= 18 and self.predicted_pos is not None:
            pred_dx = self.predicted_pos[0] - self.cx
            pred_dy = self.predicted_pos[1] - self.cy
            return {
                "is_predicting": True,
                "has_target": False,
                "vx": self.vx,
                "vy": self.vy,
                "confidence": self.confidence,
                "ticks_lost": self.ticks_lost,
                "pred_dx": pred_dx,
                "pred_dy": pred_dy,
                "direction": self._get_direction_name(pred_dx, pred_dy),
            }
        else:
            return {
                "is_predicting": False,
                "has_target": False,
                "vx": 0.0,
                "vy": 0.0,
                "confidence": 0.0,
                "ticks_lost": self.ticks_lost,
                "pred_dx": 0.0,
                "pred_dy": 0.0,
                "direction": "CENTER",
            }

    def _get_direction_name(self, dx: float, dy: float) -> str:
        """Map screen offset vector to 8-cardinal direction string."""
        if abs(dx) < 18 and abs(dy) < 18:
            return "CENTER"
        ang = math.degrees(math.atan2(dy, dx))
        if -22.5 <= ang < 22.5:
            return "RIGHT"
        elif 22.5 <= ang < 67.5:
            return "DOWN-RIGHT"
        elif 67.5 <= ang < 112.5:
            return "DOWN"
        elif 112.5 <= ang < 157.5:
            return "DOWN-LEFT"
        elif ang >= 157.5 or ang < -157.5:
            return "LEFT"
        elif -157.5 <= ang < -112.5:
            return "UP-LEFT"
        elif -112.5 <= ang < -67.5:
            return "UP"
        elif -67.5 <= ang < -22.5:
            return "UP-RIGHT"
        return "CENTER"


class PvPRewardEngine:
    """Calculates aim-dominant reinforcement learning rewards with target trajectory prediction."""

    def __init__(self, resolution: Tuple[int, int] = (640, 480)):
        self.width, self.height = resolution
        self.crosshair_x = self.width // 2
        self.crosshair_y = self.height // 2

        # Target Trajectory Predictor & Spatial Memory Map
        self.predictor = TargetTrajectoryPredictor(resolution)

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
            "jump_spam": False,
            "charge": self.get_attack_cooldown_charge(),
        }

        # 1. Jump Physics Counter & Anti-Bunny-Hop Detection
        if jump:
            if self.jump_tick_counter < 12:
                # Bunny-hopping / spamming space while already airborne (<12 ticks)
                flags["jump_spam"] = True
            elif flags["charge"] < 0.85:
                # Jumping while weapon cooldown is still recharging (<85%)
                flags["jump_spam"] = True
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

        tx = int(det.get("target_x", self.width // 2))
        ty = int(det.get("target_y", self.height // 2))
        bw = max(20, int(det.get("box_w", 60) * 0.7))
        bh = max(30, int(det.get("box_h", 120) * 0.6))

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
        """Compute scalar reward with aim-dominant weighting and predictive trajectory tracking."""
        if action_flags is None:
            action_flags = {}

        # Update target trajectory predictor
        pred_info = self.predictor.update(det)

        r_aim = 0.0
        r_pred = 0.0
        r_dist = 0.0
        r_dodge = 0.0
        r_wtap = 0.0
        r_spam = 0.0
        r_dist_atk = 0.0
        r_hit = 0.0
        r_jump_spam = 0.0
        hit_type = "none"

        # 1. Anti-Spam Click Penalty (scaled to -2.0)
        if action_flags.get("spam_attack"):
            r_spam = -2.0
            hit_type = "spam_penalty"

        # Anti-Bunny-Hop Jump Spam Penalty (-2.0)
        if action_flags.get("jump_spam"):
            r_jump_spam = -2.0
            if hit_type == "none":
                hit_type = "jump_spam_penalty"

        if det["has_target"]:
            # 2. Dominant Aim Centering & Looking-at-Enemy Reward
            # Rewarded highly for looking directly at the enemy and keeping crosshair centered!
            dist_to_ch = np.hypot(det["dx"], det["dy"])
            if dist_to_ch <= 35:
                r_aim = 15.0  # Bullseye tracking: crosshair locked dead-center! (+300 pts/sec)
            elif dist_to_ch <= 80:
                # On target body: smooth interpolation between +15.0 and +8.0
                t = (dist_to_ch - 35.0) / 45.0
                r_aim = 15.0 - 7.0 * t
            elif dist_to_ch <= 180:
                # In-frame pursuit: smooth decay from +8.0 down to +2.0
                t = (dist_to_ch - 80.0) / 100.0
                r_aim = 8.0 - 6.0 * t
            else:
                # Off-target penalty: target visible but looking away (> 180px)
                r_aim = -1.0

            # 3. Predictive Target Re-Acquisition Bonus (+6.0 pts)
            # Awarded when enemy is re-acquired along extrapolated trajectory
            if pred_info.get("reacquired_predicted", False):
                r_pred = 6.0
                if hit_type == "none":
                    hit_type = "pred_acquisition"

            # 4. Spacing Penalty: NEGATIVE if face-hugging or too far (no positive spacing reward)
            target_h = det.get("box_h", 0.0)
            self.last_target_height = target_h
            if target_h > 320:
                r_dist = -0.5  # Negative penalty for face-hugging inside enemy hitbox
            elif 0 < target_h < 120:
                r_dist = -0.5  # Negative penalty for drifting too far out of combat
            else:
                r_dist = 0.0

            # 5. NO REWARD for strafe dodge, W-tap, or distance swing (strictly 0.0 per user request)
            r_dodge = 0.0
            r_wtap = 0.0
            r_dist_atk = 0.0

            # 6. Hit Detection for Badges & Audio/Visual Feedback (r_hit = 0.0; rewards strictly for looking)
            enemy_damaged = self.detect_hurt_tint(frame, det)

            if enemy_damaged and self.hurt_cooldown_counter == 0 and self.attack_tick_counter <= 4:
                self.hurt_cooldown_counter = 8  # Debounce hurt flash
                is_distance_hit = (150 <= target_h <= 220)

                if self.is_falling():
                    hit_type = "dist_critical_hit" if is_distance_hit else "critical_hit"
                elif actions.get("sprint") and actions.get("w"):
                    if self.sprint_reset_ready or self.consecutive_sprint_hits == 0:
                        hit_type = "dist_knockback_hit" if is_distance_hit else "knockback_hit"
                        self.sprint_reset_ready = False
                        self.consecutive_sprint_hits += 1
                    else:
                        hit_type = "dist_sweep_hit" if is_distance_hit else "sweep_hit"
                else:
                    hit_type = "dist_sweep_hit" if is_distance_hit else "sweep_hit"
                r_hit = 0.0

            elif actions.get("attack") and not det.get("in_attack_range", False):
                r_hit = -2.0  # NEGATIVE: Whiff penalty
                if hit_type == "none":
                    hit_type = "whiff"

        else:
            # Target NOT currently visible: NEGATIVE penalty for losing visual lock on enemy!
            if pred_info.get("is_predicting", False) and pred_info.get("confidence", 0.0) > 0.15:
                # Target recently lost: award predictive search guidance if steering along trajectory
                act_dx = float(actions.get("dx", 0.0))
                act_dy = float(actions.get("dy", 0.0))
                act_mag = float(np.hypot(act_dx, act_dy))

                if act_mag > 8.0:
                    pred_dx = float(pred_info["pred_dx"])
                    pred_dy = float(pred_info["pred_dy"])
                    dot = (act_dx * pred_dx + act_dy * pred_dy) / (act_mag * (math.hypot(pred_dx, pred_dy) + 1e-6))
                    if dot > 0.4:
                        # Agent is actively steering camera in direction of predicted target!
                        r_pred = 1.0 * pred_info["confidence"]
                        r_aim = 0.0
                        if hit_type == "none":
                            hit_type = "pred_tracking"
                    else:
                        # Steering away from predicted target: NEGATIVE penalty
                        r_aim = -1.0
                else:
                    # Inactive while target was recently fleeing: NEGATIVE penalty
                    r_aim = -1.0
            else:
                # Fully lost and spinning blindly: NEGATIVE penalty
                r_aim = -1.0

            if actions.get("attack"):
                r_hit = -2.0  # NEGATIVE: Whiff penalty
                if hit_type == "none":
                    hit_type = "whiff"

        total_reward = float(r_aim + r_pred + r_dist + r_dodge + r_wtap + r_spam + r_dist_atk + r_hit + r_jump_spam)

        return {
            "reward": total_reward,
            "r_aim": float(r_aim),
            "r_pred": float(r_pred),
            "r_dist": float(r_dist),
            "r_dodge": float(r_dodge),
            "r_wtap": float(r_wtap),
            "r_spam": float(r_spam),
            "r_dist_atk": float(r_dist_atk),
            "r_hit": float(r_hit),
            "r_jump_spam": float(r_jump_spam),
            "hit_type": hit_type,
            "is_falling": self.is_falling(),
            "sprint_reset_ready": self.sprint_reset_ready,
            "cooldown_charge": self.get_attack_cooldown_charge(),
            "prediction": pred_info,
        }
