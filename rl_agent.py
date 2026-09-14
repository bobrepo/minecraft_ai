"""Branching Dueling Deep Q-Network (BDQ) Reinforcement Learning Agent for Minecraft PvP.

Features:
1. Multi-branch action selection (Aim, Movement, Jump, Attack).
2. Live Experience Replay Buffer and continuous GPU learning (AdamW).
3. Real-time visual reward feedback (+10 sweep, +25 crit, +40 knockback, aim centering, 3-block spacing).
4. Auto-detects Minecraft without terminal false positives.
5. Global F6 emergency toggle hotkey.
"""

import argparse
import collections
import ctypes
import os
import random
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from input_controller import EmergencyKillswitchListener, InputController
from overlay import PvPOverlayClient
from reward_system import PvPRewardEngine
from vision_detector import VisionDetector
from window_capture import WindowCapture, is_minecraft_window, list_windows

# Virtual Key Codes
VK_F6 = 0x75


def is_key_pressed(vk_code: int) -> bool:
    """Check if physical key is pressed globally via Windows API."""
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)


def auto_detect_minecraft() -> Optional[int]:
    """Find the real Minecraft window, strictly excluding terminal or browser tabs."""
    windows = list_windows()
    for hwnd, title in windows:
        if is_minecraft_window(hwnd, title):
            return hwnd
    return None


class BranchingQNetwork(nn.Module):
    """Branching Dueling Q-Network for simultaneous Aim, Move, Jump, and Attack control."""

    def __init__(self):
        super().__init__()

        # Convolutional Visual Encoder
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
        )

        latent_dim = 64 * 6 * 6
        self.shared_fc = nn.Sequential(
            nn.Linear(latent_dim, 384),
            nn.ReLU(inplace=True),
        )

        # State Value Stream V(s)
        self.val_head = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Action Advantage Streams A(s, a)
        # Branch 1: Aim [0: None, 1: Fast Left, 2: Soft Left, 3: Soft Right, 4: Fast Right, 5: Soft Up, 6: Soft Down]
        self.aim_adv = nn.Sequential(nn.Linear(384, 128), nn.ReLU(inplace=True), nn.Linear(128, 7))
        # Branch 2: Movement [0: Idle, 1: Walk W, 2: Sprint W, 3: Strafe A, 4: Strafe D, 5: Back S]
        self.move_adv = nn.Sequential(nn.Linear(384, 128), nn.ReLU(inplace=True), nn.Linear(128, 6))
        # Branch 3: Jump [0: Grounded, 1: Jump (Space)]
        self.jump_adv = nn.Sequential(nn.Linear(384, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))
        # Branch 4: Attack [0: Hold, 1: Attack Punch (Left Click)]
        self.atk_adv = nn.Sequential(nn.Linear(384, 64), nn.ReLU(inplace=True), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor):
        feat = self.conv(x)
        shared = self.shared_fc(feat)
        val = self.val_head(shared)

        # Compute dueling Q-values: Q(s, a) = V(s) + (A(s, a) - mean(A))
        aim_a = self.aim_adv(shared)
        aim_q = val + (aim_a - aim_a.mean(dim=1, keepdim=True))

        move_a = self.move_adv(shared)
        move_q = val + (move_a - move_a.mean(dim=1, keepdim=True))

        jump_a = self.jump_adv(shared)
        jump_q = val + (jump_a - jump_a.mean(dim=1, keepdim=True))

        atk_a = self.atk_adv(shared)
        atk_q = val + (atk_a - atk_a.mean(dim=1, keepdim=True))

        return aim_q, move_q, jump_q, atk_q


class ReplayBuffer:
    """Experience replay buffer for off-policy Q-learning."""

    def __init__(self, capacity: int = 8000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action_tuple, reward, next_state, done):
        self.buffer.append((state, action_tuple, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class AsyncTrainer:
    """Asynchronous background worker that continuously optimizes Q-Network on GPU.

    Decouples heavy GPU backpropagation (~40ms) from the 20 TPS combat loop,
    ensuring combat execution remains locked at 50ms per tick with zero lag.
    """

    def __init__(self, agent: "RLPvpAgent", batch_size: int = 32, interval_sec: float = 0.05):
        self.agent = agent
        self.batch_size = batch_size
        self.interval_sec = interval_sec
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while self._running:
            try:
                if self.agent.is_active and len(self.agent.replay_buffer) >= self.batch_size * 2:
                    self.agent.train_step(batch_size=self.batch_size)
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)


class RLPvpAgent:
    """Reinforcement Learning Combat Agent running at 20 TPS."""

    # Discrete Action Mappings
    AIM_DELTAS = [
        (0.0, 0.0),    # 0: None
        (-60.0, 0.0),  # 1: Fast Left
        (-18.0, 0.0),  # 2: Soft Left
        (18.0, 0.0),   # 3: Soft Right
        (60.0, 0.0),   # 4: Fast Right
        (0.0, -16.0),  # 5: Soft Up
        (0.0, 16.0),   # 6: Soft Down
    ]

    def __init__(
        self,
        target_hwnd: Optional[int] = None,
        resolution: Tuple[int, int] = (640, 480),
        epsilon_start: float = 0.30,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        gamma: float = 0.95,
        lr: float = 2e-4,
        save_path: str = "models/rl_pvp_model.pth",
        use_overlay: bool = True,
        show_cv_hud: bool = False,
    ):
        self.width, self.height = resolution
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.save_path = save_path
        self.use_overlay = use_overlay
        self.show_cv_hud = show_cv_hud

        # Hardware & Perception
        self.input_ctrl = InputController()
        self.detector = VisionDetector(resolution)
        self.reward_engine = PvPRewardEngine(resolution)

        # PyTorch Networks & Optimization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[+] Initializing Branching Q-Network on {self.device}...", flush=True)
        self.q_net = BranchingQNetwork().to(self.device)
        self.target_net = BranchingQNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        if os.path.exists(save_path):
            try:
                self.q_net.load_state_dict(torch.load(save_path, map_location=self.device))
                self.target_net.load_state_dict(self.q_net.state_dict())
                print(f"[+] Loaded existing RL weights from: {save_path}", flush=True)
            except Exception as e:
                print(f"[!] Could not load weights ({e}), starting fresh.", flush=True)

        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.replay_buffer = ReplayBuffer(capacity=8000)

        # Target window
        if target_hwnd is None:
            target_hwnd = auto_detect_minecraft()
            if target_hwnd is None:
                print("[!] Could not auto-detect Minecraft. Ensure Minecraft is running!", flush=True)
                sys.exit(1)

        self.cap = WindowCapture(target_hwnd)
        print(f"[+] Hooked to Minecraft: '{self.cap.window_title}' (HWND: {self.cap.hwnd})", flush=True)

        # Agent state & Emergency Killswitch Listener
        self.killswitch = EmergencyKillswitchListener(self.input_ctrl)
        self.prev_state_tensor: Optional[np.ndarray] = None
        self.prev_action_tuple: Optional[Tuple[int, int, int, int]] = None
        self.cumulative_reward = 0.0
        self.step_count = 0
        self.last_loss = 0.0

        # Asynchronous background GPU trainer (prevents 40ms blocking spikes in combat loop)
        self.async_trainer = AsyncTrainer(self, batch_size=32, interval_sec=0.05)

        # Desktop PvP Keystrokes & Mousepad Overlay
        self.overlay: Optional[PvPOverlayClient] = None
        if self.use_overlay:
            self.overlay = PvPOverlayClient()
            self.overlay.start()

    @property
    def is_active(self) -> bool:
        return self.killswitch.is_active

    @is_active.setter
    def is_active(self, val: bool):
        self.killswitch.set_active(val)

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Convert BGR frame to compact (3, 240, 320) float32 representation for RL buffer."""
        resized = cv2.resize(frame_bgr, (320, 240), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return (np.transpose(rgb, (2, 0, 1)) / 255.0).astype(np.float32)

    def select_action(self, state_arr: np.ndarray, det: Dict) -> Tuple[int, int, int, int]:
        """Epsilon-greedy multi-branch action selection with vision-guided heuristics."""
        # Exploration
        if random.random() < self.epsilon:
            if det["has_target"] and random.random() < 0.65:
                # 1. Vision-guided aiming heuristic
                dx = det["dx"]
                aim_act = 1 if dx < -50 else (2 if dx < -10 else (3 if dx > 10 else (4 if dx > 50 else 0)))

                # 2. Movement heuristic with W-Tap sprint reset and distance spacing:
                target_h = det.get("box_h", 0.0)
                if det["in_attack_range"] and self.reward_engine.consecutive_sprint_hits >= 1 and random.random() < 0.45:
                    move_act = 0  # Release W briefly -> resets sprint counter for another +40 KB hit!
                elif target_h > 290 and random.random() < 0.40:
                    # Too close (< 1.5 blocks)! Back up or circle-strafe to regain distance
                    move_act = 5 if random.random() < 0.5 else 3  # Back S or Strafe A
                elif self.reward_engine.sprint_reset_ready:
                    move_act = 2  # Sprint W to deliver high-knockback hit!
                else:
                    move_act = 2 if random.random() < 0.70 else 3  # Sprint or circle-strafe

                # 3. Disciplined Jump Heuristic for Critical Hits:
                # Anti-bunny-hop: ONLY jump when fully grounded (jump_tick_counter >= 12)
                # and when weapon cooldown is charged (charge >= 0.85) to time the critical strike!
                charge = self.reward_engine.get_attack_cooldown_charge()
                is_grounded = self.reward_engine.jump_tick_counter >= 12
                jump_act = 1 if (det["in_attack_range"] and charge >= 0.85 and is_grounded and random.random() < 0.25) else 0

                # 4. Anti-spam & Distance attack heuristic: Only attack when weapon cooldown is >= 85%!
                atk_act = 1 if (det["in_attack_range"] and charge >= 0.85) else 0

                return aim_act, move_act, jump_act, atk_act
            else:
                return (
                    random.randint(0, 6),
                    random.randint(0, 5),
                    random.randint(0, 1),
                    random.randint(0, 1),
                )

        # Exploitation via Q-Network
        with torch.no_grad():
            state_t = torch.from_numpy(state_arr).unsqueeze(0).to(self.device)
            aim_q, move_q, jump_q, atk_q = self.q_net(state_t)
            aim_act = int(torch.argmax(aim_q[0]).item())
            move_act = int(torch.argmax(move_q[0]).item())
            jump_act = int(torch.argmax(jump_q[0]).item())
            atk_act = int(torch.argmax(atk_q[0]).item())
            return aim_act, move_act, jump_act, atk_act

    def dispatch_action(self, action_tuple: Tuple[int, int, int, int]):
        """Execute chosen action tuple through DirectInput."""
        aim_act, move_act, jump_act, atk_act = action_tuple

        # 1. Aim
        dx, dy = self.AIM_DELTAS[aim_act]
        if dx != 0 or dy != 0:
            self.input_ctrl.move_mouse(int(dx), int(dy))

        # 2. Movement [0: Idle, 1: W, 2: Sprint W, 3: A, 4: D, 5: S]
        w = move_act in [1, 2]
        sprint = (move_act == 2)
        a = (move_act == 3)
        d = (move_act == 4)
        s = (move_act == 5)
        jump = (jump_act == 1)

        self.input_ctrl.set_movement(w=w, s=s, a=a, d=d, sprint=sprint, jump=jump)

        # 3. Attack
        if atk_act == 1:
            self.input_ctrl.attack_click()

        return {"w": w, "s": s, "a": a, "d": d, "sprint": sprint, "jump": jump, "attack": (atk_act == 1), "dx": dx, "dy": dy}

    def train_step(self, batch_size: int = 32):
        """Sample batch from replay buffer and optimize Q-network."""
        if len(self.replay_buffer) < batch_size * 2:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states_t = torch.from_numpy(states).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        rewards_t = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        dones_t = torch.from_numpy(dones).unsqueeze(1).to(self.device)

        # Current Q-values
        aim_q, move_q, jump_q, atk_q = self.q_net(states_t)

        aim_acts = torch.tensor(actions[:, 0], dtype=torch.long, device=self.device).unsqueeze(1)
        move_acts = torch.tensor(actions[:, 1], dtype=torch.long, device=self.device).unsqueeze(1)
        jump_acts = torch.tensor(actions[:, 2], dtype=torch.long, device=self.device).unsqueeze(1)
        atk_acts = torch.tensor(actions[:, 3], dtype=torch.long, device=self.device).unsqueeze(1)

        cur_aim_q = aim_q.gather(1, aim_acts)
        cur_move_q = move_q.gather(1, move_acts)
        cur_jump_q = jump_q.gather(1, jump_acts)
        cur_atk_q = atk_q.gather(1, atk_acts)

        # Target Q-values via Target Network
        with torch.no_grad():
            t_aim_q, t_move_q, t_jump_q, t_atk_q = self.target_net(next_states_t)
            max_aim = t_aim_q.max(dim=1, keepdim=True)[0]
            max_move = t_move_q.max(dim=1, keepdim=True)[0]
            max_jump = t_jump_q.max(dim=1, keepdim=True)[0]
            max_atk = t_atk_q.max(dim=1, keepdim=True)[0]
            target_val = (max_aim + max_move + max_jump + max_atk) / 4.0
            expected_q = rewards_t + (1.0 - dones_t) * self.gamma * target_val

        # Branching loss sum
        loss = (
            self.loss_fn(cur_aim_q, expected_q)
            + self.loss_fn(cur_move_q, expected_q)
            + self.loss_fn(cur_jump_q, expected_q)
            + self.loss_fn(cur_atk_q, expected_q)
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()

        self.last_loss = float(loss.item())

        # Update target network periodically (every 100 steps)
        if self.step_count % 100 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def run(self):
        """Real-time 20 TPS Reinforcement Learning loop."""
        tick_interval = 1.0 / 20.0
        print("\n" + "=" * 60, flush=True)
        print(" MINECRAFT REINFORCEMENT LEARNING (RL) AGENT", flush=True)
        print("=" * 60, flush=True)
        print("  Controls:", flush=True)
        print("    [F6]   : TOGGLE RL BOT ON / OFF (Audio Beep Feedback)", flush=True)
        print("    [ESC]  : INSTANT EMERGENCY STOP (Pauses and releases all keys)", flush=True)
        print("    [q]    : Quit & Save Trained Weights (in HUD window)", flush=True)
        print("    Ctrl+C : Stop Agent in terminal", flush=True)
        print("=" * 60, flush=True)
        print("[!] Press F6 in Minecraft to START Reinforcement Learning!\n", flush=True)

        next_tick = time.perf_counter()

        try:
            while True:
                if not self.cap.is_valid():
                    print("[!] Minecraft window closed. Stopping agent.", flush=True)
                    break

                # 0. Check interactive commands from desktop overlay
                if self.overlay:
                    for cmd in self.overlay.poll_commands():
                        action = cmd.get("action")
                        if action == "set_active":
                            val = bool(cmd.get("value", False))
                            self.killswitch.set_active(val)
                            status_str = "ACTIVE (FIGHTING)" if val else "PAUSED"
                            print(f"\n[AI STATE TOGGLED VIA OVERLAY: {status_str}]", flush=True)
                        elif action == "quit":
                            print("\n[!] Overlay exit clicked. Stopping RL agent...", flush=True)
                            return

                active = self.is_active

                # 1. Capture screen
                success, frame = self.cap.get_frame()
                if success and frame is not None:
                    # 2. Extract visual detection
                    det = self.detector.detect(frame)
                    state_arr = self.preprocess_frame(frame)

                    # 3. Choose action tuple
                    action_tuple = self.select_action(state_arr, det)

                    # 4. Dispatch action if active
                    if active:
                        action_dict = self.dispatch_action(action_tuple)
                        action_flags = self.reward_engine.record_action(
                            w=action_dict["w"],
                            jump=action_dict["jump"],
                            attack=action_dict["attack"],
                        )

                        # 5. Compute Visual Reward with PvP Mechanics Breakdown
                        reward_data = self.reward_engine.compute_reward(
                            frame, det, action_dict, action_flags=action_flags
                        )
                        reward = reward_data["reward"]
                        self.cumulative_reward += reward

                        # 6. Store in Replay Buffer
                        if self.prev_state_tensor is not None and self.prev_action_tuple is not None:
                            self.replay_buffer.push(self.prev_state_tensor, self.prev_action_tuple, reward, state_arr, False)

                        self.prev_state_tensor = state_arr
                        self.prev_action_tuple = action_tuple

                        # 7. Step counter & Epsilon decay (GPU training runs non-blocking in background)
                        self.step_count += 1
                        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                    else:
                        action_dict = {}
                        self.reward_engine.record_action(w=False, jump=False, attack=False)
                        reward_data = {
                            "reward": 0.0,
                            "hit_type": "none",
                            "sprint_reset_ready": self.reward_engine.sprint_reset_ready,
                            "cooldown_charge": self.reward_engine.get_attack_cooldown_charge(),
                        }
                        self.input_ctrl.release_all()

                    # 8. Update Desktop Keystrokes & Mousepad Overlay
                    if self.overlay:
                        target_dist = round(550.0 / max(30.0, float(det.get("box_h", 0))), 1) if det["has_target"] else 0.0
                        self.overlay.update(
                            active=active,
                            w=action_dict.get("w", False),
                            s=action_dict.get("s", False),
                            a=action_dict.get("a", False),
                            d=action_dict.get("d", False),
                            sprint=action_dict.get("sprint", False),
                            space=action_dict.get("jump", False),
                            attack=action_dict.get("attack", False),
                            dx=action_dict.get("dx", 0),
                            dy=action_dict.get("dy", 0),
                            target_locked=det["has_target"],
                            target_dist=target_dist,
                            cooldown=reward_data.get("cooldown_charge", 1.0),
                            hit_type=reward_data.get("hit_type", "none"),
                        )

                    # 9. Render Legacy OpenCV HUD (Optional with --cv-hud)
                    if self.show_cv_hud:
                        hud = self.detector.draw_hud(frame, det, action_dict)
                        state_color = (0, 255, 0) if active else (0, 255, 255)
                        state_txt = "RL: FIGHTING & LEARNING [F6: Pause | ESC: STOP]" if active else "RL: PAUSED [Press F6 in MC to Start]"
                        cv2.putText(hud, state_txt, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)

                        # Cooldown Charge Meter
                        charge = reward_data.get("cooldown_charge", 1.0)
                        bars = int(charge * 10)
                        meter_str = f"Atk Meter: [{'|' * bars}{'.' * (10 - bars)}] {int(charge * 100)}%"
                        meter_color = (0, 255, 0) if charge >= 0.85 else (0, 0, 255)
                        cv2.putText(hud, meter_str, (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.48, meter_color, 1)

                        # RL Metrics & Knockback Readiness
                        kb_ready = reward_data.get("sprint_reset_ready", True)
                        kb_txt = "KB READY (+40)" if kb_ready else "SWEEP MODE (+10)"
                        kb_color = (0, 255, 255) if kb_ready else (180, 180, 180)
                        rl_metric_txt = f"Reward: {reward_data['reward']:+.1f} (Tot: {self.cumulative_reward:.0f}) | {kb_txt} | Loss: {self.last_loss:.3f}"
                        cv2.putText(hud, rl_metric_txt, (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.48, kb_color, 1)

                        if reward_data.get("hit_type") != "none":
                            ht = reward_data["hit_type"]
                            hit_color = (
                                (0, 255, 0) if "knockback" in ht
                                else ((0, 255, 255) if "critical" in ht
                                else ((255, 215, 0) if "dist" in ht
                                else ((0, 0, 255) if "spam" in ht
                                else ((200, 200, 0) if "sweep" in ht
                                else (200, 200, 200)))))
                            )
                            cv2.putText(hud, f"EVENT: {ht.upper()}!", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hit_color, 2)

                        cv2.imshow("Minecraft PvP RL Agent - Live Training HUD (Press 'q' to stop)", hud)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                next_tick += tick_interval
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0.001:
                    time.sleep(sleep_time * 0.95)
                while time.perf_counter() < next_tick:
                    pass

        except KeyboardInterrupt:
            print("\n[+] Stop signal received.", flush=True)
        finally:
            if self.overlay:
                self.overlay.stop()
            self.async_trainer.stop()
            self.killswitch.stop()
            self.input_ctrl.release_all()
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(self.q_net.state_dict(), self.save_path)
            print(f"[+] RL Agent weights saved to: {os.path.abspath(self.save_path)}", flush=True)
            self.cap.close()
            if self.show_cv_hud:
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Minecraft RL PvP Agent with Desktop Keystrokes Overlay")
    parser.add_argument("--cv-hud", action="store_true", help="Display legacy OpenCV HUD debug window")
    parser.add_argument("--no-overlay", action="store_true", help="Disable the desktop overlay")
    args = parser.parse_args()

    agent = RLPvpAgent(use_overlay=not args.no_overlay, show_cv_hud=args.cv_hud)
    agent.run()


if __name__ == "__main__":
    main()
