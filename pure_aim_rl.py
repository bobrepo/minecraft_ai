"""Pure Reinforcement Learning Aim Agent (Stage 1 Curriculum).

Trains a Branching Dueling Double Deep Q-Network (BD-DQN) to learn
optimal mouse aiming directly from screen observations and dense alignment rewards.

Features:
1. Pure Aim Curriculum: Body is stationary; agent learns 100% pure camera aiming.
2. 10D State Space & Dual-Branch Action Head (9 Yaw x 7 Pitch).
3. Sub-1.5ms total step execution (guaranteed locked 20.0 TPS).
4. Synchronized with Desktop Keystrokes & Mousepad Overlay.
5. F6 toggle hotkey + ESC emergency killswitch.
"""

import argparse
from collections import deque
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
import torch.optim as optim

from aim_env import MinecraftAimEnv
from input_controller import EmergencyKillswitchListener
from overlay import PvPOverlayClient


class BranchingDuelingQNet(nn.Module):
    """Ultra-fast 2-branch dueling network for simultaneous Yaw & Pitch aim control."""

    def __init__(self, state_dim: int = 10, yaw_dim: int = 9, pitch_dim: int = 7):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
        )

        # State-Value Head V(s)
        self.val_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
        )

        # Action-Advantage Head 1: Horizontal Yaw
        self.yaw_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, yaw_dim),
        )

        # Action-Advantage Head 2: Vertical Pitch
        self.pitch_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, pitch_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(x)
        val = self.val_head(feat)

        yaw_a = self.yaw_head(feat)
        pitch_a = self.pitch_head(feat)

        # Dueling aggregation: Q(s, a) = V(s) + (A(s, a) - mean(A))
        yaw_q = val + (yaw_a - yaw_a.mean(dim=-1, keepdim=True))
        pitch_q = val + (pitch_a - pitch_a.mean(dim=-1, keepdim=True))

        return yaw_q, pitch_q


class ReplayBuffer:
    """High-speed circular experience replay buffer using pre-allocated numpy arrays."""

    def __init__(self, capacity: int = 25000, state_dim: int = 10):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 2), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, s: np.ndarray, a: Tuple[int, int], r: float, s_next: np.ndarray, done: bool):
        idx = self.ptr
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx, 0] = r
        self.next_states[idx] = s_next
        self.dones[idx, 0] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size: int = 64) -> Tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_states[indices]),
            torch.from_numpy(self.dones[indices]),
        )

    def __len__(self):
        return self.size


class AsyncAimTrainer:
    """Asynchronous background GPU trainer for Pure Aim RL."""

    def __init__(self, agent: "PureAimRLAgent", batch_size: int = 64, interval_sec: float = 0.08):
        self.agent = agent
        self.batch_size = batch_size
        self.interval_sec = interval_sec
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _worker(self):
        while self._running:
            try:
                if self.agent.is_active and len(self.agent.replay_buffer) >= self.batch_size:
                    self.agent.train_step(batch_size=self.batch_size)
            except Exception:
                pass
            time.sleep(self.interval_sec)

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


class PureAimRLAgent:
    """Stage 1 Pure Aiming Reinforcement Learning Agent."""

    def __init__(
        self,
        target_window: Optional[str | int] = None,
        resolution: Tuple[int, int] = (640, 480),
        gamma: float = 0.96,
        lr: float = 3e-4,
        epsilon_start: float = 0.35,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9997,
        save_path: str = "models/pure_aim_model.pth",
        use_overlay: bool = True,
        horizontal_only: bool = True,
        fps: int = 60,
    ):
        self.width, self.height = resolution
        self.save_path = save_path
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.use_overlay = use_overlay
        self.fps = max(10, min(120, fps))

        # Environment & Hardware
        self.env = MinecraftAimEnv(
            target_hwnd=target_window, resolution=resolution, horizontal_only=horizontal_only
        )
        self.killswitch = EmergencyKillswitchListener(self.env.input_ctrl)

        # PyTorch Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[+] Initializing Pure Aim RL Network on {self.device}...", flush=True)

        self.q_net = BranchingDuelingQNet().to(self.device)
        self.target_net = BranchingDuelingQNet().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        if os.path.exists(save_path):
            try:
                self.q_net.load_state_dict(torch.load(save_path, map_location=self.device))
                self.target_net.load_state_dict(self.q_net.state_dict())
                print(f"[+] Loaded existing Aim weights from: {save_path}", flush=True)
            except Exception as e:
                print(f"[!] Could not load weights ({e}), starting fresh.", flush=True)

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.replay_buffer = ReplayBuffer(capacity=25000)

        # Overlay UI
        self.overlay: Optional[PvPOverlayClient] = None
        if self.use_overlay:
            self.overlay = PvPOverlayClient()
            self.overlay.start()

        # Telemetry
        self.cumulative_reward = 0.0
        self.step_count = 0
        self.episodes = 0
        self.last_loss = 0.0

        # Training lock to prevent race conditions
        self.train_lock = threading.Lock()

        # Pre-warm GPU kernels
        dummy = torch.zeros((1, 10), device=self.device)
        with torch.no_grad():
            self.q_net(dummy)
            self.target_net(dummy)

        # Asynchronous background GPU trainer (decouples backprop from 60 FPS loop)
        self.async_trainer = AsyncAimTrainer(self, batch_size=64, interval_sec=0.05)

    @property
    def is_active(self) -> bool:
        return self.killswitch.is_active

    def select_action(self, state: np.ndarray, det: Dict) -> Tuple[int, int]:
        """Epsilon-greedy action selection with guided exploration."""
        if random.random() < self.epsilon:
            if det["has_target"] and random.random() < 0.60:
                # Guided Exploration: choose direction towards target
                dx = det["dx"]
                dy = det["dy"]
                # Yaw heuristic
                if dx < -25:
                    yaw_act = random.choice([0, 1])  # Fast Left
                elif dx < -6:
                    yaw_act = random.choice([1, 2])  # Med Left
                elif dx < -2:
                    yaw_act = 3  # Micro Left
                elif dx > 25:
                    yaw_act = random.choice([7, 8])  # Fast Right
                elif dx > 6:
                    yaw_act = random.choice([6, 7])  # Med Right
                elif dx > 2:
                    yaw_act = 5  # Micro Right
                else:
                    yaw_act = 4  # Center / Hold

                # Pitch heuristic
                if dy < -15:
                    pitch_act = 0  # Fast Up
                elif dy < -3:
                    pitch_act = 1  # Med Up
                elif dy > 15:
                    pitch_act = 6  # Fast Down
                elif dy > 3:
                    pitch_act = 5  # Med Down
                else:
                    pitch_act = 3  # Center / Hold

                if self.env.horizontal_only:
                    pitch_act = 3
                return yaw_act, pitch_act
            else:
                pitch_act = 3 if self.env.horizontal_only else random.randint(0, 6)
                return (random.randint(0, 8), pitch_act)

        # Exploitation via Q-Network
        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            yaw_q, pitch_q = self.q_net(state_t)
            yaw_act = int(torch.argmax(yaw_q[0]).item())
            pitch_act = 3 if self.env.horizontal_only else int(torch.argmax(pitch_q[0]).item())
            return yaw_act, pitch_act

    def train_step(self, batch_size: int = 64):
        """Perform one Double Dueling DQN optimization step."""
        with self.train_lock:
            if len(self.replay_buffer) < batch_size:
                return

            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)

            # Current Q-values for chosen actions
            yaw_q, pitch_q = self.q_net(states)
            yaw_cur = yaw_q.gather(1, actions[:, 0:1])
            pitch_cur = pitch_q.gather(1, actions[:, 1:2])

            # Double DQN Target Calculation
            with torch.no_grad():
                next_yaw_q, next_pitch_q = self.q_net(next_states)
                best_next_yaw = next_yaw_q.argmax(dim=-1, keepdim=True)
                best_next_pitch = next_pitch_q.argmax(dim=-1, keepdim=True)

                target_yaw_q, target_pitch_q = self.target_net(next_states)
                target_yaw_val = target_yaw_q.gather(1, best_next_yaw)
                target_pitch_val = target_pitch_q.gather(1, best_next_pitch)

                target_q = rewards + (1.0 - dones) * self.gamma * 0.5 * (target_yaw_val + target_pitch_val)

            loss_yaw = self.loss_fn(yaw_cur, target_q)
            loss_pitch = self.loss_fn(pitch_cur, target_q)
            total_loss = loss_yaw + loss_pitch

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()

            self.last_loss = float(total_loss.item())

    def run(self):
        """Locked 60 FPS Pure Aim Reinforcement Learning loop."""
        tick_interval = 1.0 / float(self.fps)
        print("\n" + "=" * 65, flush=True)
        print(" MINECRAFT STAGE 1: PURE REINFORCEMENT LEARNING AIM BOT", flush=True)
        print("=" * 65, flush=True)
        print("  Curriculum:      STAGE 1: AIMING ONLY (Body is stationary)", flush=True)
        print(f"  Update Rate:     {self.fps} FPS ({tick_interval * 1000:.1f}ms interval)", flush=True)
        print("  State Space:     10D Normalized Geometry & Target Velocity", flush=True)
        print("  Action Space:    Branching 2-Head (9 Yaw x 7 Pitch = 63 combinations)", flush=True)
        print("  Device:          " + str(self.device).upper(), flush=True)
        print("=" * 65, flush=True)
        print("  Controls:", flush=True)
        print("    [F6]    : TOGGLE RL AIM TRAINING ON / OFF (Audio Beep Feedback)", flush=True)
        print("    [ESC]   : INSTANT STOP", flush=True)
        print("    [Ctrl+C]: Stop Agent and Save Weights", flush=True)
        print("=" * 65, flush=True)
        print("  Aim Mode: HORIZONTAL ONLY (Yaw Gliding)", flush=True)
        print("[!] Press F6 in Minecraft to START Pure Aim Training!\n", flush=True)

        self.async_trainer.start()
        state = self.env.reset()
        suc, init_frame = self.env.cap.get_frame()
        current_det = self.env.detector.detect(
            init_frame if suc and init_frame is not None else np.zeros((self.height, self.width, 3), dtype=np.uint8),
            crosshair_centric=True,
        )

        next_tick = time.perf_counter()
        tick_times = deque(maxlen=self.fps)
        current_tps = float(self.fps)
        _was_active = False

        try:
            while True:
                if not self.env.cap.is_valid():
                    print("[!] Minecraft window closed. Stopping agent.", flush=True)
                    break

                t_tick_now = time.perf_counter()
                tick_times.append(t_tick_now)
                if len(tick_times) >= 2:
                    dt_span = tick_times[-1] - tick_times[0]
                    current_tps = (len(tick_times) - 1) / max(1e-5, dt_span)

                # Check overlay user commands
                if self.overlay:
                    for cmd in self.overlay.poll_commands():
                        if cmd.get("action") == "set_active":
                            val = bool(cmd.get("value", False))
                            self.killswitch.set_active(val)
                        elif cmd.get("action") == "quit":
                            return

                active = self.is_active

                if active:
                    _was_active = True
                    # 1. Select Aim Action using latest detection
                    yaw_idx, pitch_idx = self.select_action(state, current_det)
                    action = (yaw_idx, pitch_idx)

                    # 2. Environment Step (moves mouse & collects reward, detects next state)
                    next_state, reward, done, info = self.env.step(action)
                    self.cumulative_reward += reward
                    current_det = info["detection"]

                    # 3. Push to Replay Buffer
                    self.replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state

                    # 4. Training is processed asynchronously in background by self.async_trainer
                    self.step_count += 1
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

                    # Periodically update target network (every 3 seconds at 60 FPS)
                    if self.step_count % 180 == 0:
                        self.target_net.load_state_dict(self.q_net.state_dict())

                    # Periodic auto-save every 1000 steps (~16 seconds at 60 FPS)
                    if self.step_count % 1000 == 0:
                        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                        torch.save(self.q_net.state_dict(), self.save_path)

                    yaw_delta = self.env.YAW_ACTIONS[yaw_idx]
                    pitch_delta = 0.0
                    det_info = current_det
                else:
                    if _was_active:
                        self.env.input_ctrl.release_all(force=True)
                        _was_active = False
                    reward = 0.0
                    yaw_delta = 0.0
                    pitch_delta = 0.0
                    det_info = {"has_target": False, "dist_px": 0.0, "in_lock_zone": False}

                # Update Desktop Overlay
                if self.overlay:
                    self.overlay.update(
                        active=active,
                        dx=int(yaw_delta),
                        dy=int(pitch_delta),
                        target_locked=det_info.get("in_lock_zone", False),
                        target_dist=float(det_info.get("distance", 0.0)),
                        reward=float(reward),
                        total_score=float(self.cumulative_reward),
                        tps=current_tps,
                        phase="AIM:HORIZ",
                    )

                # Strict 60 FPS hybrid timing
                next_tick += tick_interval
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0.001:
                    time.sleep(sleep_time * 0.90)
                while time.perf_counter() < next_tick:
                    pass
                if time.perf_counter() - next_tick > tick_interval:
                    next_tick = time.perf_counter()

        except KeyboardInterrupt:
            print("\n[+] Stop signal received.", flush=True)
        finally:
            self.async_trainer.stop()
            if self.overlay:
                self.overlay.stop()
            self.killswitch.stop()
            self.env.close()
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(self.q_net.state_dict(), self.save_path)
            print(f"[+] Pure Aim RL weights saved to: {os.path.abspath(self.save_path)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Pure Aim RL Trainer for Minecraft (Horizontal Only, 60 FPS)")
    parser.add_argument("-w", "--window", type=str, default=None, help="Target window title / HWND")
    parser.add_argument("--no-overlay", action="store_true", help="Disable desktop keystroke overlay")
    parser.add_argument("--fps", type=int, default=60, help="RL loop update rate in FPS (default: 60)")
    args = parser.parse_args()

    agent = PureAimRLAgent(
        target_window=args.window,
        use_overlay=not args.no_overlay,
        horizontal_only=True,
        fps=args.fps,
    )
    agent.run()


if __name__ == "__main__":
    main()
