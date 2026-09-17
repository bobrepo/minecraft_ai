"""Pure Reinforcement Learning Aim Agent (Stage 1 Curriculum).

Trains a Branching Dueling Double Deep Q-Network (BD-DQN) to learn
optimal mouse aiming directly from screen observations and dense alignment rewards.

Features:
1. Pure Aim Curriculum: Body is stationary; agent learns 100% pure camera aiming.
2. 10D State Space & Dual-Branch Action Head (17 Yaw x 7 Pitch = 119 actions).
3. Sub-1.5ms total step execution (guaranteed locked 60.0 FPS).
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
from checkpoint_manager import ModelCheckpointManager
from input_controller import EmergencyKillswitchListener
from overlay import PvPOverlayClient


class BranchingDuelingQNet(nn.Module):
    """Ultra-fast 2-branch dueling network for simultaneous Yaw & Pitch aim control."""

    def __init__(self, state_dim: int = 10, yaw_dim: int = 17, pitch_dim: int = 7):
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
        saves_dir: str = "saves",
        save_interval_mins: float = 20.0,
        max_saves: int = 10,
        use_overlay: bool = True,
        horizontal_only: bool = False,
        fps: int = 60,
        eval_mode: bool = False,
    ):
        self.width, self.height = resolution
        self.save_path = save_path
        self.saves_dir = saves_dir
        self.gamma = gamma
        self.eval_mode = eval_mode
        self.epsilon = 0.0 if eval_mode else epsilon_start
        self.epsilon_min = 0.0 if eval_mode else epsilon_min
        self.epsilon_decay = epsilon_decay
        self.use_overlay = use_overlay
        self.fps = max(10, min(120, fps))

        # Environment & Hardware (Full 2D Yaw + Pitch)
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
        load_target = save_path
        # Auto-recovery: if the primary model file is missing, check saves/ for the latest periodic checkpoint
        if not os.path.exists(load_target) and os.path.exists(saves_dir):
            periodic_saves = sorted(
                [os.path.join(saves_dir, f) for f in os.listdir(saves_dir) if f.endswith(".pth") and not f.endswith(".tmp")],
                key=lambda p: (os.path.getmtime(p), p)
            )
            if periodic_saves:
                load_target = periodic_saves[-1]
                print(f"[!] '{save_path}' not found. Automatically restoring from latest checkpoint in {saves_dir}: {os.path.basename(load_target)}", flush=True)

        if os.path.exists(load_target):
            try:
                state_dict = torch.load(load_target, map_location=self.device)
                # Verify shape compatibility with 17-action yaw head
                if state_dict.get("yaw_head.2.weight", torch.empty(0)).shape[0] == 17:
                    self.q_net.load_state_dict(state_dict)
                    self.target_net.load_state_dict(self.q_net.state_dict())
                    print(f"[+] Loaded existing 17-action Aim weights from: {load_target}", flush=True)
                else:
                    backup_path = load_target.replace(".pth", "_9act_backup.pth")
                    import shutil
                    if not os.path.exists(backup_path):
                        shutil.copyfile(load_target, backup_path)
                        print(f"[!] Previous 9-action weights safely backed up to: {backup_path}", flush=True)
                    self._warm_transfer_9act(state_dict)
                    print(f"[+] Successfully warm-transferred 9-action trained weights into 17-action Q-network!", flush=True)
            except Exception as e:
                # Attempt recovery from backup if main file was interrupted
                backup_path = load_target.replace(".pth", "_prev_backup.pth")
                if os.path.exists(backup_path):
                    try:
                        self.q_net.load_state_dict(torch.load(backup_path, map_location=self.device))
                        self.target_net.load_state_dict(self.q_net.state_dict())
                        print(f"[+] Successfully recovered weights from backup: {backup_path}", flush=True)
                    except Exception:
                        print(f"[!] Could not load backup weights ({e}), starting fresh.", flush=True)
                else:
                    print(f"[!] Could not load weights ({e}), starting fresh.", flush=True)

        # Continual Multi-Task Learning: Differential learning rate groups
        # Protects existing horizontal (yaw) mastery from catastrophic forgetting while enabling rapid vertical (pitch) learning
        param_groups = [
            {"params": self.q_net.shared.parameters(), "lr": lr * 0.25},    # Stable shared feature extractor
            {"params": self.q_net.val_head.parameters(), "lr": lr * 0.25},  # Stable state-value head
            {"params": self.q_net.yaw_head.parameters(), "lr": lr * 0.20},  # Gentle fine-tuning: remembers horizontal training!
            {"params": self.q_net.pitch_head.parameters(), "lr": lr},       # Full learning rate: rapidly learns vertical movements!
        ]
        self.optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
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
        self.speed_mode = "auto"
        self.lost_target_ticks = 0

        # Training lock to prevent race conditions
        self.train_lock = threading.Lock()

        # Pre-warm GPU kernels
        dummy = torch.zeros((1, 10), device=self.device)
        with torch.no_grad():
            self.q_net(dummy)
            self.target_net(dummy)

        # Asynchronous background GPU trainer (decouples backprop from 60 FPS loop)
        self.async_trainer = AsyncAimTrainer(self, batch_size=64, interval_sec=0.05)

        # Directional search memory (1 = Right, -1 = Left) when target moves off-screen
        self.last_seen_dir = 1

        # Periodic 20-Minute Rolling Checkpoint Manager into saves/ (Max 10 models FIFO)
        base_name = os.path.splitext(os.path.basename(self.save_path))[0]
        self.checkpoint_manager = ModelCheckpointManager(
            saves_dir=saves_dir,
            max_saves=max_saves,
            save_interval_mins=save_interval_mins,
            base_name=base_name,
        )

    def _warm_transfer_9act(self, d9: Dict[str, torch.Tensor]):
        """Warm-transfer weights from 9-action model into 17-action network."""
        curr = self.q_net.state_dict()
        for k in curr.keys():
            if "yaw_head.2." not in k and k in d9 and d9[k].shape == curr[k].shape:
                curr[k] = d9[k]

        w9 = d9["yaw_head.2.weight"]
        b9 = d9["yaw_head.2.bias"]
        w17 = curr["yaw_head.2.weight"].clone()
        b17 = curr["yaw_head.2.bias"].clone()

        # Map 9 yaw actions to their exact corresponding indices in 17 actions
        mapping = {0: 3, 1: 4, 2: 5, 3: 6, 4: 8, 5: 10, 6: 11, 7: 12, 8: 13}
        for i9, i17 in mapping.items():
            w17[i17] = w9[i9]
            b17[i17] = b9[i9]

        # Fast and micro actions
        w17[0] = w17[1] = w17[2] = w9[0]
        b17[0] = b17[1] = b17[2] = b9[0]
        w17[7] = (w9[3] + w9[4]) * 0.5
        b17[7] = (b9[3] + b9[4]) * 0.5
        w17[9] = (w9[5] + w9[4]) * 0.5
        b17[9] = (b9[5] + b9[4]) * 0.5
        w17[14] = w17[15] = w17[16] = w9[8]
        b17[14] = b17[15] = b17[16] = b9[8]

        curr["yaw_head.2.weight"] = w17
        curr["yaw_head.2.bias"] = b17
        self.q_net.load_state_dict(curr)
        self.target_net.load_state_dict(curr)

    def save_checkpoint(self, path: Optional[str] = None):
        """Safely save model weights with backup preservation to prevent data loss or corruption."""
        target_path = path or self.save_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # 1. Rolling backup of previous checkpoint if it exists
        if os.path.exists(target_path):
            backup_path = target_path.replace(".pth", "_prev_backup.pth")
            try:
                import shutil
                shutil.copyfile(target_path, backup_path)
            except Exception:
                pass

        # 2. Atomic write: write to .tmp then replace
        tmp_path = target_path + ".tmp"
        with self.train_lock:
            torch.save(self.q_net.state_dict(), tmp_path)
        if os.path.exists(target_path):
            os.replace(tmp_path, target_path)
        else:
            os.rename(tmp_path, target_path)

    @property
    def is_active(self) -> bool:
        return self.killswitch.is_active

    def select_action(self, state: np.ndarray, det: Dict) -> Tuple[int, int]:
        """Target-sticky action selection with 180° Backflip and Left-to-Right scanning."""
        # 1. When no target is visible: execute 180° Backflip, then scan Left <-> Right
        if not det.get("has_target", False):
            self.lost_target_ticks += 1
            # Ticks 1..4: rapid 180° flick burst (4 ticks * 160px = 640px ≈ 180°) in last seen direction
            if self.lost_target_ticks <= 4:
                yaw_act = 16 if self.last_seen_dir >= 0 else 0
                return (yaw_act, 3)
            # Ticks 5..18: hold steady to let camera settle and acquire target behind
            elif self.lost_target_ticks <= 18:
                return (8, 3)
            # Ticks 19+: smooth Left <-> Right back-and-forth scanning sweep
            else:
                cycle = ((self.lost_target_ticks - 19) // 24) % 2
                dir_mult = self.last_seen_dir if cycle == 0 else -self.last_seen_dir
                yaw_act = 11 if dir_mult >= 0 else 5
                return (yaw_act, 3)

        # Target is visible: reset off-screen counter and update last seen direction
        self.lost_target_ticks = 0
        dx = det.get("dx", 0.0)
        dy = det.get("dy", 0.0)
        if dx < -6.0:
            self.last_seen_dir = -1  # Enemy is on the left
        elif dx > 6.0:
            self.last_seen_dir = 1   # Enemy is on the right

        # 2. Sticky Target Lock: Hitbox center deadzone & velocity feedforward matching
        bh = float(det.get("box_h", 45.0))
        bw = float(det.get("box_w", 20.0))
        lock_deadzone_x = max(4.0, bw * 0.20)
        lock_deadzone_y = max(4.5, bh * 0.12)
        is_locked_x = abs(dx) <= lock_deadzone_x
        is_locked_y = abs(dy) <= lock_deadzone_y if not self.env.horizontal_only else True
        is_locked = bool(det.get("crosshair_locked", False) or det.get("in_center", is_locked_x and is_locked_y))

        if is_locked:
            target_vx = float(det.get("vx", 0.0))
            if abs(target_vx) > 2.0:
                # Find closest discrete yaw action gear to match target lateral velocity
                best_idx = min(range(len(self.env.YAW_ACTIONS)), key=lambda i: abs(self.env.YAW_ACTIONS[i] - target_vx))
                return (best_idx, 3)
            return (8, 3)  # Hold 0.0 yaw delta, 0.0 pitch delta (Center / Steady Hold)

        # 3. Target is in view: Epsilon-greedy exploration across both Yaw (17 gears) and Pitch (7 gears)
        if random.random() < self.epsilon:
            # Horizontal exploration (17 gears)
            if is_locked_x:
                yaw_act = 8  # Hold horizontal center while pitch adjusts
            elif dx < -200:
                yaw_act = random.choice([0, 1])    # Hyper / Super Left (-160, -110)
            elif dx < -120:
                yaw_act = random.choice([1, 2])    # Super / Fast Left (-110, -75)
            elif dx < -60:
                yaw_act = random.choice([2, 3])    # Fast / Med-Fast Left (-75, -50)
            elif dx < -30:
                yaw_act = random.choice([3, 4])    # Med-Fast / Med Left (-50, -32)
            elif dx < -14:
                yaw_act = random.choice([4, 5])    # Med / Med-Slow Left (-32, -18)
            elif dx < -6:
                yaw_act = random.choice([5, 6])    # Med-Slow / Slow Left (-18, -8)
            elif dx < -2:
                yaw_act = 7                        # Micro Left (-3)
            elif dx > 200:
                yaw_act = random.choice([15, 16])  # Super / Hyper Right (+110, +160)
            elif dx > 120:
                yaw_act = random.choice([14, 15])  # Fast / Super Right (+75, +110)
            elif dx > 60:
                yaw_act = random.choice([13, 14])  # Med-Fast / Fast Right (+50, +75)
            elif dx > 30:
                yaw_act = random.choice([12, 13])  # Med / Med-Fast Right (+32, +50)
            elif dx > 14:
                yaw_act = random.choice([11, 12])  # Med-Slow / Med Right (+18, +32)
            elif dx > 6:
                yaw_act = random.choice([10, 11])  # Slow / Med-Slow Right (+8, +18)
            elif dx > 2:
                yaw_act = 9                        # Micro Right (+3)
            else:
                yaw_act = 8                        # Center / Hold (0.0)

            # Vertical exploration with Sense of Enemy Height (7 gears: [-18, -8, -2.5, 0.0, 2.5, 8, 18])
            box_h = max(15.0, bh)
            dy_rel = dy / box_h

            if self.env.horizontal_only or is_locked_y:
                pitch_act = 3  # Hold vertical center (0.0)
            elif dy_rel < -0.30 or dy < -30.0:
                pitch_act = 0  # Fast Up (-18) - aiming far above enemy head
            elif dy_rel < -0.12 or dy < -12.0:
                pitch_act = random.choice([0, 1])  # Fast/Med Up (-18, -8)
            elif dy_rel < -0.04 or dy < -3.0:
                pitch_act = random.choice([1, 2])  # Med/Slow Up (-8, -2.5) - settling into torso center
            elif dy_rel > 0.30 or dy > 30.0:
                pitch_act = 6  # Fast Down (+18) - aiming far down at feet/ground
            elif dy_rel > 0.12 or dy > 12.0:
                pitch_act = random.choice([5, 6])  # Med/Fast Down (+8, +18)
            elif dy_rel > 0.04 or dy > 3.0:
                pitch_act = random.choice([4, 5])  # Slow/Med Down (+2.5, +8) - settling into torso center
            else:
                pitch_act = 3  # Center / Hold (0.0)

            return yaw_act, pitch_act

        # 4. Exploitation via Q-Network (simultaneous 2D yaw + pitch inference)
        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            yaw_q, pitch_q = self.q_net(state_t)
            yaw_act = 8 if is_locked_x else int(torch.argmax(yaw_q[0]).item())
            pitch_act = 3 if (self.env.horizontal_only or is_locked_y) else int(torch.argmax(pitch_q[0]).item())
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
        print("  Action Space:    Branching 2-Head (17 Yaw x 7 Pitch = 119 combinations)", flush=True)
        print("  Device:          " + str(self.device).upper(), flush=True)
        print("=" * 65, flush=True)
        print("  Controls:", flush=True)
        print("    [F6]    : TOGGLE RL AIM TRAINING ON / OFF (Audio Beep Feedback)", flush=True)
        print("    [ESC]   : INSTANT STOP", flush=True)
        print("    [Ctrl+C]: Stop Agent and Save Weights", flush=True)
        print("=" * 65, flush=True)
        aim_mode_str = "HORIZONTAL ONLY (Yaw Gliding)" if self.env.horizontal_only else "FULL 2D AIM (Yaw + Pitch Simultaneous)"
        print(f"  Aim Mode:        {aim_mode_str}", flush=True)
        print("  Target Cues:     Yellow (#FFF500) Body", flush=True)
        print("  Search Mode:     180° BACKFLIP on target escape + Smooth Left <-> Right Scan", flush=True)
        interval_m = self.checkpoint_manager.save_interval_sec / 60.0
        print(f"  Checkpoints:     Rolling saves to '{self.checkpoint_manager.saves_dir}/' (Every {interval_m:.0f}m, Max {self.checkpoint_manager.max_saves} FIFO)", flush=True)
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
                            if not val:
                                self.env.input_ctrl.stop_aim()
                                self.env.input_ctrl.release_all(force=True)
                        elif cmd.get("action") == "stop_aim":
                            self.killswitch.set_active(False)
                            self.env.input_ctrl.stop_aim()
                            self.env.input_ctrl.release_all(force=True)
                        elif cmd.get("action") == "set_speed_mode":
                            new_mode = str(cmd.get("mode", "auto")).lower()
                            if new_mode in ("auto", "0.5x", "1.0x", "1.5x", "2.0x"):
                                self.speed_mode = new_mode
                                mult_map = {"auto": 1.0, "0.5x": 0.5, "1.0x": 1.0, "1.5x": 1.5, "2.0x": 2.0}
                                self.env.speed_multiplier = mult_map[new_mode]
                                print(f"[+] RL Aim Speed Mode set to: {self.speed_mode.upper()} ({self.env.speed_multiplier}x)", flush=True)
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

                    # Periodic safe auto-save every 1000 steps (~16 seconds at 60 FPS)
                    if self.step_count % 1000 == 0:
                        self.save_checkpoint(self.save_path)

                    # Periodic 20-minute rolling checkpoint rotation into saves/ (Max 10 models FIFO)
                    self.checkpoint_manager.check_and_save(self.q_net, self.train_lock)

                    yaw_delta = self.env.YAW_ACTIONS[yaw_idx] * self.env.speed_multiplier
                    pitch_delta = 0.0 if self.env.horizontal_only else float(self.env.PITCH_ACTIONS[pitch_idx] * self.env.speed_multiplier)
                    det_info = current_det

                    height_zone = det_info.get("height_zone", "")
                    if det_info.get("crosshair_locked", False) or det_info.get("in_center", False):
                        rl_phase = "AIM:CENTER"
                    elif det_info.get("has_target", False):
                        if height_zone in ("HEAD", "FEET"):
                            rl_phase = f"AIM:{height_zone}"
                        else:
                            rl_phase = "AIM:2D" if not self.env.horizontal_only else "AIM:HORIZ"
                    elif self.lost_target_ticks <= 4:
                        rl_phase = "AIM:FLIP"
                    elif self.lost_target_ticks <= 18:
                        rl_phase = "AIM:SETTLE"
                    else:
                        rl_phase = "AIM:SWEEP"
                else:
                    if _was_active:
                        self.env.input_ctrl.stop_aim()
                        self.env.input_ctrl.release_all(force=True)
                        _was_active = False
                    reward = 0.0
                    yaw_delta = 0.0
                    pitch_delta = 0.0
                    det_info = {"has_target": False, "dist_px": 0.0, "in_lock_zone": False, "in_center": False, "height_zone": "NONE"}
                    rl_phase = "AIM:IDLE"

                # Update Desktop Overlay
                if self.overlay:
                    self.overlay.update(
                        active=active,
                        dx=int(yaw_delta),
                        dy=int(pitch_delta),
                        target_locked=bool(det_info.get("crosshair_locked", False) or det_info.get("in_center", False)),
                        target_dist=float(det_info.get("distance", 0.0)),
                        reward=float(reward),
                        total_score=float(self.cumulative_reward),
                        tps=current_tps,
                        phase=rl_phase,
                        speed_mode=self.speed_mode,
                        height_zone=det_info.get("height_zone", "NONE"),
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
            self.env.input_ctrl.stop_aim()
            self.env.close()
            self.save_checkpoint(self.save_path)
            print(f"[+] Pure Aim RL weights safely preserved & saved to: {os.path.abspath(self.save_path)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Pure Aim RL Trainer for Minecraft (Full 2D Yaw + Pitch, 60 FPS)")
    parser.add_argument("-w", "--window", type=str, default=None, help="Target window title / HWND")
    parser.add_argument("--no-overlay", action="store_true", help="Disable desktop keystroke overlay")
    parser.add_argument("--fps", type=int, default=60, help="RL loop update rate in FPS (default: 60)")
    parser.add_argument("--play", "--eval", action="store_true", help="Play mode: zero exploration (epsilon=0) for pure high-performance match play")
    parser.add_argument("--horizontal-only", action="store_true", help="Restrict RL strictly to horizontal yaw")
    parser.add_argument("-m", "--model", type=str, default="models/pure_aim_model.pth", help="Path to primary model weights file (default: models/pure_aim_model.pth)")
    parser.add_argument("--saves-dir", type=str, default="saves", help="Directory for periodic 20-minute checkpoints (default: saves)")
    parser.add_argument("--save-interval-mins", type=float, default=20.0, help="Periodic checkpoint save interval in minutes (default: 20)")
    parser.add_argument("--max-saves", type=int, default=10, help="Maximum number of periodic checkpoints to keep in saves/ (default: 10)")
    args = parser.parse_args()

    agent = PureAimRLAgent(
        target_window=args.window,
        use_overlay=not args.no_overlay,
        horizontal_only=args.horizontal_only,
        fps=args.fps,
        eval_mode=args.play,
        save_path=args.model,
        saves_dir=args.saves_dir,
        save_interval_mins=args.save_interval_mins,
        max_saves=args.max_saves,
    )
    agent.run()


if __name__ == "__main__":
    main()
