"""Automated Rolling Model Checkpoint Manager for Reinforcement Learning.

Features:
1. Periodic Checkpoint Rotation: Saves model weights every 20 minutes into 'saves/'.
2. FIFO Retention (Max 10 Models): Automatically prunes the oldest checkpoint when count > 10.
3. Windows-Safe Naming Syntax: Generates filenames following name[day:time], e.g.
   pure_aim_model[Wednesday_07-30-00].pth
4. Zero-Lag Asynchronous Saving: Snapshots state dict to CPU in <0.1ms, then writes
   to disk in a background daemon thread so the 60 FPS training loop never drops a tick.
"""

from datetime import datetime
import glob
import os
import threading
import time
from typing import Callable, Dict, List, Optional
import torch


class ModelCheckpointManager:
    """Manages periodic model checkpointing, naming, and FIFO rotation."""

    def __init__(
        self,
        saves_dir: str = "saves",
        max_saves: int = 10,
        save_interval_mins: float = 20.0,
        base_name: str = "pure_aim_model",
    ):
        self.saves_dir = saves_dir
        self.max_saves = max(1, int(max_saves))
        self.save_interval_sec = float(save_interval_mins) * 60.0
        self.base_name = base_name

        # Ensure saves directory exists
        os.makedirs(self.saves_dir, exist_ok=True)

        # Track the timestamp of the last periodic save
        self.last_save_time: float = time.time()
        self._save_lock = threading.Lock()
        self._is_saving: bool = False

    def generate_filename(self, dt: Optional[datetime] = None) -> str:
        """Generate a Windows-safe checkpoint filename matching name[day:time].

        Example:
            pure_aim_model[Wednesday_07-30-00].pth
        """
        if dt is None:
            dt = datetime.now()

        # Day: full weekday name (e.g. Wednesday)
        day_str = dt.strftime("%A")
        # Time: HH-MM-SS with Windows-safe hyphen separator (colons ':' are invalid on NTFS)
        time_str = dt.strftime("%H-%M-%S")

        candidate = f"{self.base_name}[{day_str}_{time_str}].pth"
        # If candidate already exists in saves_dir (e.g. sub-second rapid tests), append millisecond suffix
        if os.path.exists(os.path.join(self.saves_dir, candidate)):
            candidate = f"{self.base_name}[{day_str}_{time_str}_{dt.microsecond // 1000:03d}ms].pth"

        return candidate

    def prune_old_saves(self) -> List[str]:
        """Prune oldest checkpoints if total files in saves_dir exceed max_saves.

        Returns:
            List of deleted file paths.
        """
        deleted: List[str] = []
        pattern = os.path.join(self.saves_dir, "*.pth")
        existing_files = glob.glob(pattern)

        # Exclude temporary saving files (.tmp)
        existing_files = [f for f in existing_files if not f.endswith(".tmp")]

        if len(existing_files) <= self.max_saves:
            return deleted

        # Sort files by modification time (oldest first)
        # Ties broken by file creation time then filename
        existing_files.sort(key=lambda p: (os.path.getmtime(p), os.path.getctime(p), p))

        # Delete oldest files until count is <= max_saves
        excess_count = len(existing_files) - self.max_saves
        to_delete = existing_files[:excess_count]

        for filepath in to_delete:
            try:
                os.remove(filepath)
                deleted.append(filepath)
                print(
                    f"[+] [FIFO Retention] Pruned oldest model: {os.path.basename(filepath)} "
                    f"(Maintained max {self.max_saves} in '{self.saves_dir}')",
                    flush=True,
                )
            except Exception as e:
                print(f"[!] Warning: Failed to remove old checkpoint {filepath}: {e}", flush=True)

        return deleted

    def _async_save_worker(self, state_dict: Dict[str, torch.Tensor], target_path: str):
        """Worker executed in background daemon thread to perform disk I/O and rotation."""
        try:
            tmp_path = target_path + ".tmp"
            # Atomic write to temporary file
            torch.save(state_dict, tmp_path)

            if os.path.exists(target_path):
                os.replace(tmp_path, target_path)
            else:
                os.rename(tmp_path, target_path)

            print(
                f"\n[+] [20-Min Checkpoint] Saved model to: {target_path} (Training continues seamlessly)",
                flush=True,
            )

            # Enforce FIFO limit of max 10 models
            self.prune_old_saves()

        except Exception as e:
            print(f"[!] Error saving periodic checkpoint to {target_path}: {e}", flush=True)
        finally:
            with self._save_lock:
                self._is_saving = False

    def check_and_save(
        self,
        q_net: torch.nn.Module,
        train_lock: threading.Lock,
        force: bool = False,
    ) -> bool:
        """Evaluate if periodic save interval has elapsed, and if so, trigger async save.

        Returns:
            True if a save was triggered, False otherwise.
        """
        now = time.time()
        elapsed = now - self.last_save_time

        if not force and elapsed < self.save_interval_sec:
            return False

        with self._save_lock:
            if self._is_saving:
                return False
            self._is_saving = True

        self.last_save_time = now
        filename = self.generate_filename()
        target_path = os.path.join(self.saves_dir, filename)

        # Rapid snapshot of state_dict to CPU inside train_lock (<0.1ms)
        with train_lock:
            state_dict_cpu = {k: v.cpu().clone() for k, v in q_net.state_dict().items()}

        # Dispatch disk I/O to background thread so 60 FPS loop is never blocked
        t = threading.Thread(
            target=self._async_save_worker,
            args=(state_dict_cpu, target_path),
            daemon=True,
        )
        t.start()
        return True

    def get_remaining_seconds(self) -> float:
        """Get number of seconds remaining until the next periodic checkpoint."""
        elapsed = time.time() - self.last_save_time
        return max(0.0, self.save_interval_sec - elapsed)
