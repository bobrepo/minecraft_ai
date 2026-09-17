"""Unit tests for ModelCheckpointManager (Periodic 20-min saves, FIFO retention, Windows-safe naming)."""

from datetime import datetime
import os
import shutil
import tempfile
import threading
import time
import unittest
import torch
import torch.nn as nn

from checkpoint_manager import ModelCheckpointManager


class SimpleTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class TestModelCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.net = SimpleTestNet()
        self.lock = threading.Lock()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_naming_syntax(self):
        """Verify filename conforms to name[day:time] syntax and is valid on Windows."""
        mgr = ModelCheckpointManager(saves_dir=self.temp_dir, base_name="pure_aim_model")
        dt = datetime(2026, 9, 16, 14, 30, 0)
        filename = mgr.generate_filename(dt)
        expected = "pure_aim_model[Wednesday_14-30-00].pth"
        self.assertEqual(filename, expected)
        # Verify no illegal characters for Windows filenames
        for char in [':', '<', '>', '"', '/', '\\', '|', '?', '*']:
            self.assertNotIn(char, filename)

    def test_fifo_pruning_max_10_models(self):
        """Verify that when more than 10 models exist, the oldest ones are deleted first."""
        mgr = ModelCheckpointManager(saves_dir=self.temp_dir, max_saves=10)

        # Create 14 dummy model files with distinct modification timestamps
        created_files = []
        for i in range(14):
            fname = f"pure_aim_model[Day_{i:02d}-00-00].pth"
            fpath = os.path.join(self.temp_dir, fname)
            with open(fpath, "w") as f:
                f.write(f"model_version_{i}")
            # Ensure strictly increasing mtime
            mtime = time.time() - (14 - i) * 10
            os.utime(fpath, (mtime, mtime))
            created_files.append(fpath)

        # Before pruning
        self.assertEqual(len(os.listdir(self.temp_dir)), 14)

        # Prune
        deleted = mgr.prune_old_saves()

        # Check that exactly 4 files were deleted (14 - 10)
        self.assertEqual(len(deleted), 4)
        # Check that the 4 oldest files (indices 0, 1, 2, 3) were the ones deleted
        for f in created_files[:4]:
            self.assertIn(f, deleted)
            self.assertFalse(os.path.exists(f))

        # Check that the 10 newest files (indices 4..13) still exist
        remaining_files = os.listdir(self.temp_dir)
        self.assertEqual(len(remaining_files), 10)
        for f in created_files[4:]:
            self.assertTrue(os.path.exists(f))

    def test_check_and_save_timing(self):
        """Verify check_and_save only triggers when interval has elapsed or forced."""
        # 1-second interval for test
        mgr = ModelCheckpointManager(saves_dir=self.temp_dir, save_interval_mins=0.0166)  # ~1 sec

        # Immediately after init, elapsed is 0 -> should not save
        saved = mgr.check_and_save(self.net, self.lock, force=False)
        self.assertFalse(saved)

        # Force save -> should trigger
        saved = mgr.check_and_save(self.net, self.lock, force=True)
        self.assertTrue(saved)

        # Wait for async thread to complete
        time.sleep(0.1)
        files = [f for f in os.listdir(self.temp_dir) if f.endswith(".pth")]
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].startswith("pure_aim_model["))
        self.assertTrue(files[0].endswith("].pth"))

    def test_async_save_content_integrity(self):
        """Verify that saved checkpoint can be reloaded by PyTorch without corruption."""
        mgr = ModelCheckpointManager(saves_dir=self.temp_dir)
        mgr.check_and_save(self.net, self.lock, force=True)

        time.sleep(0.1)
        files = [f for f in os.listdir(self.temp_dir) if f.endswith(".pth")]
        self.assertEqual(len(files), 1)

        saved_path = os.path.join(self.temp_dir, files[0])
        loaded_sd = torch.load(saved_path, map_location="cpu")
        orig_sd = self.net.state_dict()

        for k in orig_sd.keys():
            self.assertTrue(torch.allclose(orig_sd[k], loaded_sd[k]))


if __name__ == "__main__":
    unittest.main()
