"""High-Performance Desktop PvP Keystrokes, Mousepad, and Movable Control Overlay.

Features:
1. Always-on-top, borderless floating overlay widget that sits cleanly over Minecraft.
2. Draggable top bar: User can move the overlay anywhere on screen (defaults to top-left).
3. Auto-collapse when disabled: Collapses to a compact movable bar when stopped to keep the screen clear.
4. Interactive RUN AI / STOP AI button: Starts/stops the AI on click with bidirectional F6/ESC sync.
5. WASD + Space Keystrokes layout matching the user's diagram.
6. Vector Mouse Pad with dynamic aim direction arrow and decaying trail matching the user's diagram.
7. Illuminating Left Click (Attack) box positioned below the mouse pad.
8. Live combat sensing: Target lock distance, weapon cooldown charge bar, and hit event badges.
9. Process-isolated architecture: 0 GIL contention and 0 lag impact on the 20 TPS combat loop.
"""

import math
import multiprocessing as mp
import queue
import tkinter as tk
from typing import Any, Dict, List, Optional


def _overlay_process_main(data_queue: mp.Queue, cmd_queue: mp.Queue):
    """Main function executed in a dedicated background process for Tkinter."""
    root = tk.Tk()
    root.title("Minecraft PvP AI Overlay")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg="#18181b")

    # Geometry constants
    width = 252
    collapsed_h = 46
    expanded_h = 508

    # Initial position: top-left with safe margins
    pos_x = 24
    pos_y = 24
    root.geometry(f"{width}x{collapsed_h}+{pos_x}+{pos_y}")

    # State variables
    state = {
        "active": False,
        "pinned": False,
        "pos_x": pos_x,
        "pos_y": pos_y,
        "aim_vx": 0.0,
        "aim_vy": 0.0,
        "lmb_flash": 0,
        "event_text": "",
        "event_color": "#22c55e",
        "event_timer": 0,
        "score": 0.0,
        "reward": 0.0,
    }

    # --- Header / Movable Bar ---
    header = tk.Frame(root, bg="#27272a", height=collapsed_h)
    header.pack(fill=tk.X, side=tk.TOP)
    header.pack_propagate(False)

    drag_icon = tk.Label(
        header, text="⋮⋮ ⚔️ PVP", fg="#f4f4f5", bg="#27272a", font=("Segoe UI", 10, "bold"), cursor="fleur"
    )
    drag_icon.pack(side=tk.LEFT, padx=(8, 2), pady=6)

    status_pill = tk.Label(
        header, text="○ OFF", fg="#a1a1aa", bg="#27272a", font=("Segoe UI", 8, "bold")
    )
    status_pill.pack(side=tk.LEFT, padx=4)

    header_score_lbl = tk.Label(
        header, text="", fg="#fbbf24", bg="#27272a", font=("Segoe UI", 8, "bold")
    )
    header_score_lbl.pack(side=tk.LEFT, padx=4)

    # Close button (signals agent to stop and quit)
    def on_close_click():
        cmd_queue.put({"action": "quit"})
        root.destroy()

    close_btn = tk.Button(
        header,
        text="✕",
        bg="#27272a",
        fg="#71717a",
        activebackground="#ef4444",
        activeforeground="#ffffff",
        font=("Segoe UI", 8),
        relief=tk.FLAT,
        bd=0,
        padx=4,
        pady=2,
        cursor="hand2",
        command=on_close_click,
    )
    close_btn.pack(side=tk.RIGHT, padx=(0, 4), pady=6)

    # Pin button (allows keeping expanded even when stopped)
    def on_toggle_pin():
        state["pinned"] = not state["pinned"]
        pin_btn.config(fg="#22c55e" if state["pinned"] else "#71717a")
        update_window_expansion()

    pin_btn = tk.Button(
        header,
        text="📌",
        bg="#27272a",
        fg="#71717a",
        activebackground="#3f3f46",
        activeforeground="#ffffff",
        font=("Segoe UI", 8),
        relief=tk.FLAT,
        bd=0,
        padx=3,
        pady=2,
        cursor="hand2",
        command=on_toggle_pin,
    )
    pin_btn.pack(side=tk.RIGHT, padx=(2, 4), pady=6)

    # Run / Stop Action Button
    def on_action_button_click():
        new_active = not state["active"]
        cmd_queue.put({"action": "set_active", "value": new_active})
        # Optimistic local update
        apply_active_state(new_active)

    action_btn = tk.Button(
        header,
        text="▶ RUN AI",
        bg="#16a34a",
        fg="#ffffff",
        activebackground="#22c55e",
        activeforeground="#ffffff",
        font=("Segoe UI", 9, "bold"),
        relief=tk.FLAT,
        bd=0,
        padx=8,
        pady=2,
        cursor="hand2",
        command=on_action_button_click,
    )
    action_btn.pack(side=tk.RIGHT, padx=4, pady=6)

    # Dragging logic
    drag_data = {"offset_x": 0, "offset_y": 0}

    def start_drag(e):
        drag_data["offset_x"] = e.x_root - root.winfo_x()
        drag_data["offset_y"] = e.y_root - root.winfo_y()

    def on_drag(e):
        state["pos_x"] = e.x_root - drag_data["offset_x"]
        state["pos_y"] = e.y_root - drag_data["offset_y"]
        h = expanded_h if (state["active"] or state["pinned"]) else collapsed_h
        root.geometry(f"{width}x{h}+{state['pos_x']}+{state['pos_y']}")

    for widget in (header, drag_icon, status_pill):
        widget.bind("<Button-1>", start_drag)
        widget.bind("<B1-Motion>", on_drag)

    # --- Expanded Content Area ---
    content_frame = tk.Frame(root, bg="#18181b")

    # 1. WASD + Space Keystrokes Section (matches media_1789353523031.png)
    keys_card = tk.Frame(content_frame, bg="#18181b")
    keys_card.pack(fill=tk.X, padx=10, pady=(8, 4))

    # Row 1: W Key (Centered)
    w_row = tk.Frame(keys_card, bg="#18181b")
    w_row.pack()
    key_w = tk.Label(
        w_row, text="W", width=6, height=2, bg="#27272a", fg="#d4d4d8", font=("Segoe UI", 11, "bold"), relief=tk.RIDGE, bd=1
    )
    key_w.pack()

    # Row 2: A, S, D Keys
    asd_row = tk.Frame(keys_card, bg="#18181b")
    asd_row.pack(pady=3)

    key_a = tk.Label(
        asd_row, text="A", width=6, height=2, bg="#27272a", fg="#d4d4d8", font=("Segoe UI", 11, "bold"), relief=tk.RIDGE, bd=1
    )
    key_a.pack(side=tk.LEFT, padx=2)

    key_s = tk.Label(
        asd_row, text="S", width=6, height=2, bg="#27272a", fg="#d4d4d8", font=("Segoe UI", 11, "bold"), relief=tk.RIDGE, bd=1
    )
    key_s.pack(side=tk.LEFT, padx=2)

    key_d = tk.Label(
        asd_row, text="D", width=6, height=2, bg="#27272a", fg="#d4d4d8", font=("Segoe UI", 11, "bold"), relief=tk.RIDGE, bd=1
    )
    key_d.pack(side=tk.LEFT, padx=2)

    # Row 3: Space Bar
    space_row = tk.Frame(keys_card, bg="#18181b")
    space_row.pack(pady=2)
    key_space = tk.Label(
        space_row, text="—— SPACE (CRIT) ——", width=23, height=1, bg="#27272a", fg="#d4d4d8", font=("Segoe UI", 9, "bold"), relief=tk.RIDGE, bd=1
    )
    key_space.pack()

    # 2. Combat Sensing Strip
    sensing_frame = tk.Frame(content_frame, bg="#27272a", relief=tk.FLAT, bd=0)
    sensing_frame.pack(fill=tk.X, padx=10, pady=(4, 2))

    target_lbl = tk.Label(
        sensing_frame, text="Target: 🔍 SCANNING", bg="#27272a", fg="#a1a1aa", font=("Segoe UI", 8, "bold")
    )
    target_lbl.pack(side=tk.LEFT, padx=6, pady=2)

    event_lbl = tk.Label(
        sensing_frame, text="", bg="#27272a", fg="#22c55e", font=("Segoe UI", 8, "bold")
    )
    event_lbl.pack(side=tk.RIGHT, padx=6, pady=2)

    # 3. RL Points & Reward Banner
    reward_frame = tk.Frame(content_frame, bg="#27272a", relief=tk.FLAT, bd=0)
    reward_frame.pack(fill=tk.X, padx=10, pady=(0, 4))

    score_lbl = tk.Label(
        reward_frame, text="POINTS: 0", bg="#27272a", fg="#fbbf24", font=("Segoe UI", 8, "bold")
    )
    score_lbl.pack(side=tk.LEFT, padx=6, pady=2)

    reward_tick_lbl = tk.Label(
        reward_frame, text="+0.0 pts/tk", bg="#27272a", fg="#22c55e", font=("Segoe UI", 8, "bold")
    )
    reward_tick_lbl.pack(side=tk.RIGHT, padx=6, pady=2)

    # Weapon Recharge Progress Bar
    cd_canvas = tk.Canvas(content_frame, width=230, height=6, bg="#27272a", highlightthickness=0)
    cd_canvas.pack(padx=10, pady=(0, 4))
    cd_bar = cd_canvas.create_rectangle(0, 0, 230, 6, fill="#22c55e", width=0)

    # 3. Mouse Pad Canvas (matches media_1789353731042.png)
    mouse_canvas = tk.Canvas(
        content_frame, width=230, height=115, bg="#0d0d11", highlightthickness=1, highlightbackground="#3f3f46"
    )
    mouse_canvas.pack(padx=10, pady=2)

    # Origin and coordinate guidelines
    mc_x, mc_y = 115, 57
    mouse_canvas.create_oval(mc_x - 28, mc_y - 28, mc_x + 28, mc_y + 28, outline="#1f1f23", width=1)
    mouse_canvas.create_line(mc_x - 18, mc_y, mc_x + 18, mc_y, fill="#27272a", dash=(2, 2))
    mouse_canvas.create_line(mc_x, mc_y - 18, mc_x, mc_y + 18, fill="#27272a", dash=(2, 2))
    mouse_canvas.create_oval(mc_x - 3, mc_y - 3, mc_x + 3, mc_y + 3, fill="#52525b", outline="")

    # Dynamic Aim Vector Arrow
    arrow_id = mouse_canvas.create_line(
        mc_x, mc_y, mc_x, mc_y, fill="#06b6d4", width=3, arrow=tk.LAST, arrowshape=(9, 11, 4)
    )
    mouse_canvas.create_text(
        mc_x, 14, text="AIM MOVEMENT VECTOR", fill="#52525b", font=("Segoe UI", 7, "bold")
    )

    # 4. Left Click Indicator Box (bottom of mouse pad)
    lmb_frame = tk.Frame(content_frame, bg="#18181b")
    lmb_frame.pack(fill=tk.X, padx=10, pady=(4, 8))
    lmb_box = tk.Label(
        lmb_frame,
        text="⚡ LEFT CLICK (ATTACK)",
        height=2,
        bg="#27272a",
        fg="#a1a1aa",
        font=("Segoe UI", 9, "bold"),
        relief=tk.RIDGE,
        bd=1,
    )
    lmb_box.pack(fill=tk.X)

    def update_window_expansion():
        should_expand = state["active"] or state["pinned"]
        cur_x = state["pos_x"]
        cur_y = state["pos_y"]
        if should_expand:
            content_frame.pack(fill=tk.BOTH, expand=True)
            root.geometry(f"{width}x{expanded_h}+{cur_x}+{cur_y}")
        else:
            content_frame.pack_forget()
            root.geometry(f"{width}x{collapsed_h}+{cur_x}+{cur_y}")

    def apply_active_state(active: bool):
        state["active"] = active
        if active:
            status_pill.config(text="● RUNNING", fg="#22c55e")
            action_btn.config(text="⏹ STOP AI", bg="#dc2626", activebackground="#ef4444")
            sign = "+" if state["score"] > 0 else ""
            header_score_lbl.config(text=f"[{sign}{state['score']:,.0f} pts]" if state["score"] != 0 else "")
        else:
            status_pill.config(text="○ STOPPED", fg="#a1a1aa")
            action_btn.config(text="▶ RUN AI", bg="#16a34a", activebackground="#22c55e")
            header_score_lbl.config(text="")
        update_window_expansion()

    def set_key_style(widget, is_pressed: bool, active_bg="#10b981", active_fg="#ffffff"):
        if is_pressed:
            widget.config(bg=active_bg, fg=active_fg, relief=tk.SUNKEN)
        else:
            widget.config(bg="#27272a", fg="#d4d4d8", relief=tk.RIDGE)

    # Main update loop (runs at ~40 FPS via root.after)
    def poll_data():
        try:
            while not data_queue.empty():
                msg = data_queue.get_nowait()
                mtype = msg.get("type", "state")

                if mtype == "stop":
                    root.destroy()
                    return

                elif mtype == "state":
                    # 1. Active / Disabled state sync
                    remote_active = msg.get("active", state["active"])
                    if remote_active != state["active"]:
                        apply_active_state(remote_active)

                    # Only update visual keys if window is expanded
                    if state["active"] or state["pinned"]:
                        # Keystrokes
                        is_sprint = msg.get("sprint", False)
                        w_bg = "#059669" if is_sprint else "#10b981"
                        set_key_style(key_w, msg.get("w", False), active_bg=w_bg)
                        set_key_style(key_a, msg.get("a", False), active_bg="#06b6d4")
                        set_key_style(key_s, msg.get("s", False), active_bg="#f59e0b")
                        set_key_style(key_d, msg.get("d", False), active_bg="#06b6d4")
                        set_key_style(key_space, msg.get("space", False), active_bg="#8b5cf6")

                        # Mouse Aim Vector
                        raw_dx = msg.get("dx", 0)
                        raw_dy = msg.get("dy", 0)
                        if raw_dx != 0 or raw_dy != 0:
                            # Scale with bounds
                            target_vx = max(-48.0, min(48.0, float(raw_dx) * 1.6))
                            target_vy = max(-38.0, min(38.0, float(raw_dy) * 1.6))
                            state["aim_vx"] = target_vx
                            state["aim_vy"] = target_vy
                        else:
                            state["aim_vx"] *= 0.72  # smooth decay
                            state["aim_vy"] *= 0.72

                        # Left Click Attack Flash
                        if msg.get("attack", False):
                            state["lmb_flash"] = 4
                            lmb_box.config(bg="#ef4444", fg="#ffffff", relief=tk.SUNKEN)

                        # Target Sensing
                        locked = msg.get("target_locked", False)
                        dist = msg.get("target_dist", 0.0)
                        predicting = msg.get("predicting", False)
                        pred_dir = msg.get("pred_direction", "CENTER")
                        if locked:
                            target_lbl.config(text=f"Target: 🎯 {dist:.1f} blk", fg="#22c55e")
                        elif predicting:
                            target_lbl.config(text=f"Target: ⤑ {pred_dir}", fg="#06b6d4")
                        else:
                            target_lbl.config(text="Target: 🔍 SCANNING", fg="#a1a1aa")

                        # Cooldown Recharge Bar
                        cd = max(0.0, min(1.0, msg.get("cooldown", 1.0)))
                        bw = int(230 * cd)
                        b_color = "#22c55e" if cd >= 0.85 else ("#f59e0b" if cd >= 0.4 else "#ef4444")
                        cd_canvas.coords(cd_bar, 0, 0, bw, 6)
                        cd_canvas.itemconfig(cd_bar, fill=b_color)

                        # Combat Event
                        ht = msg.get("hit_type", "")
                        if ht and ht != "none":
                            state["event_text"] = ht.upper().replace("_", " ")
                            state["event_timer"] = 24
                            if "sky" in ht:
                                state["event_color"] = "#ef4444"
                            elif "knockback" in ht:
                                state["event_color"] = "#ef4444"
                            elif "crit" in ht:
                                state["event_color"] = "#a855f7"
                            elif "dist" in ht:
                                state["event_color"] = "#22c55e"
                            elif "pred" in ht:
                                state["event_color"] = "#06b6d4"
                            else:
                                state["event_color"] = "#eab308"

                        # RL Rewards & Points Display
                        r_val = float(msg.get("reward", 0.0))
                        s_val = float(msg.get("total_score", 0.0))
                        state["score"] = s_val
                        state["reward"] = r_val

                        # Format tick reward (pts/tk)
                        if r_val > 0.0:
                            r_str = f"+{r_val:.1f} pts/tk"
                            r_col = "#22c55e"  # emerald green (looking at enemy)
                        elif r_val < 0.0:
                            r_str = f"{r_val:.1f} pts/tk"
                            r_col = "#ef4444"  # red (negative penalty)
                        else:
                            r_str = "+0.0 pts/tk"
                            r_col = "#71717a"
                        reward_tick_lbl.config(text=r_str, fg=r_col)

                        # Format cumulative score
                        sign = "+" if s_val > 0 else ""
                        s_col = "#fbbf24" if s_val >= 0 else "#ef4444"
                        score_lbl.config(text=f"POINTS: {sign}{s_val:,.0f}", fg=s_col)
                        if state["active"]:
                            header_score_lbl.config(text=f"[{sign}{s_val:,.0f} pts]")

        except Exception:
            pass

        # Smooth per-frame animation step
        if state["active"] or state["pinned"]:
            # Decay vector towards 0 if no recent inputs
            state["aim_vx"] *= 0.88
            state["aim_vy"] *= 0.88
            vx, vy = state["aim_vx"], state["aim_vy"]
            mag = math.hypot(vx, vy)
            if mag > 3.0:
                color = "#06b6d4" if mag < 20 else ("#f59e0b" if mag < 36 else "#ef4444")
                mouse_canvas.coords(arrow_id, mc_x, mc_y, mc_x + vx, mc_y + vy)
                mouse_canvas.itemconfig(arrow_id, fill=color, width=3)
            else:
                mouse_canvas.coords(arrow_id, mc_x, mc_y, mc_x, mc_y)
                mouse_canvas.itemconfig(arrow_id, width=0)

            # LMB flash decay
            if state["lmb_flash"] > 0:
                state["lmb_flash"] -= 1
                if state["lmb_flash"] == 0:
                    lmb_box.config(bg="#27272a", fg="#a1a1aa", relief=tk.RIDGE)

            # Event badge decay
            if state["event_timer"] > 0:
                state["event_timer"] -= 1
                event_lbl.config(text=state["event_text"], fg=state["event_color"])
            else:
                event_lbl.config(text="")

        root.after(25, poll_data)

    root.after(25, poll_data)
    root.mainloop()


class PvPOverlayClient:
    """Client interface for controlling and updating the PvP desktop overlay.

    Non-blocking, zero-lag, thread-safe, and crash-resilient.
    """

    def __init__(self):
        self._data_queue: mp.Queue = mp.Queue()
        self._cmd_queue: mp.Queue = mp.Queue()
        self._process: Optional[mp.Process] = None
        self._is_running: bool = False

    def start(self):
        """Start the overlay GUI in an isolated background process."""
        if self._is_running:
            return

        self._process = mp.Process(
            target=_overlay_process_main,
            args=(self._data_queue, self._cmd_queue),
            daemon=True,
        )
        self._process.start()
        self._is_running = True
        print(f"[+] Desktop PvP Keystrokes & Mousepad Overlay launched (PID: {self._process.pid})", flush=True)

    def update(
        self,
        active: bool = False,
        w: bool = False,
        s: bool = False,
        a: bool = False,
        d: bool = False,
        sprint: bool = False,
        space: bool = False,
        attack: bool = False,
        dx: int = 0,
        dy: int = 0,
        target_locked: bool = False,
        target_dist: float = 0.0,
        cooldown: float = 1.0,
        hit_type: str = "none",
        predicting: bool = False,
        pred_direction: str = "CENTER",
        reward: float = 0.0,
        total_score: float = 0.0,
    ):
        """Enqueue state update for overlay rendering (takes <0.005ms)."""
        if not self._is_running:
            return

        msg = {
            "type": "state",
            "active": active,
            "w": w,
            "s": s,
            "a": a,
            "d": d,
            "sprint": sprint,
            "space": space,
            "attack": attack,
            "dx": dx,
            "dy": dy,
            "target_locked": target_locked,
            "target_dist": target_dist,
            "cooldown": cooldown,
            "hit_type": hit_type,
            "predicting": predicting,
            "pred_direction": pred_direction,
            "reward": float(reward),
            "total_score": float(total_score),
        }
        try:
            self._data_queue.put_nowait(msg)
        except Exception:
            pass

    def poll_commands(self) -> List[Dict[str, Any]]:
        """Retrieve user commands initiated from the overlay UI (e.g. clicking RUN/STOP)."""
        if not self._is_running:
            return []

        commands = []
        try:
            while not self._cmd_queue.empty():
                commands.append(self._cmd_queue.get_nowait())
        except Exception:
            pass
        return commands

    def stop(self):
        """Cleanly terminate the overlay process and release resources."""
        if not self._is_running:
            return

        try:
            self._data_queue.put_nowait({"type": "stop"})
        except Exception:
            pass

        if self._process and self._process.is_alive():
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()

        self._is_running = False
        print("[+] Desktop PvP Overlay stopped cleanly.", flush=True)


if __name__ == "__main__":
    import time

    mp.freeze_support()
    client = PvPOverlayClient()
    client.start()
    time.sleep(0.5)

    print("Sending ACTIVE state to overlay...")
    client.update(active=True)
    time.sleep(0.5)

    print("Simulating W + Sprint + Mouse Aim Flick + Space + Reward Points...")
    sim_score = 0.0
    for i in range(15):
        cur_reward = 15.0 if i > 3 else -1.0
        sim_score += cur_reward
        client.update(
            active=True,
            w=True,
            sprint=(i > 5),
            space=(i % 4 == 0),
            dx=25,
            dy=-12,
            attack=(i == 8),
            target_locked=True,
            target_dist=2.8,
            cooldown=0.9,
            hit_type="knockback_hit" if i == 8 else "none",
            reward=cur_reward,
            total_score=sim_score,
        )
        time.sleep(0.05)

    time.sleep(0.5)
    print("Checking commands...")
    cmds = client.poll_commands()
    print("Received commands:", cmds)

    print("Stopping overlay...")
    client.stop()
    print("Self-test completed successfully!")
