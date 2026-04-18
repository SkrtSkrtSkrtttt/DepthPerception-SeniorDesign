"""
audio_feedback.py  (priority queue)
-------------------------------------
Priority levels — lower number = higher priority, always interrupts lower:

  1  DANGER    fire / smoke detected          → interrupts everything
  2  OBSTACLE  something within 0.8m          → interrupts route narration
  3  ROUTE     escape route step-by-step      → plays in gaps
  4  STATUS    exit found, path clear, etc.   → lowest priority

How it works:
  - A single background TTS thread owns the COM engine (Windows fix)
  - A priority queue holds pending messages
  - When a higher-priority message arrives, the current speech is
    interrupted and the new message plays immediately
  - Route narration is a sequence — each step only plays if no higher
    priority message is waiting
  - Cooldowns are tracked per priority level to avoid spam
"""

import threading
import time
import queue
import pythoncom
import pyttsx3
from dataclasses import dataclass, field
from typing import Optional, List

# ── Priority levels ───────────────────────────────────────────────────────────
PRIORITY_DANGER   = 1
PRIORITY_OBSTACLE = 2
PRIORITY_ROUTE    = 3
PRIORITY_STATUS   = 4

# ── Cooldowns per priority (seconds) ─────────────────────────────────────────
COOLDOWNS = {
    PRIORITY_DANGER:   3.0,   # fire alert — don't repeat for 3s
    PRIORITY_OBSTACLE: 2.5,   # obstacle — don't repeat for 2.5s
    PRIORITY_ROUTE:    0.0,   # route steps — no cooldown, managed by sequence
    PRIORITY_STATUS:   5.0,   # status — don't repeat for 5s
}

# ── Stable detection window for obstacles ────────────────────────────────────
OBSTACLE_STABLE_SEC = 0.5


@dataclass(order=True)
class AudioMessage:
    priority:  int
    text:      str = field(compare=False)
    timestamp: float = field(compare=False, default_factory=time.time)


class AudioFeedback:
    def __init__(self):
        self._queue          = queue.PriorityQueue()
        self._lock           = threading.Lock()
        self._last_spoken    = {}       # priority → last spoken time
        self._engine         = None
        self._current_prio   = 99      # priority of currently playing speech
        self._stop_flag      = threading.Event()

        # Obstacle stable detection
        self._obstacle_start = None

        # Route narration state
        self._route_steps:   List[str] = []
        self._route_idx:     int = 0
        self._route_active:  bool = False
        self._route_repeat:  bool = True
        self._route_lock     = threading.Lock()

        # Start TTS worker thread
        self._tts_thread = threading.Thread(
            target=self._tts_worker, daemon=True
        )
        self._tts_thread.start()
        time.sleep(0.6)   # let engine init

        print("[Audio] Priority queue system ready.")
        print(f"[Audio] Priorities: DANGER={PRIORITY_DANGER} "
              f"OBSTACLE={PRIORITY_OBSTACLE} "
              f"ROUTE={PRIORITY_ROUTE} "
              f"STATUS={PRIORITY_STATUS}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def alert_fire(self, label: str = "fire", direction: str = ""):
        """Highest priority — always interrupts everything."""
        dir_str = f" {direction}" if direction else ""
        msg = f"Warning! {label} detected{dir_str}. Follow the escape route immediately."
        self._enqueue(msg, PRIORITY_DANGER, interrupt=True)

    def alert_smoke(self, direction: str = ""):
        """Danger level — smoke warning."""
        dir_str = f" {direction}" if direction else ""
        msg = f"Warning! Smoke detected{dir_str}. Evacuate now."
        self._enqueue(msg, PRIORITY_DANGER, interrupt=True)

    def speak_hazard(self, dist_m, direction, hazard_now, force_clear=False):
        """Rate-limited obstacle warning."""
        now = time.time()

        if hazard_now:
            if self._obstacle_start is None:
                self._obstacle_start = now
                return
            if (now - self._obstacle_start) < OBSTACLE_STABLE_SEC:
                return
            last = self._last_spoken.get(PRIORITY_OBSTACLE, 0)
            if (now - last) < COOLDOWNS[PRIORITY_OBSTACLE]:
                return
            msg = f"Obstacle. {dist_m:.1f} metres. {direction}."
            self._enqueue(msg, PRIORITY_OBSTACLE, interrupt=False)
            self._obstacle_start = None
        elif force_clear:
            self._obstacle_start = None
            self._enqueue("Path clear.", PRIORITY_STATUS, interrupt=False)

    def speak_route(self, instructions: List[str], repeat: bool = True):
        """
        Set a new escape route sequence.
        Cancels any current route and starts the new one.
        Route steps play only when no higher-priority audio is pending.
        If repeat=True, the sequence loops until cancel_route() is called.
        """
        with self._route_lock:
            self._route_steps  = [s for s in instructions if s]
            self._route_idx    = 0
            self._route_active = True
            self._route_repeat = repeat

    def cancel_route(self):
        with self._route_lock:
            self._route_active = False
            self._route_steps  = []
            self._route_idx    = 0
            self._route_repeat = False

    def speak_status(self, text: str):
        """Low priority status message — exit found, path clear, etc."""
        self._enqueue(text, PRIORITY_STATUS, interrupt=False)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _enqueue(self, text: str, priority: int, interrupt: bool):
        """Add message to queue. If interrupt=True and priority is higher than
        current speech, signal the engine to stop mid-sentence."""
        now = time.time()
        last = self._last_spoken.get(priority, 0)
        cooldown = COOLDOWNS.get(priority, 2.0)

        if cooldown > 0 and (now - last) < cooldown:
            return   # still in cooldown

        msg = AudioMessage(priority=priority, text=text)
        self._queue.put(msg)

        if interrupt and priority < self._current_prio:
            self._stop_flag.set()   # interrupt current speech

    def _tts_worker(self):
        """
        Dedicated TTS thread — owns the COM engine on Windows.
        Continuously drains the priority queue.
        Between queue items, advances route narration if active.
        """
        pythoncom.CoInitialize()
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        self._engine = engine

        while True:
            # ── Check priority queue first ───────────────────────────────────
            try:
                msg = self._queue.get_nowait()
                self._current_prio = msg.priority
                self._stop_flag.clear()
                self._last_spoken[msg.priority] = time.time()
                self._say(engine, msg.text)
                self._current_prio = 99
                self._queue.task_done()
                continue
            except queue.Empty:
                pass

            # ── Advance route narration if active and queue is empty ─────────
            with self._route_lock:
                if (self._route_active and
                        self._route_steps and
                        self._route_idx < len(self._route_steps)):
                    step = self._route_steps[self._route_idx]
                    self._route_idx += 1
                    if self._route_idx >= len(self._route_steps):
                        self._route_active = False
                else:
                    step = None

            if step:
                self._current_prio = PRIORITY_ROUTE
                self._stop_flag.clear()
                self._say(engine, step)
                self._current_prio = 99
                time.sleep(0.3)   # brief breath between steps
            else:
                # If repeat is on and we finished the sequence, pause then restart
                with self._route_lock:
                    should_restart = (
                        self._route_repeat and
                        self._route_steps and
                        not self._route_active
                    )
                if should_restart:
                    time.sleep(4.0)   # 4 second gap between repeats
                    with self._route_lock:
                        self._route_idx    = 0
                        self._route_active = True
                else:
                    time.sleep(0.05)

    def _say(self, engine, text: str):
        """Speak text. Checks stop_flag between sentences for interruptibility."""
        if self._stop_flag.is_set():
            return   # already interrupted before we started

        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[Audio] TTS error: {e}")