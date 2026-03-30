"""
audio_feedback.py
-----------------
Thread-safe TTS audio feedback using pyttsx3.
Fixes:
  - Windows COM threading conflict (WinError -2147417850)
  - Audio repeating correctly after first trigger
"""

import threading
import time
import pyttsx3
import pythoncom


class AudioFeedback:
    def __init__(self, cooldown_sec: float = 2.0, stable_sec: float = 0.6):
        self.cooldown_sec  = cooldown_sec
        self.stable_sec    = stable_sec

        self._engine       = None
        self._lock         = threading.Lock()
        self._last_spoken  = 0.0
        self._stable_start = None

        self._tts_queue    = []
        self._route_thread = None
        self._route_cancel = threading.Event()

        # Dedicated TTS thread — owns the COM engine on Windows
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()
        time.sleep(0.5)   # let engine initialise

    # ── Public API ─────────────────────────────────────────────────────────────

    def speak(self, text: str, force: bool = False):
        """Speak text if cooldown has elapsed or force=True."""
        now = time.time()
        if not force and (now - self._last_spoken) < self.cooldown_sec:
            return
        self._say(text)

    def speak_hazard(self, dist_m, direction, hazard_now, force_clear=False):
        """
        Rate-limited hazard announcement.
        Resets stable_start after each announcement so it can trigger again.
        """
        now = time.time()

        if hazard_now:
            # Wait for stable detection before first announcement
            if self._stable_start is None:
                self._stable_start = now
                return
            if (now - self._stable_start) < self.stable_sec:
                return
            # Respect cooldown between announcements
            if (now - self._last_spoken) < self.cooldown_sec:
                return
            self._say(f"Obstacle. {dist_m:.1f} metres. {direction}.")
            self._stable_start = None   # ← reset so it can trigger again next detection
        elif force_clear:
            self._stable_start = None
            self._say("Path clear.")

    def speak_route(self, instructions):
        """
        Speak escape route instructions sequentially in a background thread.
        Cancels any previously running narration before starting new one.
        """
        self._cancel_route()
        self._route_cancel.clear()
        self._route_thread = threading.Thread(
            target=self._route_worker, args=(instructions,), daemon=True
        )
        self._route_thread.start()

    def cancel_route(self):
        self._cancel_route()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _tts_worker(self):
        """
        Dedicated TTS thread.
        CoInitialize() must be called here so pyttsx3's sapi5 driver
        works without the Windows COM thread-mode conflict.
        """
        pythoncom.CoInitialize()
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        self._engine = engine

        while True:
            with self._lock:
                pending = self._tts_queue[:]
                self._tts_queue.clear()

            for text in pending:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"[Audio] TTS error: {e}")

            time.sleep(0.05)

    def _say(self, text: str):
        self._last_spoken = time.time()
        with self._lock:
            self._tts_queue.append(text)

    def _route_worker(self, instructions):
        for step in instructions:
            if self._route_cancel.is_set():
                break
            self._say(step)
            time.sleep(1.5)   # pause between steps so speech finishes

    def _cancel_route(self):
        if self._route_thread and self._route_thread.is_alive():
            self._route_cancel.set()
            self._route_thread.join(timeout=2.0)