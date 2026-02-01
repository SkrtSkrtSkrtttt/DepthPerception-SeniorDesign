import time
import subprocess
from typing import Optional


class AudioFeedback:
    """
    According to stackoverflow: we should use the following call:
    Windows TTS via PowerShell (System.Speech), with:
    - rate limiting
    - quantized distance (reduces chatter)
    - debounce (only speaks when state is stable briefly)
    - avoids stacking voices
    """

    def __init__(self, cooldown_sec: float = 2.0, stable_sec: float = 0.6):
        self.cooldown = cooldown_sec
        self.stable_sec = stable_sec

        self.last_spoken_time = 0.0
        self.last_spoken_message = ""
        self._proc = None

        # For debounce
        self._candidate_message = ""
        self._candidate_since = 0.0

    @staticmethod
    def _escape_for_powershell(text: str) -> str:
        return text.replace("'", "''")

    def _speak_raw(self, message: str, force: bool = False) -> None:
        now = time.time()

        if not force:
            if message == self.last_spoken_message and (now - self.last_spoken_time) < self.cooldown:
                return
            if (now - self.last_spoken_time) < self.cooldown:
                return

        # Don't stack voices
        if self._proc is not None and self._proc.poll() is None and not force:
            return

        self.last_spoken_message = message
        self.last_spoken_time = now

        safe = self._escape_for_powershell(message)
        ps_cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Speak('{safe}');"
        )

        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _quantize_distance_m(dist_m: float) -> float:
        # Round to nearest 0.25m so audio doesn't chatter
        step = 0.25
        return round(dist_m / step) * step

    def speak_hazard(self, dist_m: Optional[float], direction: str, hazard_now: bool, force_clear: bool = False) -> None:
        """
        High-level call:
        - If hazard_now True, announce "Near hazard < insert dist> meters <direction>"
          only if the message stays stable for stable_sec.
        - If hazard_now False and force_clear True, announce "Clear."
        """
        now = time.time()

        if not hazard_now:
            if force_clear:
                self._candidate_message = ""
                self._candidate_since = 0.0
                self._speak_raw("Clear.", force=True)
            return

        if dist_m is None:
            return

        q = self._quantize_distance_m(dist_m)
        msg = f"Near hazard. {q:.2f} meters. {direction}."

        # Debounce: only speak after message remains the same for stable_sec
        if msg != self._candidate_message:
            self._candidate_message = msg
            self._candidate_since = now
            return

        if (now - self._candidate_since) >= self.stable_sec:
            self._speak_raw(msg, force=False)
