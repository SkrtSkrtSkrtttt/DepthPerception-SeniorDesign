"""
thermal_receiver.py
-------------------
Background TCP receiver for the ESP32 FLIR Lepton thermal stream.

Responsibilities
----------------
- Listen for ESP32 thermal packets over Wi-Fi/TCP
- Parse the current ESP32 packet/header format
- Convert TLinear counts to Celsius/Fahrenheit when available
- Store the latest thermal state in a thread-safe way
- Optionally keep the latest full thermal frame for future fusion/overlay work

Designed to be imported by main.py later.
This file does NOT create any windows and does NOT depend on RealSense.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ThermalState:
    connected: bool = False
    frame_id: int = -1
    width: int = 0
    height: int = 0

    max_value: Optional[int] = None
    hot_threshold_value: Optional[int] = None
    hot_from_esp32: bool = False

    max_temp_c: Optional[float] = None
    max_temp_f: Optional[float] = None
    threshold_temp_c: Optional[float] = None
    threshold_temp_f: Optional[float] = None

    above_user_threshold: bool = False
    user_threshold_f: float = 95.0

    test_pattern_enabled: bool = False
    high_gain_enabled: bool = False
    tlinear_enabled: bool = False
    tlinear_0_01k_enabled: bool = False

    frame_min: Optional[int] = None
    frame_max: Optional[int] = None

    last_update_time: float = 0.0
    peer_ip: Optional[str] = None
    peer_port: Optional[int] = None
    error: Optional[str] = None


class ThermalReceiver:
    """
    Threaded background receiver for the ESP32 thermal stream.

    Usage
    -----
    thermal = ThermalReceiver(host="0.0.0.0", port=5001, alert_threshold_f=95.0)
    thermal.start()

    # later in your main loop:
    state = thermal.get_latest_state()
    if state.above_user_threshold:
        ...

    # optional:
    frame = thermal.get_latest_frame()

    # shutdown:
    thermal.stop()
    """

    MAGIC = 0x4D524854  # "THRM"

    # ESP32 header layout:
    # uint32_t magic;
    # uint16_t width;
    # uint16_t height;
    # uint32_t frame_id;
    # uint32_t payload_bytes;
    # uint16_t max_value;
    # uint16_t hot_threshold;
    # uint8_t  hot_flag;
    # uint8_t  mode_flags;
    # uint16_t reserved;
    HEADER_FORMAT = "<IHHIIHHBBH"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5001,
        alert_threshold_f: float = 95.0,
        expected_width: int = 80,
        expected_height: int = 60,
        keep_latest_frame: bool = True,
        verbose: bool = False,
    ):
        self.host = host
        self.port = port
        self.alert_threshold_f = alert_threshold_f
        self.expected_width = expected_width
        self.expected_height = expected_height
        self.keep_latest_frame = keep_latest_frame
        self.verbose = verbose

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._server_socket: Optional[socket.socket] = None

        self._lock = threading.Lock()
        self._state = ThermalState(user_threshold_f=alert_threshold_f)
        self._latest_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        with self._lock:
            self._state.connected = False

    def get_latest_state(self) -> ThermalState:
        with self._lock:
            return ThermalState(**self._state.__dict__)

    def get_latest_frame(self, copy: bool = True) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy() if copy else self._latest_frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[ThermalReceiver] {msg}")

    def _recv_exact(self, conn: socket.socket, size: int) -> Optional[bytes]:
        data = b""
        while len(data) < size and self._running:
            try:
                chunk = conn.recv(size - len(data))
            except socket.timeout:
                continue
            except OSError:
                return None

            if not chunk:
                return None
            data += chunk
        return data if len(data) == size else None

    @staticmethod
    def _decode_mode_flags(mode_flags: int) -> dict:
        return {
            "test_pattern_enabled": bool(mode_flags & 0x01),
            "high_gain_enabled": bool(mode_flags & 0x02),
            "tlinear_enabled": bool(mode_flags & 0x04),
            "tlinear_0_01k_enabled": bool(mode_flags & 0x08),
        }

    @staticmethod
    def _counts_to_celsius(
        value: int,
        tlinear_enabled: bool,
        tlinear_0_01k_enabled: bool,
    ) -> Optional[float]:
        if not tlinear_enabled:
            return None
        if tlinear_0_01k_enabled:
            return (value / 100.0) - 273.15
        return (value / 10.0) - 273.15

    @staticmethod
    def _celsius_to_fahrenheit(temp_c: float) -> float:
        return temp_c * 9.0 / 5.0 + 32.0

    def _update_disconnected(self, error: Optional[str] = None) -> None:
        with self._lock:
            self._state.connected = False
            self._state.peer_ip = None
            self._state.peer_port = None
            self._state.error = error

    def _update_state(
        self,
        addr,
        width: int,
        height: int,
        frame_id: int,
        max_value: int,
        hot_threshold: int,
        hot_flag: int,
        mode_flags: int,
        frame: np.ndarray,
    ) -> None:
        flags = self._decode_mode_flags(mode_flags)

        max_temp_c = self._counts_to_celsius(
            max_value,
            flags["tlinear_enabled"],
            flags["tlinear_0_01k_enabled"],
        )
        threshold_temp_c = self._counts_to_celsius(
            hot_threshold,
            flags["tlinear_enabled"],
            flags["tlinear_0_01k_enabled"],
        )

        max_temp_f = (
            None if max_temp_c is None else self._celsius_to_fahrenheit(max_temp_c)
        )
        threshold_temp_f = (
            None
            if threshold_temp_c is None
            else self._celsius_to_fahrenheit(threshold_temp_c)
        )

        above_user_threshold = (
            max_temp_f is not None and max_temp_f > self.alert_threshold_f
        )

        with self._lock:
            self._state.connected = True
            self._state.frame_id = frame_id
            self._state.width = width
            self._state.height = height

            self._state.max_value = max_value
            self._state.hot_threshold_value = hot_threshold
            self._state.hot_from_esp32 = bool(hot_flag)

            self._state.max_temp_c = max_temp_c
            self._state.max_temp_f = max_temp_f
            self._state.threshold_temp_c = threshold_temp_c
            self._state.threshold_temp_f = threshold_temp_f

            self._state.above_user_threshold = above_user_threshold
            self._state.user_threshold_f = self.alert_threshold_f

            self._state.test_pattern_enabled = flags["test_pattern_enabled"]
            self._state.high_gain_enabled = flags["high_gain_enabled"]
            self._state.tlinear_enabled = flags["tlinear_enabled"]
            self._state.tlinear_0_01k_enabled = flags["tlinear_0_01k_enabled"]

            self._state.frame_min = int(frame.min())
            self._state.frame_max = int(frame.max())

            self._state.last_update_time = time.time()
            self._state.peer_ip = addr[0]
            self._state.peer_port = addr[1]
            self._state.error = None

            if self.keep_latest_frame:
                self._latest_frame = frame.copy()

    def _run(self) -> None:
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            server.settimeout(1.0)
            self._server_socket = server
            self._log(f"Listening on {self.host}:{self.port}")
        except Exception as e:
            self._update_disconnected(error=f"Server bind/listen failed: {e}")
            return

        while self._running:
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            self._log(f"Connected by {addr}")

            try:
                conn.settimeout(1.0)

                while self._running:
                    header_bytes = self._recv_exact(conn, self.HEADER_SIZE)
                    if header_bytes is None:
                        self._log("Connection closed while reading header")
                        break

                    (
                        magic,
                        width,
                        height,
                        frame_id,
                        payload_bytes,
                        max_value,
                        hot_threshold,
                        hot_flag,
                        mode_flags,
                        reserved,
                    ) = struct.unpack(self.HEADER_FORMAT, header_bytes)

                    if magic != self.MAGIC:
                        self._update_disconnected(
                            error=f"Bad magic: 0x{magic:08X}"
                        )
                        self._log(f"Bad magic: 0x{magic:08X}")
                        break

                    if width != self.expected_width or height != self.expected_height:
                        self._update_disconnected(
                            error=f"Unexpected frame size: {width}x{height}"
                        )
                        self._log(f"Unexpected frame size: {width}x{height}")
                        break

                    expected_payload = width * height * 2
                    if payload_bytes != expected_payload:
                        self._update_disconnected(
                            error=(
                                f"Bad payload size: got {payload_bytes}, "
                                f"expected {expected_payload}"
                            )
                        )
                        self._log(
                            f"Bad payload size: got {payload_bytes}, "
                            f"expected {expected_payload}"
                        )
                        break

                    payload = self._recv_exact(conn, payload_bytes)
                    if payload is None:
                        self._log("Connection closed while reading payload")
                        break

                    frame = np.frombuffer(payload, dtype=np.uint16).reshape(
                        (height, width)
                    )

                    self._update_state(
                        addr=addr,
                        width=width,
                        height=height,
                        frame_id=frame_id,
                        max_value=max_value,
                        hot_threshold=hot_threshold,
                        hot_flag=hot_flag,
                        mode_flags=mode_flags,
                        frame=frame,
                    )

                    if self.verbose:
                        state = self.get_latest_state()
                        msg = (
                            f"Frame {state.frame_id} | "
                            f"max_value={state.max_value} | "
                            f"hot_from_esp32={state.hot_from_esp32} | "
                            f"high_gain={state.high_gain_enabled} | "
                            f"tlinear={state.tlinear_enabled}"
                        )
                        if state.max_temp_f is not None:
                            msg += f" | max_temp_f={state.max_temp_f:.2f}"
                        self._log(msg)

            except Exception as e:
                self._update_disconnected(error=str(e))
                self._log(f"Receiver exception: {e}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                self._update_disconnected()

        try:
            server.close()
        except Exception:
            pass
        self._log("Stopped")


if __name__ == "__main__":
    """
    Optional standalone debug mode.
    This is only for testing the thermal receiver by itself.
    """
    receiver = ThermalReceiver(verbose=True)
    receiver.start()

    print("ThermalReceiver running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
            state = receiver.get_latest_state()
            print(state)
    except KeyboardInterrupt:
        print("\nStopping ThermalReceiver...")
    finally:
        receiver.stop()