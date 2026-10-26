"""Raw access to the Ultraleap Leap Motion Controller 2 over plain UVC (V4L2).

No Ultraleap SDK needed. The device streams "YUYV" frames whose bytes are really
two 8-bit IR images side by side (left | right). The right sensor is mounted
upside down; the factory calibration (read from a UVC extension unit) already
contains that 180 degree rotation, so frames are used as-is.
"""

from __future__ import annotations

import ctypes
import fcntl
import glob
import os
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# UVC extension unit holding the factory stereo calibration (156 bytes, starts with b"CA").
CALIB_XU_UNIT = 224
CALIB_XU_SELECTOR = 25
CALIB_XU_LEN = 156

RESOLUTIONS = (256, 512, 1024)


class _XuQuery(ctypes.Structure):
    _fields_ = [
        ("unit", ctypes.c_uint8),
        ("selector", ctypes.c_uint8),
        ("query", ctypes.c_uint8),
        ("size", ctypes.c_uint16),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


def _iowr(t: str, nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (ord(t) << 8) | nr


_UVCIOC_CTRL_QUERY = _iowr("u", 0x21, ctypes.sizeof(_XuQuery))
_UVC_GET_CUR = 0x81


def find_leap_device() -> str:
    """Return /dev/videoN of the Ultraleap capture node."""
    for sysdir in sorted(glob.glob("/sys/class/video4linux/video*")):
        name = Path(sysdir, "name").read_text().strip()
        index = Path(sysdir, "index").read_text().strip() if Path(sysdir, "index").exists() else "0"
        if "ultraleap" in name.lower() and index == "0":
            return "/dev/" + os.path.basename(sysdir)
    raise RuntimeError("No Ultraleap camera found (is the Leap 2 plugged in?)")


def read_calibration_blob(dev: str) -> bytes:
    """Read the raw factory calibration via a read-only UVC GET_CUR query."""
    fd = os.open(dev, os.O_RDWR)
    try:
        buf = (ctypes.c_uint8 * CALIB_XU_LEN)()
        q = _XuQuery(CALIB_XU_UNIT, CALIB_XU_SELECTOR, _UVC_GET_CUR, CALIB_XU_LEN,
                     ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)))
        fcntl.ioctl(fd, _UVCIOC_CTRL_QUERY, q)
        blob = bytes(buf)
    finally:
        os.close(fd)
    if blob[:2] != b"CA":
        raise RuntimeError(f"Unexpected calibration header {blob[:4]!r}")
    return blob


def device_serial(dev: str) -> str:
    """USB serial string of the device (used to name cached calibration files)."""
    sysdir = Path("/sys/class/video4linux", os.path.basename(dev), "device").resolve().parent
    p = sysdir / "serial"
    return p.read_text().strip() if p.exists() else "unknown"


class LeapCamera:
    """Threaded grabber that always holds the newest stereo pair."""

    def __init__(self, dev: str | None = None, resolution: int = 512):
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {RESOLUTIONS}")
        self.dev = dev or find_leap_device()
        self.res = resolution
        self.cap = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {self.dev}")
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame: tuple[np.ndarray, np.ndarray] | None = None
        self._stamp = 0.0
        self._seq = 0
        self._running = True
        self.fps = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        n, t0 = 0, time.time()
        while self._running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            f = f.reshape(self.res, 2 * self.res)
            left = f[:, : self.res].copy()
            right = f[:, self.res :].copy()
            # The last row of the left image carries embedded metadata bytes.
            left[-1] = left[-2]
            right[-1] = right[-2]
            with self._lock:
                self._frame = (left, right)
                self._stamp = time.time()
                self._seq += 1
            n += 1
            if time.time() - t0 > 1.0:
                self.fps = n / (time.time() - t0)
                n, t0 = 0, time.time()

    def read(self, last_seq: int = -1, timeout: float = 1.0):
        """Return (seq, stamp, left, right); waits for a frame newer than last_seq."""
        t_end = time.time() + timeout
        while time.time() < t_end:
            with self._lock:
                if self._frame is not None and self._seq != last_seq:
                    return self._seq, self._stamp, self._frame[0], self._frame[1]
            time.sleep(0.002)
        raise TimeoutError("No frame from Leap camera")

    def set_gain(self, gain: int):
        self.cap.set(cv2.CAP_PROP_GAIN, int(np.clip(gain, 0, 15)))

    def close(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


def parse_blob_floats(blob: bytes) -> np.ndarray:
    return np.array(struct.unpack("<37f", blob[8:156]), np.float64)
