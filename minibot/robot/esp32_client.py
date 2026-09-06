"""ESP32 transport (§4).

The only place in the application that knows the robot speaks HTTP. Every
method here maps 1:1 onto an endpoint in ai_mini_bot.ino and the wire contract
is unchanged from mac_realtime.py's `Bot`.

Added on top of the original: retries with backoff, rate limiting, connection
health tracking, and request serialization.

Serialization is not optional. The firmware runs `server.handleClient()` inside
`loop()`, so it services exactly one request at a time and a long handler
stalls every other endpoint — /mic alone can own the device for 15 seconds.
Issuing requests in parallel would queue them on the socket and time out rather
than gain anything.
"""

from __future__ import annotations

import threading
import time

import requests

from ..obs.logger import ROBOT

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "thinking",
            "listening", "talking", "sleep", "searching", "loading",
            "scanning", "wifi", "memory", "saving"]


class Esp32Unavailable(RuntimeError):
    """Raised when the robot cannot be reached after retries."""


class Esp32Client:
    # recordMic() multiplies samples by 16 on the way out of /mic while /level
    # reports raw ADC counts; audio/capture.py reads this to compare the two.
    wav_gain = 16

    def __init__(self, base: str, timeout: float = 8.0, retries: int = 2,
                 min_interval: float = 0.02, on_state_change=None):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.min_interval = min_interval
        self._s = requests.Session()
        self._lock = threading.RLock()
        self._last_request = 0.0
        self._online = True
        self._on_state_change = on_state_change
        self.stats = {"requests": 0, "failures": 0, "retries": 0}

    # -- health ----------------------------------------------------
    @property
    def online(self) -> bool:
        return self._online

    def _mark(self, online: bool) -> None:
        if online != self._online:
            self._online = online
            ROBOT.warn("esp32 " + ("reconnected" if online else "unreachable"))
            if self._on_state_change:
                self._on_state_change(online)

    # -- core request ----------------------------------------------
    def _request(self, method: str, path: str, *, params=None, data=None,
                 headers=None, timeout: float | None = None):
        """Serialized, rate limited, retried with backoff."""
        url = self.base + path
        timeout = timeout or self.timeout
        last: Exception | None = None

        with self._lock:
            for attempt in range(self.retries + 1):
                gap = time.monotonic() - self._last_request
                if gap < self.min_interval:
                    time.sleep(self.min_interval - gap)
                t0 = time.perf_counter()
                try:
                    r = self._s.request(method, url, params=params, data=data,
                                        headers=headers, timeout=timeout)
                    self._last_request = time.monotonic()
                    self.stats["requests"] += 1
                    r.raise_for_status()
                    ms = (time.perf_counter() - t0) * 1000
                    ROBOT.debug(f"{method} {path} -> {r.status_code} ({ms:.0f}ms)")
                    self._mark(True)
                    return r
                except requests.RequestException as e:
                    last = e
                    self._last_request = time.monotonic()
                    self.stats["failures"] += 1
                    status = getattr(e.response, "status_code", None)
                    # The robot answered, so the link is fine either way.
                    #
                    # 503 from this firmware always means a subsystem failed to
                    # initialise at boot — cameraReady / servosReady / micReady
                    # / audioReady are set once in setup() and never change. It
                    # will not recover in 300 ms, and retrying only burns the
                    # device's single request thread. 4xx is likewise final.
                    # A 500 ("capture failed", "no memory") genuinely can be a
                    # transient hiccup, so that one is still worth retrying.
                    if status is not None and (status < 500 or status == 503):
                        ROBOT.debug(f"{method} {path} -> {status}")
                        self._mark(True)
                        raise
                    if attempt < self.retries:
                        self.stats["retries"] += 1
                        time.sleep(0.15 * (2 ** attempt))

        self._mark(False)
        raise Esp32Unavailable(f"{method} {path}: {last}") from last

    def _get(self, path: str, **params):
        return self._request("GET", path, params=params or None)

    # -- endpoints (unchanged contract) ----------------------------
    def status(self) -> dict:
        return self._get("/status").json()

    def capture(self) -> bytes:
        return self._get("/capture").content

    def level(self) -> dict:
        return self._get("/level").json()

    def center(self) -> str:
        return self._get("/look/center").text

    def look_status(self) -> dict:
        return self._get("/look/status").json()

    def stop(self) -> str:
        return self._get("/stop").text

    def beep(self, f: int = 880, ms: int = 150) -> str:
        return self._get("/beep", f=f, ms=ms).text

    def volume(self, v: int) -> str:
        return self._get("/volume", v=int(v)).text

    def face(self, emotion: str) -> str:
        if emotion not in EMOTIONS:
            emotion = "neutral"
        return self._get("/set", e=emotion).text

    def look(self, pan: int | None = None, tilt: int | None = None):
        p = {}
        if pan is not None:
            p["pan"] = int(pan)
        if tilt is not None:
            p["tilt"] = int(tilt)
        return self._get("/look", **p).text if p else None

    def play(self, pcm16k: bytes):
        """Raw int16 LE mono at 16 kHz. Returns when the upload finishes, not
        when playback does."""
        if not pcm16k:
            return None
        r = self._request("POST", "/say", data=pcm16k,
                          headers={"Content-Type": "application/octet-stream"},
                          timeout=30)
        return r.json()

    def record(self, max_ms: int = 3500, silence_ms: int = 0, thresh: int = 0,
               lead_ms: int = 1500) -> bytes:
        """Fixed-length by default. With silence_ms the firmware endpoints the
        recording itself and returns as soon as you stop talking."""
        p = {"ms": int(max_ms)}
        if silence_ms:
            p.update(silence=int(silence_ms), thresh=int(thresh), lead=int(lead_ms))
        return self._request("GET", "/mic", params=p,
                             timeout=max_ms / 1000 + 10).content

    def probe_vad(self) -> bool:
        """True if the firmware endpoints recordings itself. Older builds just
        ignore the extra args and record the full span, so tell them apart by
        how much audio comes back: an unreachable threshold means nothing ever
        counts as speech, so a VAD build gives up after `lead` instead of `ms`."""
        from ..audio.dsp import wav_to_pcm
        try:
            wav = self.record(max_ms=3000, silence_ms=100, thresh=32000, lead_ms=200)
            pcm, rate = wav_to_pcm(wav)
        except Exception:
            return False
        return len(pcm) / 2 / rate < 1.0

    def wait_quiet(self, timeout: float = 30) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if not self.status().get("speaking"):
                    return
            except Exception:
                pass
            time.sleep(0.15)
