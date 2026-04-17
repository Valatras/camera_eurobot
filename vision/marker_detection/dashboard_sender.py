"""Envoi des detections vers le dashboard web via Socket.IO."""

from __future__ import annotations

import logging
import time

import socketio

logger = logging.getLogger(__name__)


class DashboardSender:
    """Envoie les detections de marqueurs au dashboard (Socket.IO)."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._sio = socketio.Client(reconnection=True, logger=False)
        self._connected = False
        self._last_error_log = 0.0

        self._sio.on("connect", self._on_connect)
        self._sio.on("disconnect", self._on_disconnect)

    def _on_connect(self) -> None:
        print("[Dashboard] Socket.IO connecte.")
        self._connected = True

    def _on_disconnect(self) -> None:
        print("[Dashboard] Socket.IO deconnecte.")
        self._connected = False

    def connect(self) -> bool:
        try:
            self._sio.connect(self._url, wait_timeout=3)
            return True
        except socketio.exceptions.ConnectionError as exc:
            print(f"[Dashboard] Connexion echouee: {exc}")
            return False

    def disconnect(self) -> None:
        if self._connected:
            self._sio.disconnect()

    def send_detections(
        self,
        detected: list[tuple[str, float, float, float]],
        corners_ok: bool,
        zone_sequences: dict[int, list[str]] | None = None,
        gm_counts: dict[int, int] | None = None,
        stale: bool = False,
    ) -> bool:
        """Envoie les detections categorisees au dashboard.

        Args:
            detected: Liste de tuples (label, x_cm, y_cm, angle_deg).
            corners_ok: True si les 4 coins de table sont detectes.
            zone_sequences: Sequences de noisettes par zone de ramassage.
            gm_counts: Comptage de noisettes par garde-manger.
            stale: True si la homographie est perimee (cache).
        """
        if not self._connected:
            now = time.time()
            if now - self._last_error_log > 5.0:
                logger.warning("Dashboard non connecte — detections ignorees.")
                self._last_error_log = now
            return False

        nuts: list[dict[str, float | str]] = []
        robots: list[dict[str, float | str]] = []
        opponents: list[dict[str, float | str]] = []

        from marker_detection import config
        print(detected)

        for label, x, y, angle in detected:
            entry = {
                "label": label,
                "x": round(x, 1),
                "y": round(y, 1),
                "angle": round(angle, 1),
            }
            if label.startswith("NUT_"):
                nuts.append(entry)
            elif label.startswith("BR"):
                if config.TEAM_COLOR == "blue":
                    robots.append(entry)
                else:
                    opponents.append(entry)
            elif label.startswith("YR"):
                if config.TEAM_COLOR == "yellow":
                    robots.append(entry)
                else:
                    opponents.append(entry)

        payload = {
            "nuts": nuts,
            "robots": robots,
            "us": robots,
            "opponents": opponents,
            "corners_ok": corners_ok,
            "stale": bool(stale),
            "zones": {
                str(int(zid)): list(seq)
                for zid, seq in (zone_sequences or {}).items()
            },
            "gms": {
                str(int(gid)): int(count)
                for gid, count in (gm_counts or {}).items()
            },
            "ts_ms": int(time.time() * 1000),
        }

        try:
            self._sio.emit("visionDetections", payload)
            return True
        except Exception as exc:
            now = time.time()
            if now - self._last_error_log > 5.0:
                logger.warning("Dashboard emit failed: %s", exc)
                self._last_error_log = now
            return False
