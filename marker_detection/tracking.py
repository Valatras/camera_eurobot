"""Filtrage temporel simple des detections."""

from __future__ import annotations

from typing import Any

import numpy as np


class Tracker:
    """Lisse les detections sur quelques frames et supprime les intermittences."""

    def __init__(self, buffer_size: int = 3, min_hits: int = 2) -> None:
        self.buf: dict[Any, list[np.ndarray]] = {}
        self.hits: dict[Any, int] = {}
        self.buf_size = buffer_size
        self.min_hits = min_hits

    def update(self, keys: list[Any], values: list[np.ndarray]) -> tuple[list[Any], list[np.ndarray]]:
        """Met a jour l'historique et renvoie uniquement les detections stables."""
        current = dict(zip(keys, values))

        for key, val in current.items():
            if key not in self.buf:
                self.buf[key] = []
                self.hits[key] = 0
            self.buf[key].append(val)
            self.hits[key] += 1
            if len(self.buf[key]) > self.buf_size:
                self.buf[key].pop(0)

        for key in list(self.buf):
            if key not in current:
                self.hits[key] -= 1
                if self.hits[key] <= 0:
                    del self.buf[key]
                    del self.hits[key]

        out_keys: list[Any] = []
        out_vals: list[np.ndarray] = []
        for key, history in self.buf.items():
            if len(history) >= self.min_hits:
                out_keys.append(key)
                out_vals.append(np.mean(history, axis=0))

        return out_keys, out_vals
