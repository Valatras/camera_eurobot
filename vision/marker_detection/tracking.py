"""Filtrage temporel simple des detections."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


class Tracker:
    """Lisse les detections sur quelques frames et supprime les intermittences.

    Supporte plusieurs marqueurs avec le meme ID grace a un systeme de slots
    associes par proximite spatiale.
    """

    def __init__(self, buffer_size: int = 3, min_hits: int = 2) -> None:
        self.buf: dict[Any, list[np.ndarray]] = {}
        self.hits: dict[Any, int] = {}
        self.buf_size = buffer_size
        self.min_hits = min_hits

    def update(self, keys: list[Any], values: list[np.ndarray]) -> tuple[list[Any], list[np.ndarray]]:
        """Met a jour l'historique et renvoie uniquement les detections stables."""
        current: dict[Any, np.ndarray] = {}

        grouped_vals: dict[Any, list[np.ndarray]] = defaultdict(list)
        for key, val in zip(keys, values):
            grouped_vals[key].append(val)

        # Associe les detections courantes a des "slots" stables pour
        # permettre plusieurs marqueurs avec le meme ID.
        for base_key, items in grouped_vals.items():
            existing_slots = [
                slot for (key, slot) in self.buf.keys()
                if isinstance(key, tuple) and len(key) == 2 and key[0] == base_key
            ]

            if not existing_slots:
                for idx, val in enumerate(items):
                    current[(base_key, idx)] = val
                continue

            # Associe par proximite pour stabiliser les slots.
            existing_centers = []
            for slot in existing_slots:
                history = self.buf[(base_key, slot)]
                existing_centers.append(np.mean(history[-1][0], axis=0))

            used_slots: set[int] = set()
            for val in items:
                center = np.mean(val[0], axis=0)
                best_slot = None
                best_dist = None
                for slot, ref_center in zip(existing_slots, existing_centers):
                    if slot in used_slots:
                        continue
                    dist = float(np.linalg.norm(center - ref_center))
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_slot = slot
                if best_slot is None:
                    next_slot = max(existing_slots) + 1 if existing_slots else 0
                    best_slot = next_slot
                    existing_slots.append(best_slot)
                used_slots.add(best_slot)
                current[(base_key, best_slot)] = val

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
                if isinstance(key, tuple) and len(key) == 2:
                    out_keys.append(key[0])
                else:
                    out_keys.append(key)
                out_vals.append(np.mean(history, axis=0))

        return out_keys, out_vals
