"""Lissage temporel EMA des positions (x, y, angle) en coordonnees table.

Le Tracker existant ([tracking.py](tracking.py)) filtre les intermittences
dans le repere pixel. Ce module complete en lissant les positions projetees
en cm/degres, ce qui reduit le jitter visible sur les objets statiques
(ArUco bruite au pixel -> ~2 cm de jitter sur les bords de table).

Strategie : EMA par label (ex. "NUT_BLUE", "YR1") avec reinitialisation si
le marqueur n'a pas ete vu depuis EMA_RESET_TIMEOUT_S. Les angles sont
lisses en tenant compte du wrap a +/-180 deg.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from marker_detection import config


@dataclass
class _SmoothedEntry:
    x: float
    y: float
    angle: float
    last_update: float


def _angle_ema(prev: float, new: float, alpha: float) -> float:
    """EMA sur angles circulaires: interpole le chemin le plus court."""
    diff = ((new - prev + 180.0) % 360.0) - 180.0
    return (prev + alpha * diff + 180.0) % 360.0 - 180.0


class PoseSmoother:
    """EMA sur (x, y) en cm et sur angle en degres, par label."""

    def __init__(
        self,
        alpha_pos: float = config.EMA_ALPHA_POSITION,
        alpha_angle: float = config.EMA_ALPHA_ANGLE,
        reset_timeout_s: float = config.EMA_RESET_TIMEOUT_S,
    ) -> None:
        self._alpha_pos = alpha_pos
        self._alpha_angle = alpha_angle
        self._reset_timeout = reset_timeout_s
        self._state: dict[str, _SmoothedEntry] = {}

    def smooth(
        self,
        detected: list[tuple[str, float, float, float]],
    ) -> list[tuple[str, float, float, float]]:
        """Applique l'EMA et retourne la liste lissee.

        Args:
            detected: Liste (label, x_cm, y_cm, angle_deg).

        Returns:
            Meme structure, avec (x, y, angle) lisses.
        """
        now = time.monotonic()
        out: list[tuple[str, float, float, float]] = []
        seen_labels: set[str] = set()

        # Un label peut apparaitre plusieurs fois (ex. 2 adversaires avec
        # le meme ID par erreur). On utilise label + index d'occurrence.
        counts: dict[str, int] = {}
        for label, x, y, angle in detected:
            idx = counts.get(label, 0)
            counts[label] = idx + 1
            key = f"{label}#{idx}"
            seen_labels.add(key)

            prev = self._state.get(key)
            if prev is None or (now - prev.last_update) > self._reset_timeout:
                entry = _SmoothedEntry(x=x, y=y, angle=angle, last_update=now)
            else:
                entry = _SmoothedEntry(
                    x=prev.x + self._alpha_pos * (x - prev.x),
                    y=prev.y + self._alpha_pos * (y - prev.y),
                    angle=_angle_ema(prev.angle, angle, self._alpha_angle),
                    last_update=now,
                )
            self._state[key] = entry
            out.append((label, entry.x, entry.y, entry.angle))

        # Purge des entrees perimees pour eviter une fuite memoire.
        stale = [k for k, v in self._state.items()
                 if k not in seen_labels and (now - v.last_update) > self._reset_timeout]
        for k in stale:
            del self._state[k]

        return out

    def reset(self) -> None:
        self._state.clear()
