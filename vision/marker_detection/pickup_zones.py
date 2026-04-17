"""Agregation des detections de noisettes par zone de ramassage (ZR).

Probleme physique : dans une ZR, 5 noisettes sont empilees/alignees le long
de l'axe long de la zone (axis "x" ou "y"). Seule la noisette du dessus
(ou celle qui n'est pas cachee par une autre) porte un ArUco detectable.
On maintient donc un historique temporel par zone qui reconstruit la
sequence de couleurs au fur et a mesure que les noisettes sont retirees.

Couleurs :
    "B" = bleue (NUT_BLUE)
    "Y" = jaune (NUT_YELLOW)
    "E" = vide / noisette neutre (NUT_EMPTY)
    "?" = inconnue (ArUco manquant ou corrompu)

Sortie destinee au firmware : pour chaque ZR, la liste ordonnee des
couleurs deja observees du "debut" vers la "fin" de la zone (orientation
determinee par `axis`), limitee a ZONE_MAX_SLOTS slots.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from marker_detection import config

COLOR_BLUE = "B"
COLOR_YELLOW = "Y"
COLOR_EMPTY = "E"
COLOR_UNKNOWN = "?"


_LABEL_TO_COLOR = {
    "NUT_BLUE": COLOR_BLUE,
    "NUT_YELLOW": COLOR_YELLOW,
    "NUT_EMPTY": COLOR_EMPTY,
}


def _label_to_color(label: str) -> str:
    return _LABEL_TO_COLOR.get(label, COLOR_UNKNOWN)


def _point_in_zone(x: float, y: float, zone: dict, margin: float) -> bool:
    dx = abs(x - zone["cx_cm"])
    dy = abs(y - zone["cy_cm"])
    return dx <= zone["len_cm"] / 2.0 + margin and dy <= zone["wid_cm"] / 2.0 + margin


def _axis_offset(x: float, y: float, zone: dict) -> float:
    """Position le long de l'axe long de la zone, depuis le centre."""
    if zone["axis"] == "x":
        return x - zone["cx_cm"]
    return y - zone["cy_cm"]


@dataclass
class _Slot:
    """Un emplacement de noisette dans une ZR."""
    offset: float       # position le long de l'axe long (cm, depuis centre)
    color: str
    last_seen: float    # timestamp monotonic


@dataclass
class ZoneState:
    """Etat courant d'une zone de ramassage."""
    zone_id: int
    slots: list[_Slot] = field(default_factory=list)
    last_update: float = 0.0

    def sequence(self, max_slots: int = config.ZONE_MAX_SLOTS) -> list[str]:
        """Sequence ordonnee selon l'axe long (du plus petit offset au plus grand).

        Renvoie toujours max_slots entrees, completees par COLOR_UNKNOWN
        quand le slot n'a jamais ete observe.
        """
        sorted_slots = sorted(self.slots, key=lambda s: s.offset)
        seq = [s.color for s in sorted_slots[:max_slots]]
        while len(seq) < max_slots:
            seq.append(COLOR_UNKNOWN)
        return seq


class PickupZoneTracker:
    """Suit les couleurs observees dans chaque ZR dans le temps.

    Pour chaque noisette detectee, on la rattache a une zone par
    proximite (point-in-rect etendu). Dans la zone, on l'associe au slot
    existant dont l'offset est le plus proche (a <slot_merge_cm>), sinon
    on cree un nouveau slot.
    """

    def __init__(
        self,
        zones: list[dict] = config.PICKUP_ZONES,
        margin: float = config.ZONE_MEMBERSHIP_MARGIN_CM,
        slot_merge_cm: float = 3.0,
        history_ttl_s: float = config.ZONE_HISTORY_TTL_S,
    ) -> None:
        self._zones = zones
        self._margin = margin
        self._slot_merge_cm = slot_merge_cm
        self._history_ttl = history_ttl_s
        self._states: dict[int, ZoneState] = {
            z["id"]: ZoneState(zone_id=z["id"]) for z in zones
        }

    def update(
        self,
        detected: list[tuple[str, float, float, float]],
    ) -> dict[int, ZoneState]:
        """Met a jour l'historique avec les detections courantes.

        Args:
            detected: Liste (label, x_cm, y_cm, angle_deg) des marqueurs
                vus dans la frame (deja lisses par PoseSmoother).

        Returns:
            dict zone_id -> ZoneState (etat mis a jour).
        """
        now = time.monotonic()

        # Purge des slots perimes : le robot a pu passer depuis.
        for state in self._states.values():
            state.slots = [
                s for s in state.slots if (now - s.last_seen) <= self._history_ttl
            ]

        # Classifier chaque noisette detectee.
        for label, x_cm, y_cm, _angle in detected:
            color = _label_to_color(label)
            if color == COLOR_UNKNOWN and not label.startswith("NUT_"):
                continue  # ignorer les non-noisettes

            zone = self._find_zone(x_cm, y_cm)
            if zone is None:
                continue  # noisette hors zone (ex. deja recuperee)

            state = self._states[zone["id"]]
            offset = _axis_offset(x_cm, y_cm, zone)

            merged = False
            for slot in state.slots:
                if abs(slot.offset - offset) <= self._slot_merge_cm:
                    slot.color = color
                    slot.last_seen = now
                    merged = True
                    break

            if not merged and len(state.slots) < config.ZONE_MAX_SLOTS:
                state.slots.append(
                    _Slot(offset=offset, color=color, last_seen=now))

            state.last_update = now

        return self._states

    def _find_zone(self, x_cm: float, y_cm: float) -> dict | None:
        for z in self._zones:
            if _point_in_zone(x_cm, y_cm, z, self._margin):
                return z
        return None

    def snapshot(self) -> dict[int, list[str]]:
        """Sequence actuelle par zone (sans mutation)."""
        return {zid: state.sequence() for zid, state in self._states.items()}


def count_nuts_in_garde_mangers(
    detected: list[tuple[str, float, float, float]],
    garde_mangers: list[dict] = config.GARDE_MANGERS,
) -> dict[int, int]:
    """Compte les noisettes (toutes couleurs confondues) par garde-manger."""
    counts: dict[int, int] = {gm["id"]: 0 for gm in garde_mangers}
    for label, x, y, _ in detected:
        if not label.startswith("NUT_"):
            continue
        for gm in garde_mangers:
            half = gm["size_cm"] / 2.0
            if abs(x - gm["cx_cm"]) <= half and abs(y - gm["cy_cm"]) <= half:
                counts[gm["id"]] += 1
                break
    return counts
