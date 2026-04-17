"""Tests de l'aggregation de noisettes par zone de ramassage."""

from __future__ import annotations

from marker_detection.pickup_zones import (
    PickupZoneTracker,
    count_nuts_in_garde_mangers,
)
from marker_detection import config


def test_nut_inside_zone_is_tracked():
    t = PickupZoneTracker()
    # Zone 1 centree en (17.5, 120), axe y, 20x15 cm.
    # On ajoute une noisette bleue a l'interieur.
    t.update([("NUT_BLUE", 17.5, 120.0, 0.0)])
    states = t.update([("NUT_BLUE", 17.5, 120.0, 0.0)])
    seq = states[1].sequence()
    assert seq[0] == "B"
    # Les autres slots non observes restent "?".
    assert all(s in ("B", "?") for s in seq)
    assert len(seq) == config.ZONE_MAX_SLOTS


def test_nut_outside_zone_ignored():
    t = PickupZoneTracker()
    # Point tres loin de toutes les zones.
    t.update([("NUT_BLUE", 500.0, 500.0, 0.0)])
    for zid, state in t._states.items():  # noqa: SLF001 (test internal)
        assert state.slots == []


def test_distinct_offsets_create_distinct_slots():
    t = PickupZoneTracker(slot_merge_cm=1.0)
    # Zone 1 : axe y, donc deux noisettes en (17.5, 115) et (17.5, 125)
    # ont des offsets differents.
    t.update([
        ("NUT_BLUE", 17.5, 115.0, 0.0),
        ("NUT_YELLOW", 17.5, 125.0, 0.0),
    ])
    seq = t._states[1].sequence()  # noqa: SLF001
    assert seq[0] == "B"  # offset -5
    assert seq[1] == "Y"  # offset +5


def test_gm_count_inside_boundary():
    detected = [
        ("NUT_BLUE", 125.0, 145.0, 0.0),   # GM 1 centre
        ("NUT_YELLOW", 130.0, 150.0, 0.0),  # GM 1 dans le carre
        ("NUT_BLUE", 500.0, 500.0, 0.0),    # hors de tout GM
    ]
    counts = count_nuts_in_garde_mangers(detected)
    assert counts[1] == 2
    # Les autres GM doivent rester a 0.
    for gid, c in counts.items():
        if gid != 1:
            assert c == 0
