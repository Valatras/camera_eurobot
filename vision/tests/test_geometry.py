"""Tests elementaires de la geometrie (compensation de hauteur)."""

from __future__ import annotations

from marker_detection.geometry import compensate_height


def test_compensate_height_no_effect_if_object_on_floor():
    x, y = compensate_height(100.0, 50.0, (150.0, 75.0), 0.0, 200.0)
    assert (x, y) == (100.0, 50.0)


def test_compensate_height_shifts_toward_camera_nadir():
    # Camera en (150, 75) a 200 cm, objet a 20 cm de haut vu a (100, 50).
    x, y = compensate_height(100.0, 50.0, (150.0, 75.0), 20.0, 200.0)
    # Le point au sol doit etre plus proche du centre camera.
    assert 100.0 < x < 150.0
    assert 50.0 < y < 75.0


def test_compensate_height_guard_degenerate_case():
    # Si la camera est plus basse que l'objet, on ne corrige pas.
    x, y = compensate_height(100.0, 50.0, (150.0, 75.0), 250.0, 200.0)
    assert (x, y) == (100.0, 50.0)
