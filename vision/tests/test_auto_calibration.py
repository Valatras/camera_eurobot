"""Tests de l'auto-calibration intrinseque sans damier.

On genere des frames synthetiques en projetant les 4 coins ArUco connus
(positions 3D en mm dans le repere table) via une matrice K choisie et
sans distorsion. L'auto-calibration doit retrouver une K proche et une
pose coherente.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from marker_detection import config
from marker_detection.auto_calibration import (
    AutoCalibrator,
    corner_object_points_table,
)


def _project_corners(K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> dict:
    """Retourne un dict marker_id -> corners (4, 2) image."""
    dist = np.zeros((5,), dtype=np.float64)
    corners_by_id: dict[int, np.ndarray] = {}
    for cid in sorted(config.CORNER_IDS):
        obj = corner_object_points_table(cid)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        corners_by_id[cid] = proj.reshape(-1, 2).astype(np.float32)
    return corners_by_id


def test_autocalib_converges_on_fixed_camera():
    # Camera fixe (cas degenere) : Zhang ne convergera pas, mais le
    # fallback "distortion nulle" avec K par defaut doit tenir le RMS.
    w, h = 3840, 2160
    # K "cible" : focale = 0.85*w, centre = milieu image.
    f = 0.85 * w
    K_true = np.array(
        [[f, 0, w / 2],
         [0, f, h / 2],
         [0, 0, 1]], dtype=np.float64,
    )
    # Camera a hauteur 136 cm = 1360 mm au-dessus du centre de table (1500, 1000).
    # Rotation : axe optique vers le bas -> R = Rx(pi).
    rvec = np.array([np.pi, 0.0, 0.0], dtype=np.float64)
    tvec = np.array([-1500.0, 1000.0, 1360.0], dtype=np.float64)

    calib = AutoCalibrator(image_size=(w, h), min_frames=10, timeout_s=60.0)
    for _ in range(15):
        cbi = _project_corners(K_true, rvec, tvec)
        accepted = calib.feed(cbi)
        assert accepted

    intr = calib.calibrate()
    assert intr is not None
    # Image size conservee.
    assert intr.image_size == (w, h)
    # K recuperee proche de la verite (au pire +/- 5% grace au fallback).
    assert abs(intr.camera_matrix[0, 0] - f) / f < 0.1
    assert abs(intr.camera_matrix[0, 2] - w / 2) < w * 0.05
    assert abs(intr.camera_matrix[1, 2] - h / 2) < h * 0.05


def test_autocalib_rejects_insufficient_frames():
    calib = AutoCalibrator(image_size=(3840, 2160), min_frames=10, timeout_s=60.0)
    # Pas assez de frames.
    intr = calib.calibrate()
    assert intr is None


def test_autocalib_rejects_frame_without_all_corners():
    calib = AutoCalibrator(image_size=(3840, 2160))
    # Seulement 2 coins : doit etre rejete.
    cbi = {20: np.zeros((4, 2), dtype=np.float32),
           21: np.zeros((4, 2), dtype=np.float32)}
    assert calib.feed(cbi) is False
    assert calib.n_frames == 0


def test_corner_object_points_are_consistent_with_config():
    # Les 4 coins d'un ArUco de table doivent etre centres sur la
    # position connue avec +/- 50 mm de cote (100 mm / 2).
    pts = corner_object_points_table(20)
    cx_mm, cy_mm = (c * 10 for c in config.CORNER_REAL_POSITIONS_CM[20])
    assert pts.shape == (4, 3)
    assert np.allclose(pts.mean(axis=0), [cx_mm, cy_mm, 0.0])
    # Cote : max(x) - min(x) = 100 mm.
    assert abs(pts[:, 0].max() - pts[:, 0].min() - 100.0) < 1e-3
    assert abs(pts[:, 1].max() - pts[:, 1].min() - 100.0) < 1e-3
