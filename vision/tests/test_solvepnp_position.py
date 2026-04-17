"""Tests solvePnP par marqueur : precision de retour 3D.

On simule la projection d'un marqueur noisette (40mm, h=3cm) vu par une
camera dont on connait la pose, puis on verifie que la reconstruction
via `build_detected_list` (chemin solvePnP) retrouve la position
connue sous le cm.
"""

from __future__ import annotations

import cv2
import numpy as np

from marker_detection import config
from marker_detection.markers import build_detected_list


def _synth_frame(K, rvec_cam, tvec_cam, marker_center_mm, marker_size_mm, marker_height_mm):
    """Projette un marqueur synthetique + les 4 coins de table en image."""
    dist = np.zeros((5,), dtype=np.float64)

    # Coins de la table (4 ArUco, 100mm, z=0).
    corners_by_id: dict[int, np.ndarray] = {}
    for cid in sorted(config.CORNER_IDS):
        cx_cm, cy_cm = config.CORNER_REAL_POSITIONS_CM[cid]
        s = 50.0
        obj = np.array([
            [cx_cm * 10 - s, cy_cm * 10 + s, 0.0],
            [cx_cm * 10 + s, cy_cm * 10 + s, 0.0],
            [cx_cm * 10 + s, cy_cm * 10 - s, 0.0],
            [cx_cm * 10 - s, cy_cm * 10 - s, 0.0],
        ], dtype=np.float32)
        proj, _ = cv2.projectPoints(obj, rvec_cam, tvec_cam, K, dist)
        corners_by_id[cid] = proj.reshape(-1, 2).astype(np.float32)

    # Le marqueur noisette (ID 36), centre en marker_center_mm, z = marker_height_mm.
    s = marker_size_mm / 2.0
    cx, cy = marker_center_mm
    obj_marker = np.array([
        [cx - s, cy + s, marker_height_mm],
        [cx + s, cy + s, marker_height_mm],
        [cx + s, cy - s, marker_height_mm],
        [cx - s, cy - s, marker_height_mm],
    ], dtype=np.float32)
    proj_m, _ = cv2.projectPoints(obj_marker, rvec_cam, tvec_cam, K, dist)
    # Shape attendu par le pipeline : (1, 4, 2).
    obj_aruco = [(config.NUT_BLUE_ID, proj_m.reshape(1, 4, 2).astype(np.float32))]

    return corners_by_id, obj_aruco


def _K_default(w=3840, h=2160):
    f = 0.85 * w
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def test_solvepnp_nut_center_of_table():
    # Camera a (1500, 1000, 1360) mm = centre top en table flipped. Axe
    # optique vers le sol (rotation Rx(pi)).
    K = _K_default()
    rvec = np.array([np.pi, 0.0, 0.0], dtype=np.float64)
    tvec = np.array([-1500.0, 1000.0, 1360.0], dtype=np.float64)

    # Noisette a (1500, 1000) mm = centre de table, z = 30 mm.
    marker_xy_mm = (1500.0, 1000.0)
    corners_by_id, obj_aruco = _synth_frame(
        K, rvec, tvec, marker_xy_mm,
        marker_size_mm=config.MARKER_SIZE_NUT_MM,
        marker_height_mm=config.MARKER_HEIGHT_NUT_CM * 10,
    )

    # Pose camera = la verite qu'on a injectee.
    detected = build_detected_list(
        corners_by_id, obj_aruco, h_img_to_cm=None,
        K=K, dist=np.zeros(5),
        camera_rvec=rvec, camera_tvec=tvec,
    )
    # La noisette doit apparaitre avec label NUT_BLUE.
    nut = next((d for d in detected if d[0] == "NUT_BLUE"), None)
    assert nut is not None
    label, x_cm, y_cm, _angle = nut
    # Repere table reel (0 = bord gauche) : centre = (150, 100) cm.
    assert abs(x_cm - 150.0) < 0.5, f"x_cm={x_cm}"
    assert abs(y_cm - 100.0) < 0.5, f"y_cm={y_cm}"



def test_solvepnp_nut_near_table_edge():
    # Bord de table : precision historiquement la plus faible.
    K = _K_default()
    rvec = np.array([np.pi, 0.0, 0.0], dtype=np.float64)
    tvec = np.array([-1500.0, 1000.0, 1360.0], dtype=np.float64)

    marker_xy_mm = (2800.0, 200.0)  # pres d'un coin
    corners_by_id, obj_aruco = _synth_frame(
        K, rvec, tvec, marker_xy_mm,
        marker_size_mm=config.MARKER_SIZE_NUT_MM,
        marker_height_mm=config.MARKER_HEIGHT_NUT_CM * 10,
    )
    detected = build_detected_list(
        corners_by_id, obj_aruco, h_img_to_cm=None,
        K=K, dist=np.zeros(5),
        camera_rvec=rvec, camera_tvec=tvec,
    )
    nut = next((d for d in detected if d[0] == "NUT_BLUE"), None)
    assert nut is not None
    _, x_cm, y_cm, _ = nut
    # Repere table reel (0 = bord gauche) : x = 280 cm, y = 20 cm.
    assert abs(x_cm - 280.0) < 1.0, f"x_cm={x_cm}"
    assert abs(y_cm - 20.0) < 1.0, f"y_cm={y_cm}"
