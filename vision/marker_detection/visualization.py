"""Fonctions de rendu pour la vue aerienne (stream MJPEG).

La vue aerienne couvre la zone de jeu 300x155 cm (sans grenier/nids).
Overlay minimal : marqueurs avec label, statut en haut.
"""

from __future__ import annotations

import cv2
import numpy as np

from marker_detection import config
from marker_detection.geometry import to_cm
from marker_detection.markers import classify_marker_id, is_opponent, is_ally

# Couleurs par type de marqueur (BGR).
_C_OPPONENT = (0, 0, 230)      # rouge
_C_ALLY = (230, 180, 0)        # bleu clair
_C_NUT_BLUE = (200, 120, 0)    # bleu
_C_NUT_YELLOW = (0, 200, 240)  # jaune
_C_NUT_EMPTY = (180, 180, 180) # gris
_C_CORNER = (0, 200, 200)      # jaune-vert discret
_C_OTHER = (0, 200, 0)         # vert


def _marker_color(marker_id: int) -> tuple[int, int, int]:
    """Couleur d'affichage selon le type de marqueur."""
    if marker_id in config.CORNER_IDS:
        return _C_CORNER
    if is_opponent(marker_id):
        return _C_OPPONENT
    if is_ally(marker_id):
        return _C_ALLY
    if marker_id == config.NUT_BLUE_ID:
        return _C_NUT_BLUE
    if marker_id == config.NUT_YELLOW_ID:
        return _C_NUT_YELLOW
    if marker_id == config.NUT_EMPTY_ID:
        return _C_NUT_EMPTY
    return _C_OTHER


def _marker_label(marker_id: int) -> str:
    """Label court pour l'overlay."""
    if marker_id in config.CORNER_IDS:
        return ""
    if is_opponent(marker_id):
        return "OPP"
    if is_ally(marker_id):
        return "US"
    if marker_id in config.NUT_IDS:
        return "NUT"
    return classify_marker_id(marker_id)


def compute_aerial(frame: np.ndarray, h_aerial: np.ndarray | None) -> np.ndarray | None:
    """Calcule la vue aerienne (zone de jeu 300x155 cm)."""
    if h_aerial is None:
        return None
    return cv2.warpPerspective(frame, h_aerial, (config.AERIAL_W, config.AERIAL_H))


def draw_aerial_overlay(
    aerial: np.ndarray,
    corners_by_id: dict[int, np.ndarray],
    obj_aruco: list[tuple[int, np.ndarray]],
    h_img_to_aerial: np.ndarray,
    h_img_to_cm: np.ndarray | None,
    n_corners: int,
) -> None:
    """Dessine un overlay minimaliste sur la vue aerienne."""
    h, w = aerial.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Bordure de table fine.
    cv2.rectangle(aerial, (0, 0), (w - 1, h - 1), (80, 80, 80), 1)

    # Marqueurs objets.
    for marker_id, corner in obj_aruco:
        color = _marker_color(marker_id)
        label = _marker_label(marker_id)
        pts_cam = corner[0].reshape(-1, 1, 2).astype(np.float32)
        pts_a = cv2.perspectiveTransform(pts_cam, h_img_to_aerial)
        center = pts_a.mean(axis=0)[0]
        cx, cy = int(center[0]), int(center[1])

        # Cercle + label.
        radius = 12 if is_opponent(marker_id) or is_ally(marker_id) else 8
        cv2.circle(aerial, (cx, cy), radius, color, -1)
        cv2.circle(aerial, (cx, cy), radius, (255, 255, 255), 1)

        if label:
            cam_center = corner[0].mean(axis=0)
            cm = to_cm(cam_center[0], cam_center[1], h_img_to_cm)
            if cm:
                txt = f"{label} ({cm[0]:.0f},{cm[1]:.0f})"
            else:
                txt = label
            (tw, _th), _ = cv2.getTextSize(txt, font, 0.35, 1)
            cv2.putText(aerial, txt, (cx - tw // 2, cy - radius - 4),
                        font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # Coins de table (petits cercles discrets).
    for _mid, corner in corners_by_id.items():
        pts_cam = corner[0].reshape(-1, 1, 2).astype(np.float32)
        pts_a = cv2.perspectiveTransform(pts_cam, h_img_to_aerial)
        center = pts_a.mean(axis=0)[0]
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(aerial, (cx, cy), 5, _C_CORNER, -1)
        cv2.circle(aerial, (cx, cy), 5, (255, 255, 255), 1)

    # Statut en haut a gauche (compact).
    calib = "OK" if n_corners >= 3 else f"{n_corners}/3"
    status = f"{n_corners}C  {len(obj_aruco)}obj  calib:{calib}"
    cv2.putText(aerial, status, (6, 16), font, 0.40,
                (200, 200, 200), 1, cv2.LINE_AA)


def draw_fallback_overlay(
    frame: np.ndarray,
    corners_by_id: dict[int, np.ndarray],
    obj_aruco: list[tuple[int, np.ndarray]],
) -> None:
    """Overlay minimal sur la vue perspective (quand calibration echoue)."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Coins detectes : point + id.
    for mid, corner in corners_by_id.items():
        center = corner[0].mean(axis=0).astype(int)
        cv2.circle(frame, tuple(center), 6, _C_CORNER, -1)
        cv2.putText(frame, str(mid), (center[0] + 8, center[1] - 4),
                    font, 0.4, _C_CORNER, 1, cv2.LINE_AA)

    # Objets : contour + id.
    for mid, corner in obj_aruco:
        pts = corner[0].astype(np.int32).reshape(-1, 1, 2)
        color = _marker_color(mid)
        cv2.polylines(frame, [pts], True, color, 2)
        center = corner[0].mean(axis=0).astype(int)
        cv2.putText(frame, str(mid), (center[0] + 6, center[1] - 6),
                    font, 0.4, color, 1, cv2.LINE_AA)

    # Statut : calibration manquee.
    n = len(corners_by_id)
    missing = config.CORNER_IDS - set(corners_by_id)
    status = f"Calibration: {n}/3 coins"
    if missing:
        status += f"  manquants: {sorted(missing)}"
    cv2.putText(frame, status, (8, 22), font, 0.45,
                (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "En attente de 3+ marqueurs de coins...",
                (8, 44), font, 0.40, (0, 140, 255), 1, cv2.LINE_AA)
