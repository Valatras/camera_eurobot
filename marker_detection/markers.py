"""Helpers pour classifier et filtrer les marqueurs detectes."""

from __future__ import annotations

import numpy as np

from marker_detection import config
from marker_detection.geometry import to_cell
from marker_detection.esp32_sender import ESP32Sender


def classify_marker_id(marker_id: int) -> str:
    """Mappe un id ArUco vers un type lisible."""
    if marker_id in config.CORNER_IDS:
        return f"TABLE{marker_id}"
    if 1 <= marker_id <= 5:
        return f"BR{marker_id}"
    if 6 <= marker_id <= 10:
        return f"YR{marker_id - 5}"
    if 11 <= marker_id <= 50 and marker_id not in config.CORNER_IDS:
        return f"AREA{marker_id}"
    if 51 <= marker_id <= 70:
        return f"BLUE{marker_id}"
    if 71 <= marker_id <= 90:
        return f"YELLOW{marker_id}"
    return f"ARUCO{marker_id}"


def separate_markers(
    a_ids: list[int],
    a_corners: list[np.ndarray],
) -> tuple[dict[int, np.ndarray], list[tuple[int, np.ndarray]]]:
    """Separe les coins de table des autres ArUco."""
    corners_by_id: dict[int, np.ndarray] = {}
    obj_aruco: list[tuple[int, np.ndarray]] = []

    for marker_id, corner in zip(a_ids, a_corners):
        if marker_id in config.CORNER_IDS:
            corners_by_id[marker_id] = corner
        else:
            obj_aruco.append((marker_id, corner))

    return corners_by_id, obj_aruco


def _build_detected_list(
    corners_by_id: dict[int, np.ndarray],
    obj_aruco: list[tuple[int, np.ndarray]],
    h_img_to_grid: np.ndarray | None,
) -> list[tuple[str, int, int]]:
    """Construit la liste triee des marqueurs detectes en coords grille.

    Returns:
        Liste de tuples (label, grid_x, grid_y).
    """
    detected: list[tuple[str, int, int]] = []

    all_markers = list(corners_by_id.items()) + obj_aruco

    for marker_id, corner in all_markers:
        center = corner[0].mean(axis=0)
        pos = to_cell(center[0], center[1], h_img_to_grid)

        if pos is None:
            continue

        gx = int(round(pos[0]))
        gy = int(round(pos[1]))

        detected.append((classify_marker_id(marker_id), gx, gy))

    detected.sort(key=lambda item: item[0])
    return detected


def print_detected_objects(
    corners_by_id: dict[int, np.ndarray],
    obj_aruco: list[tuple[int, np.ndarray]],
    h_img_to_grid: np.ndarray | None,
) -> None:
    """Imprime les objets detectes en coordonnees grille."""
    detected = _build_detected_list(corners_by_id, obj_aruco, h_img_to_grid)

    if detected:
        print(detected)


def send_detected_objects(
    corners_by_id: dict[int, np.ndarray],
    obj_aruco: list[tuple[int, np.ndarray]],
    h_img_to_grid: np.ndarray | None,
    sender: ESP32Sender,
) -> bool:
    """Envoie les objets detectes a un ESP32 via USB serie.

    Chaque marqueur est transmis sur une ligne au format :
        TYPE,X,Y\\n
    Suivi d'une ligne de fin de trame :
        END\\n

    Args:
        corners_by_id: Coins de table detectes, indexes par id.
        obj_aruco:     Autres marqueurs ArUco detectes.
        h_img_to_grid: Homographie image -> grille.
        sender:        Instance d'ESP32Sender deja connectee.

    Returns:
        True si l'envoi a reussi, False sinon.
    """
    detected = _build_detected_list(corners_by_id, obj_aruco, h_img_to_grid)

    if not detected:
        return True  # Rien a envoyer, pas une erreur

    return sender.send_markers(detected)
