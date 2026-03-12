"""Geometrie de la table: selection de coins et matrices de perspective."""

from __future__ import annotations

import cv2
import numpy as np

from marker_detection import config

# top-left, top-right, bottom-right, bottom-left
def select_table_corner_point(marker_pts: np.ndarray, position: str) -> np.ndarray:
    """Choisit le coin du marqueur qui correspond au coin reel de table."""
    if position == "TL":
        idx = int(np.argmin(marker_pts[:, 0] + marker_pts[:, 1]))
    elif position == "TR":
        idx = int(np.argmin(marker_pts[:, 1] - marker_pts[:, 0]))
    elif position == "BR":
        idx = int(np.argmax(marker_pts[:, 0] + marker_pts[:, 1]))
    else:  # BL
        idx = int(np.argmax(marker_pts[:, 1] - marker_pts[:, 0]))

    return marker_pts[idx]


def select_table_points(corners_by_id: dict[int, np.ndarray]) -> np.ndarray | None:
    """Selectionne les 4 points ArUco utilises comme base d'extrapolation."""
    if not all(mid in corners_by_id for mid in config.CORNER_ORDER):
        return None

    positions = ["TL", "TR", "BR", "BL"]
    aruco_pts: list[np.ndarray] = []
    for mid, pos in zip(config.CORNER_ORDER, positions):
        marker_pts = corners_by_id[mid][0]
        aruco_pts.append(select_table_corner_point(marker_pts, pos))

    points = np.array(aruco_pts, dtype=np.float32)
    if abs(cv2.contourArea(points)) < 1000:
        return None

    return points


def extrapolate_table_corners(aruco_pts: np.ndarray) -> np.ndarray:
    """Extrapole les 4 coins reels de table depuis la zone ArUco interieure."""
    top_vec = aruco_pts[1] - aruco_pts[0]
    right_vec = aruco_pts[2] - aruco_pts[1]
    bottom_vec = aruco_pts[2] - aruco_pts[3]
    left_vec = aruco_pts[3] - aruco_pts[0]

    width_ratio = config.TABLE_W_MM / (config.TABLE_W_MM - 2 * config.ARUCO_INSET_MM)
    height_ratio = config.TABLE_H_MM / (config.TABLE_H_MM - 2 * config.ARUCO_INSET_MM)

    width_ext = (width_ratio - 1) / 2
    height_ext = (height_ratio - 1) / 2

    table_tl = aruco_pts[0] - width_ext * top_vec - height_ext * left_vec
    table_tr = aruco_pts[1] + width_ext * top_vec - height_ext * right_vec
    table_br = aruco_pts[2] + width_ext * bottom_vec + height_ext * right_vec
    table_bl = aruco_pts[3] - width_ext * bottom_vec + height_ext * left_vec

    return np.array([table_tl, table_tr, table_br, table_bl], dtype=np.float32)


def build_transforms(
    corners_by_id: dict[int, np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Construit les matrices de perspective image<->grille et image->vue aerienne."""
    aruco_pts = select_table_points(corners_by_id)
    if aruco_pts is None:
        return None, None, None, None, None

    table_pts = extrapolate_table_corners(aruco_pts)

    grid_pts = np.float32(
        [[0.0, 0.0], [config.GRID_COLS, 0.0], [config.GRID_COLS, config.GRID_ROWS], [0.0, config.GRID_ROWS]]
    )
    aerial_pts = np.float32(
        [[0.0, 0.0], [config.AERIAL_W, 0.0], [config.AERIAL_W, config.AERIAL_H], [0.0, config.AERIAL_H]]
    )

    h_img_to_grid = cv2.getPerspectiveTransform(table_pts, grid_pts)
    h_grid_to_img = cv2.getPerspectiveTransform(grid_pts, table_pts)
    h_img_to_aerial = cv2.getPerspectiveTransform(table_pts, aerial_pts)

    return h_img_to_grid, h_grid_to_img, h_img_to_aerial, table_pts, aruco_pts


def to_cell(px: float, py: float, h_img_to_grid: np.ndarray | None) -> tuple[float, float] | None:
    """Projette un point image dans le repere grille (colonne, ligne)."""
    if h_img_to_grid is None:
        return None

    out = cv2.perspectiveTransform(np.float32([[[px, py]]]), h_img_to_grid)
    gx, gy = float(out[0, 0, 0]), float(out[0, 0, 1])
    gx = config.GRID_COLS - gx  # Inverser l'axe horizontal pour top-right = (0,0)

    if gx < -0.5 or gx > config.GRID_COLS + 0.5 or gy < -0.5 or gy > config.GRID_ROWS + 0.5:
        return None

    return gx, gy
