"""Fonctions de rendu pour la vue camera et la vue aerienne."""

from __future__ import annotations

import cv2
import numpy as np

from marker_detection import config
from marker_detection.geometry import to_cell


def draw_grid(frame: np.ndarray, h_grid_to_img: np.ndarray | None) -> None:
    """Dessine la grille de jeu projetee dans l'image camera."""
    if h_grid_to_img is None:
        return

    for x in range(0, config.GRID_COLS + 1, 5):
        line_grid = np.float32([[[x, 0]], [[x, config.GRID_ROWS]]])
        line_img = cv2.perspectiveTransform(line_grid, h_grid_to_img).astype(np.int32)

        if config.ARUCO_OFFSET_COLS <= x <= (config.ARUCO_OFFSET_COLS + config.ARUCO_INNER_COLS):
            color = (0, 255, 0)
        else:
            color = (0, 150, 0)

        cv2.line(frame, tuple(line_img[0, 0]), tuple(line_img[1, 0]), color, 1)

    for y in range(0, config.GRID_ROWS + 1, 5):
        line_grid = np.float32([[[0, y]], [[config.GRID_COLS, y]]])
        line_img = cv2.perspectiveTransform(line_grid, h_grid_to_img).astype(np.int32)

        if config.ARUCO_OFFSET_ROWS <= y <= (config.ARUCO_OFFSET_ROWS + config.ARUCO_INNER_ROWS):
            color = (0, 255, 0)
        else:
            color = (0, 150, 0)

        cv2.line(frame, tuple(line_img[0, 0]), tuple(line_img[1, 0]), color, 1)


def draw_table_outline(frame: np.ndarray, table_pts: np.ndarray | None, aruco_pts: np.ndarray | None) -> None:
    """Dessine les contours de table et les points ArUco retenus."""
    if table_pts is not None:
        cv2.polylines(frame, [table_pts.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 2)

    if aruco_pts is None:
        return

    cv2.polylines(frame, [aruco_pts.astype(np.int32).reshape(-1, 1, 2)], True, (255, 255, 0), 2)

    labels = ["TL(23)", "TR(22)", "BR(20)", "BL(21)"]
    colors = [(255, 0, 255), (255, 128, 0), (0, 255, 255), (128, 255, 0)]
    for pt, label, color in zip(aruco_pts, labels, colors):
        p = tuple(pt.astype(int))
        cv2.circle(frame, p, 8, color, -1)
        cv2.putText(frame, label, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_aerial_grid(aerial: np.ndarray) -> None:
    """Dessine la grille dans la vue aerienne."""
    cell_w = config.AERIAL_W / config.GRID_COLS
    cell_h = config.AERIAL_H / config.GRID_ROWS

    for col in range(0, config.GRID_COLS + 1, 5):
        x = int(col * cell_w)
        color = (0, 255, 0) if config.ARUCO_OFFSET_COLS <= col <= (config.ARUCO_OFFSET_COLS + config.ARUCO_INNER_COLS) else (0, 150, 0)
        cv2.line(aerial, (x, 0), (x, config.AERIAL_H), color, 1)
        if col % 10 == 0:
            cv2.putText(aerial, f"{col}", (x + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    for row in range(0, config.GRID_ROWS + 1, 5):
        y = int(row * cell_h)
        color = (0, 255, 0) if config.ARUCO_OFFSET_ROWS <= row <= (config.ARUCO_OFFSET_ROWS + config.ARUCO_INNER_ROWS) else (0, 150, 0)
        cv2.line(aerial, (0, y), (config.AERIAL_W, y), color, 1)
        if row % 5 == 0:
            cv2.putText(aerial, f"{row}", (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def compute_aerial(frame: np.ndarray, h_aerial: np.ndarray | None, frame_count: int) -> np.ndarray | None:
    """Calcule la vue aerienne."""
    if h_aerial is None or frame_count % 2 != 0:
        return None

    aerial = cv2.warpPerspective(frame, h_aerial, (config.AERIAL_W, config.AERIAL_H))
    draw_aerial_grid(aerial)

    return aerial


def draw_corner_markers(frame: np.ndarray, corners_by_id: dict[int, np.ndarray]) -> None:
    """Dessine les ArUco servant de coins de table."""
    for marker_id, corner in corners_by_id.items():
        draw_detection(frame, corner[0], f"C{marker_id}", (0, 255, 255))


def draw_object_markers(
    frame: np.ndarray,
    aerial: np.ndarray | None,
    obj_aruco: list[tuple[int, np.ndarray]],
    h_img_to_grid: np.ndarray | None,
    h_aerial: np.ndarray | None,
    frame_count: int,
) -> None:
    """Dessine les ArUco objets."""
    for marker_id, corner in obj_aruco:
        center = corner[0].mean(axis=0)
        pos = to_cell(center[0], center[1], h_img_to_grid)

        label = f"A{marker_id}[{pos[0]:.1f},{pos[1]:.1f}]" if pos else f"A{marker_id}"

        draw_detection(frame, corner[0], label, (0, 255, 0))

        if aerial is not None and frame_count % 2 == 0:
            draw_aerial_detection(
                aerial,
                corner[0],
                f"A{marker_id}",
                (0, 255, 0),
                h_aerial,
            )


def draw_qr_codes(
    frame: np.ndarray,
    aerial: np.ndarray | None,
    q_data: list[str],
    q_corners: list[np.ndarray],
    h_img_to_grid: np.ndarray | None,
    h_aerial: np.ndarray | None,
    frame_count: int,
) -> None:
    """Dessine les QR codes detectes."""
    for data, corners in zip(q_data, q_corners):
        center = corners.mean(axis=0)
        pos = to_cell(center[0], center[1], h_img_to_grid)

        short = data[:10] + "..." if len(data) > 10 else data
        label = f"QR:{short}[{pos[0]:.1f},{pos[1]:.1f}]" if pos else f"QR:{short}"

        draw_detection(frame, corners, label, (0, 165, 255))

        if aerial is not None and frame_count % 2 == 0:
            draw_aerial_detection(
                aerial,
                corners,
                f"QR:{short}",
                (0, 165, 255),
                h_aerial,
            )


def draw_status(
    frame: np.ndarray,
    corners_by_id: dict[int, np.ndarray],
    obj_aruco: list[tuple[int, np.ndarray]],
    q_data: list[str],
    h_img_to_grid: np.ndarray | None,
) -> None:
    """Dessine les informations de statut."""
    n_corners = len(corners_by_id)
    missing = config.CORNER_IDS - set(corners_by_id)
    grid_ok = h_img_to_grid is not None

    cv2.putText(
        frame,
        f"C:{n_corners}/4 Obj:{len(obj_aruco) + len(q_data)} Grid:{'OK' if grid_ok else '...'}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if grid_ok else (0, 165, 255),
        2,
    )

    if missing:
        cv2.putText(
            frame,
            f"Missing: {sorted(missing)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )


def draw_detection(frame: np.ndarray, pts_raw: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    """Dessine un polygone detecte et son libelle sur une image."""
    pts = pts_raw.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], True, color, 2)
    center = pts_raw.mean(axis=0).astype(int)
    cv2.putText(frame, label, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)


def draw_aerial_detection(
    aerial: np.ndarray,
    pts_camera: np.ndarray,
    label: str,
    color: tuple[int, int, int],
    h_img_to_aerial: np.ndarray | None,
) -> None:
    """Projette et dessine une detection camera sur la vue aerienne."""
    if h_img_to_aerial is None:
        return

    pts_aerial = cv2.perspectiveTransform(pts_camera.reshape(-1, 1, 2).astype(np.float32), h_img_to_aerial)
    pts_int = pts_aerial.astype(np.int32)

    cv2.polylines(aerial, [pts_int], True, color, 2)
    center = pts_aerial.mean(axis=0)[0].astype(int)
    cv2.putText(aerial, label, (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
